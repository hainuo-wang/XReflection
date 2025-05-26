import lightning as L
import torch
import os
from os import path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from xreflection.utils.registry import MODEL_REGISTRY
from xreflection.metrics import calculate_metric
from xreflection.utils import imwrite, tensor2img
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info, rank_zero_warn
from torchmetrics import MetricCollection


@MODEL_REGISTRY.register()
class BaseModel(L.LightningModule):
    """Classification Module for reflection removal using PyTorch Lightning.
    
    This module implements a classification-based approach for single image reflection removal.
    It supports progressive multi-scale image processing, EMA model updates,
    and extensive validation metrics.
    """

    def __init__(self, opt):
        """Initialize the ClsModel.
        
        Args:
            opt (dict): Configuration options.
        """
        from xreflection.archs import build_network
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.is_train = opt['is_train']

        # Define network
        self.net_g = build_network(opt['network_g'])

        self.current_val_metrics = {}

        # Flag to indicate if using EMA - will be set by EMACallback
        self.use_ema = False

    def setup(self, stage: Optional[str] = None):
        """Setup module based on stage.
        
        Args:
            stage (str, optional): 'fit', 'validate', 'test', or 'predict'
        """
        self.print_network()

        # Load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_weights(load_path)

        if stage == 'fit' or stage is None:
            self.setup_losses()
        
        # 初始化数据集名称映射
        self.val_dataset_names = {}

    @rank_zero_only
    def print_network(self):
        """Print network information"""
        net_params = sum(map(lambda x: x.numel(), self.net_g.parameters()))
        rank_zero_info(f'Network: {self.net_g.__class__.__name__}, with parameters: {net_params:,d}')

    def load_weights(self, load_path):
        """Load pretrained weights.
        
        Args:
            load_path (str): Path to the checkpoint file.
        """
        param_key = self.opt['path'].get('param_key_g', 'params')
        strict_load = self.opt['path'].get('strict_load_g', True)

        if self.trainer is None or self.trainer.global_rank == 0:
            rank_zero_info(f'Loading weights from {load_path} with param key: [{param_key}]')

        # Load weights
        checkpoint = torch.load(load_path, map_location='cpu')

        # Check available keys in checkpoint for better debugging
        if self.trainer is None or self.trainer.global_rank == 0:
            if isinstance(checkpoint, dict):
                rank_zero_info(f"Available keys in checkpoint: {list(checkpoint.keys())}")

        # Try to load with specified param_key, then fallback to alternatives
        if param_key in checkpoint:
            weights = checkpoint[param_key]
            if self.trainer is None or self.trainer.global_rank == 0:
                rank_zero_info(f"Successfully loaded weights using key '{param_key}'")
        elif 'params_ema' in checkpoint and param_key != 'params_ema':
            weights = checkpoint['params_ema']
            if self.trainer is None or self.trainer.global_rank == 0:
                rank_zero_info(f"Key '{param_key}' not found, using 'params_ema' instead")
        elif 'params' in checkpoint and param_key != 'params':
            weights = checkpoint['params']
            if self.trainer is None or self.trainer.global_rank == 0:
                rank_zero_info(f"Key '{param_key}' not found, using 'params' instead")
        else:
            # If no recognized keys, use the entire checkpoint
            weights = checkpoint
            if self.trainer is None or self.trainer.global_rank == 0:
                rank_zero_info(f"No recognized parameter keys found, using entire checkpoint")

        # Remove the prefix for torch.compile
        for k, v in list(weights.items()):
            if k.startswith('_orig_mod.'):
                weights[k[10:]] = weights.pop(k)

        # Remove unnecessary 'module.' prefix
        for k, v in list(weights.items()):
            if k.startswith('module.'):
                weights[k[7:]] = weights.pop(k)

        # Load to model
        self._print_different_keys_loading(self.net_g, weights, strict_load)
        self.net_g.load_state_dict(weights, strict=strict_load)

    @rank_zero_only
    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print key differences when loading models.
        
        Args:
            crt_net (nn.Module): Current network.
            load_net (dict): Loaded network state dict.
            strict (bool): Whether to strictly enforce parameter shapes.
        """
        # Get network state dict
        crt_net_keys = set(crt_net.state_dict().keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            rank_zero_warn('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                rank_zero_warn(f'  {v}')
            rank_zero_warn('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                rank_zero_warn(f'  {v}')

        # Check sizes of the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net.state_dict()[k].size() != load_net[k].size():
                    rank_zero_warn(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net.state_dict()[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def setup_losses(self):
        pass

    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            tuple: Classification outputs and image outputs.
        """
        return self.net_g(x)

    def training_step(self, batch, batch_idx):
        pass
    
    def testing(self, inp):
        if self.use_ema:
            model = self.ema_model
        else:
            model = self.net_g
        with torch.no_grad():
            x_cls_out, x_img_out = model(inp)
            output_clean, output_reflection = x_img_out[-1][:, :3, ...], x_img_out[-1][:, 3:, ...]
            self.output = [output_clean, output_reflection]
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """验证步骤。
        
        Args:
            batch (dict): 输入批次。
            batch_idx (int): 批次索引。
            dataloader_idx (int, optional): 数据加载器索引，用于多个验证集。
            
        Returns:
            dict: 包含清晰图像和反射图像的输出字典。
        """
        # 获取当前验证数据集的名称
        dataset_name = self.val_dataset_names[dataloader_idx]
        
        # 验证批次是否包含所需字段
        required_keys = ['input']
        for key in required_keys:
            if key not in batch:
                rank_zero_warn(f"Required key '{key}' missing from batch during validation")
                return {'error': f"Missing required key: {key}"}

        # 保存输入图像信息
        inp = batch['input']
        self.testing(inp)
        output_clean, output_reflection = self.output

        # 优雅地处理缺失的inp_path
        if 'inp_path' in batch and len(batch['inp_path']) > 0:
            img_name = osp.splitext(osp.basename(batch['inp_path'][0]))[0]
        else:
            # 如果缺少inp_path，生成一个后备名称
            img_name = f"sample_{batch_idx}"
            rank_zero_warn(f"'inp_path' key missing in batch, using fallback name: {img_name}")

        # 处理图像用于指标计算和可视化
        clean_img = tensor2img([output_clean])
        reflection_img = tensor2img([output_reflection])
        target_t_img = tensor2img([batch['target_t']])

        metric_data = {'img': clean_img, 'img2': target_t_img}


        # 保存验证图像
        if self.trainer.is_global_zero and self.opt['val'].get('save_img', False):
            try:
                current_iter = self.current_epoch if self.trainer.sanity_checking else self.global_step

                # 创建保存图像的路径
                if self.opt['is_train']:
                    # 在训练过程中保存图像，添加数据集名称到路径
                    save_dir = osp.join(self.opt['path']['visualization'], dataset_name, img_name)
                    os.makedirs(save_dir, exist_ok=True)
                    save_clean_img_path = osp.join(save_dir, f'{img_name}_clean_{current_iter}.png')
                    save_reflection_img_path = osp.join(save_dir, f'{img_name}_reflection_{current_iter}.png')
                else:
                    # 在测试过程中保存图像
                    save_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    os.makedirs(save_dir, exist_ok=True)
                    if self.opt['val'].get('suffix'):
                        save_clean_img_path = osp.join(save_dir, f'{img_name}_clean_{self.opt["val"]["suffix"]}.png')
                        save_reflection_img_path = osp.join(save_dir, f'{img_name}_reflection_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_clean_img_path = osp.join(save_dir, f'{img_name}_clean_{self.opt["name"]}.png')
                        save_reflection_img_path = osp.join(save_dir, f'{img_name}_reflection_{self.opt["name"]}.png')

                # 保存图像
                imwrite(clean_img, save_clean_img_path)
                imwrite(reflection_img, save_reflection_img_path)
            except Exception as e:
                rank_zero_warn(f"Error saving validation images: {str(e)}")

        # 计算指标
        if 'img2' in metric_data and self.opt['val'].get('metrics') is not None:
            for name, opt_ in self.opt['val']['metrics'].items():
                try:
                    metric_value = calculate_metric(metric_data, opt_)
                    # 使用数据集命名空间记录指标
                    # self.log(f'{dataset_name}/{name}', metric_value, sync_dist=True, add_dataloader_idx=False)
                    # 存储以供后续聚合
                    if dataset_name not in self.current_val_metrics:
                        self.current_val_metrics[dataset_name] = {}
                    if name not in self.current_val_metrics[dataset_name]:
                        self.current_val_metrics[dataset_name][name] = []
                    self.current_val_metrics[dataset_name][name].append(metric_value)
                except Exception as e:
                    rank_zero_warn(f"Error calculating metric '{name}': {str(e)}") 

        return {
            'output_clean': output_clean,
            'output_reflection': output_reflection,
            'img_name': img_name,
            'dataset_name': dataset_name
        }

    def on_validation_epoch_start(self):
        """Setup metrics collection at the start of validation epoch."""
        self.current_val_metrics = {}
        
        # 获取验证数据集名称，用于后续记录和显示
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'val_dataloaders'):
            if isinstance(self.trainer.val_dataloaders, list):
                for idx, loader in enumerate(self.trainer.val_dataloaders):
                    if hasattr(loader.dataset, 'opt') and 'name' in loader.dataset.opt:
                        dataset_name = loader.dataset.opt['name']
                    else:
                        dataset_name = f"val_{idx}"
                    self.val_dataset_names[idx] = dataset_name
            else:
                # 单个验证集情况
                loader = self.trainer.val_dataloaders
                if hasattr(loader.dataset, 'opt') and 'name' in loader.dataset.opt:
                    dataset_name = loader.dataset.opt['name']
                else:
                    dataset_name = "val"
                self.val_dataset_names[0] = dataset_name

    def on_validation_epoch_end(self):
        """Operations at the end of validation epoch."""
        # Calculate and log average metrics across all validation samples
        if self.current_val_metrics:
            total_average_metrics = {}
            for dataset_name, metrics in self.current_val_metrics.items():
                log_str = f'Validation [{dataset_name}] Epoch {self.current_epoch}\n'

                for metric_name, values in metrics.items():
                    
                    avg_value = sum(values) / len(values)
                    log_str += f'\t # {metric_name}: {avg_value:.4f}'

                    # Log to tensorboard 
                    self.logger.experiment.add_scalar(
                        f'metrics/{dataset_name}/{metric_name}', avg_value, self.current_epoch
                    )
                    if metric_name not in total_average_metrics.keys():
                        total_average_metrics[metric_name] = {
                            'val': sum(values),
                            'counts' : len(values)
                        }
                    else:
                        total_average_metrics[metric_name]['val'] += sum(values)
                        total_average_metrics[metric_name]['counts'] += len(values)
                # Log to console
                rank_zero_info(log_str)
            total_average_metrics = {
                k: v['val'] / v['counts'] for k, v in total_average_metrics.items()
            }
            for metric_name, metric_value in total_average_metrics.items():
                self.logger.experiment.add_scalar(
                    f'metrics/average/{metric_name}', metric_value, self.current_epoch
                )
            
            log_str = f'Validation Epoch {self.current_epoch} Average Metrics:\n'
            for metric_name, metric_value in total_average_metrics.items():
                log_str += f'\t # {metric_name}: {metric_value:.4f}\n'
                self.log(f'metrics/average/{metric_name}', metric_value, sync_dist=True)
                
            rank_zero_info(log_str)

    def test_step(self, batch, batch_idx):
        """Test step.
        
        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
            
        Returns:
            dict: Output dict with clean and reflection images.
        """
        return self.validation_step(batch, batch_idx)

    def configure_optimizer_params(self):
        """Configure optimizer parameters.
        
        Returns:
            list: List of parameter groups.
        """
        pass

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            dict: Optimizer and scheduler configuration.
        """
        train_opt = self.opt['train']
        optimizer = self.get_optimizer(**self.configure_optimizer_params())

        # Setup learning rate scheduler without modifying original config
        scheduler_type = train_opt['scheduler']['type']
        scheduler_config = {k: v for k, v in train_opt['scheduler'].items()
                            if k != 'type'}

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
        elif scheduler_type == 'CosineAnnealingRestartLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

        # Get the monitor metric from checkpoint config if available
        monitor_metric = self.opt.get('checkpoint', {}).get('monitor', 'val/psnr')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor_metric,  # Use the same metric as checkpoint monitor
                "interval": "epoch",
                "frequency": 1
            }
        }

    def get_optimizer(self, optim_type, params, **kwargs):
        """Get optimizer based on type.
        
        Args:
            optim_type (str): Optimizer type.
            params (list): Parameter groups.
            **kwargs: Additional optimizer arguments.
            
        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def get_current_visuals(self):
        """获取当前可视化用于展示和比较的张量。
        
        Returns:
            OrderedDict: 当前可视张量的字典。
        """
        out_dict = OrderedDict()
        
        # 训练可视化
        if hasattr(self, 'last_inp'):
            out_dict['inp'] = self.last_inp.detach().cpu()
        if hasattr(self, 'last_output_clean'):
            out_dict['result_clean'] = self.last_output_clean.detach().cpu()
        if hasattr(self, 'last_output_reflection'):
            out_dict['result_reflection'] = self.last_output_reflection.detach().cpu()
        if hasattr(self, 'last_target_t'):
            out_dict['target_t'] = self.last_target_t.detach().cpu()

        # 添加所有验证数据集的可视化结果
        if hasattr(self, 'val_visualization'):
            for dataset_name, visuals in self.val_visualization.items():
                # 为每个数据集添加前缀，避免名称冲突
                prefix = dataset_name.replace(' ', '_')
                
                # 添加该数据集的所有可视化结果
                out_dict[f'{prefix}_inp'] = visuals['inp'].detach().cpu()
                out_dict[f'{prefix}_result_clean'] = visuals['output_clean'].detach().cpu()
                out_dict[f'{prefix}_result_reflection'] = visuals['output_reflection'].detach().cpu()
                if 'target_t' in visuals:
                    out_dict[f'{prefix}_target_t'] = visuals['target_t'].detach().cpu()

        return out_dict
