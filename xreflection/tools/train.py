#!/usr/bin/env python
import os
import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from pprint import pprint
from datetime import datetime
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.plugins import BitsandbytesPrecision
from lightning.pytorch.utilities import rank_zero_info

from xreflection.models import build_model
from xreflection.utils.ema_callback import EMACallback
from xreflection.utils.lr_warmup_callback import LRWarmupCallback
from xreflection.data import build_dataloader, build_dataset


def parse_args():
    """Parse command-line arguments.
    
    The main configuration is loaded from a YAML file specified by --config.
    Only essential arguments that might need frequent changes are exposed as 
    command-line arguments, while most settings are in the config file.
    Users can override any config setting using --override.
    """
    parser = argparse.ArgumentParser(description='Train or test a reflection removal model using Lightning')
    
    # Essential arguments
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--seed', type=int, default=None, help='Random seed, overrides config if provided')
    
    # Optional overrides for quick testing/debugging without editing config
    parser.add_argument('--test_only', type=str, default=None, help='Only test the model, overrides config')
    parser.add_argument('--resume', nargs='?', const='_FIND_LAST_', default=None, 
                        help='Resume from checkpoint, overrides config if provided. If used without a path (e.g., --resume), it will try to find and load "last.ckpt".'
                             'If used with a path (e.g., --resume path/to/ckpt), it will load the specified checkpoint.')
    
    # General override mechanism - allows overriding any config setting from command line
    parser.add_argument('--override', nargs='+', default=[], 
                      help='Override config options, format: key=value pairs (can specify multiple)')
    
    args = parser.parse_args()
    return args


def setup_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path):
    """Load config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_config_overrides(config, args):
    """Process command-line overrides and update config accordingly.
    
    Args:
        config (dict): Configuration dictionary loaded from YAML
        args (argparse.Namespace): Command-line arguments
        
    Returns:
        dict: Updated configuration
    """
    # Handle specific overrides
    if args.seed is not None:
        config['manual_seed'] = args.seed
    
    if args.test_only:
        config['test_only'] = True
    
    # Process general overrides
    for override in args.override:
        if '=' not in override:
            print(f"Warning: Ignoring malformed override '{override}'. Format should be key=value")
            continue
            
        key, value = override.split('=', 1)
        
        # Try to convert value to appropriate type
        try:
            # Try to evaluate as literal (handles integers, floats, booleans, None)
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If not a literal, keep as string
            pass
            
        # Update nested configuration using dot notation (e.g., 'train.optim_g.lr')
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
        
        print(f"Override applied: {key} = {value}")
        
    return config


def create_datamodule(config):
    """Create Lightning DataModule from config"""
    class ReflectionDataModule(L.LightningDataModule):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.val_datasets = []
            
        def setup(self, stage=None):
            # Build datasets
            if stage == 'fit' or stage is None:
                train_config = self.config['datasets']['train']
                train_config['phase'] = 'train'
                self.train_dataset = build_dataset(train_config)
                
                # 创建所有验证数据集
                self.val_datasets = []
                for val_idx, val_config in enumerate(self.config['datasets']['val_datasets']):
                    val_config['phase'] = 'val'
                    val_dataset = build_dataset(val_config)
                    self.val_datasets.append(val_dataset)
            
            if stage == 'test' or stage is None:
                self.test_datasets = []
                # Use validation set configurations for testing as per user request
                for val_config_original in self.config['datasets']['val_datasets']:
                    # Create a copy to avoid modifying the original val_config
                    test_config = val_config_original.copy()
                    test_config['phase'] = 'test' # Set phase to test
                    test_dataset = build_dataset(test_config)
                    self.test_datasets.append(test_dataset)
            
        def train_dataloader(self):
            return build_dataloader(self.train_dataset, self.config['datasets']['train'], seed=self.config['manual_seed'])
            
        def val_dataloader(self):
            val_loaders = []
            for val_idx, val_dataset in enumerate(self.val_datasets):
                val_config = self.config['datasets']['val_datasets'][val_idx]
                val_loaders.append(build_dataloader(val_dataset, val_config))
            
            return val_loaders

        def test_dataloader(self):
            test_loaders = []
            for test_idx, test_dataset in enumerate(self.test_datasets):
                test_config = self.config['datasets']['val_datasets'][test_idx]
                test_config['phase'] = 'test'
                test_loaders.append(build_dataloader(test_dataset, test_config))
            
            return test_loaders
        
    return ReflectionDataModule(config)


def create_callbacks(config):
    """Create Lightning callbacks from config"""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_config = config.get('checkpoint', {})
    monitor = checkpoint_config.get('monitor', 'val/psnr')
    mode = checkpoint_config.get('mode', 'max')  # 'max' for metrics like PSNR, 'min' for loss
    save_top_k = checkpoint_config.get('save_top_k', 3)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['path']['experiments_root'], config['name'], 'checkpoints'),
        filename='{epoch}-{step}-{' + monitor + ':.4f}',
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        save_on_train_epoch_end=False,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process overrides
    config = process_config_overrides(config, args)
    
    # Set random seed
    setup_random_seed(config['manual_seed'])
    
    # Print configuration
    # Will be printed later only on main process within the trainer
    
    # Define paths
    exp_dir = os.path.join(config['path']['experiments_root'], config['name'])
    vis_dir = os.path.join(exp_dir, 'visualization')
    
    # Resume from checkpoint if specified
    resume_path = config['path'].get('resume_state', None)
    if args.resume is not None:
        if args.resume == '_FIND_LAST_':
            resume_path = os.path.join(exp_dir, 'checkpoints', 'last.ckpt')
        else:
            resume_path = args.resume
    
    # Create directories only on main process (rank 0)
    # Check if this is the main process using Lightning's utility
    is_main_process = L.fabric.utilities.rank_zero.rank_zero_only.rank == 0
    
    if is_main_process:
        if os.path.exists(exp_dir) and resume_path is None:
            os.rename(f"{exp_dir}", f"{exp_dir}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        # Create experiment directory
        os.makedirs(exp_dir, exist_ok=True)
        # Create visualization directory
        os.makedirs(vis_dir, exist_ok=True)
    
    # Update paths in config for all processes
    config['path']['visualization'] = vis_dir
    config['path']['log'] = os.path.join(exp_dir, 'logs')
    
    # Create data module
    datamodule = create_datamodule(config)
    
    # Create model
    model = build_model(config)
    
    # Create loggers
    logger_list = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config['path']['log'],
        name=config['name'],
        default_hp_metric=False
    )
    logger_list.append(tb_logger)
    
    # Weights & Biases logger
    wandb_config = config['logger'].get('wandb', {})
    if wandb_config.get('enable', False):
        wandb_logger = WandbLogger(
            name=config.get('name'),
            project=wandb_config.get('project', 'xreflection'),
            entity=wandb_config.get('entity', None),
            save_dir=exp_dir,
            log_model=wandb_config.get('log_model', False),
            tags=wandb_config.get('tags', None),
            notes=wandb_config.get('notes', None),
        )
        logger_list.append(wandb_logger)
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Create trainer
    trainer_kwargs = {
        'accelerator': config.get('accelerator', 'auto'),
        'devices': config.get('devices', 'auto'),
        'precision': config.get('precision', '32-true'),
        'logger': logger_list,
        'callbacks': callbacks,
        'log_every_n_steps': config.get('log_every_n_steps', 50),
        'max_epochs': config['lightning'].get('max_epochs', 100),
        'val_check_interval': config.get('val_check_interval', 1.0),
        'gradient_clip_val': config['lightning'].get('gradient_clip_val', 0),
        'accumulate_grad_batches': config['lightning'].get('accumulate_grad_batches', 1),
        'deterministic': config['lightning'].get('deterministic', False),
        'strategy': "deepspeed_stage_1",
    }
    
    # Add strategy for distributed training if specified
    if config['lightning'].get('strategy'):
        if config['lightning']['strategy'] == 'ddp_find_unused_parameters_true':
            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            strategy = config['lightning']['strategy']
        trainer_kwargs['strategy'] = strategy
    
    # Add Lightning profiler if requested
    if config['lightning'].get('profiler'):
        from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
        profiler_type = config['lightning']['profiler']
        if profiler_type == 'simple':
            trainer_kwargs['profiler'] = SimpleProfiler()
            print("Using Simple Profiler")
        elif profiler_type == 'advanced':
            trainer_kwargs['profiler'] = AdvancedProfiler()
            print("Using Advanced Profiler")
    
    trainer = L.Trainer(**trainer_kwargs)
    
    # Print configuration only on main process using Lightning's trainer
    rank_zero_info(f"Configuration: {config}")
    
    # Test only or train + validate
    if args.test_only is not None:
        trainer.test(model, datamodule=datamodule, ckpt_path=args.test_only)
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)
        if len(callbacks) > 0 and hasattr(callbacks[0], 'best_model_path'):
            best_model_path = callbacks[0].best_model_path
            if best_model_path:
                trainer.test(model, datamodule=datamodule, ckpt_path=best_model_path)
                
        # Close WandB logger to ensure logs are saved
        if wandb_config.get('enable', False):
            import wandb
            wandb.finish()


if __name__ == '__main__':
    main()
