# XReflection - An Easy-to-use Toolbox for Single-image Reflection Removal


<div align="center"><img src="./docs/_static/XReflection_logo.png" alt="XReflection Logo" width="50%" height="50%"/></div>

<!-- ---
## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>
-->

<br>

XReflection is a neat toolbox tailored for single-image reflection removal(SIRR). We offer state-of-the-art SIRR solutions for training and inference, with a high-performance data pipeline, multi-GPU/TPU support, and more!


---
## 📰 News and Updates

- **[2025-07-16]** DSRNet is available in the model zoo. More models are on the way!
- **[2025-05-26]** Release a training/testing pipeline. 

---
## 💡 Key Features

+ All-in-one intergration for the state-of-the-art SIRR solutions. We aim to create an out-of-the-box experience for SIRR research.
+ Multi-GPU/TPU support via PyTorchLightning. 
+ Pretrained model zoo.
+ Fast data synthesis pipeline.
---
## 📝 Introduction

Please visit the [documentation](https://xreflection.readthedocs.io/en/latest/) for more features and usage.


---

## 🚀 Installation


### Installation Commands
```bash
# Build from source
git clone https://github.com/hainuo-wang/XReflection.git
cd XReflection

# Install dependencies
pip install -r requirements.txt
python setup.py develop
```

---

## 📦 Getting Started

### Testing a Pretrained Model
```python
python tools/train.py --config configs/train_config.yaml --resume pretrained.ckpt --test_only
```

### Training a Model
```bash
python tools/train.py --config configs/train_config.yaml
```

### Resume Training from a Checkpoint
```python
python tools/train.py --config configs/train_config.yaml --resume pretrained.ckpt
```

### Data Preparation
#### Training dataset
* 7,643 images from the
  [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/), center-cropped as 224 x 224 slices to synthesize training pairs;
* 90 real-world training pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal);
* 200 real-world training pairs provided by [IBCLN](https://github.com/JHL-HUST/IBCLN).

#### Testing dataset
* 45 real-world testing images from [CEILNet dataset](https://github.com/fqnchina/CEILNet);
* 20 real testing pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal);
* 20 real testing pairs provided by [IBCLN](https://github.com/JHL-HUST/IBCLN);
* 454 real testing pairs from [SIR^2 dataset](https://sir2data.github.io/), containing three subsets (i.e., Objects (200), Postcard (199), Wild (55)). 

Download all in one from https://checkpoints.mingjia.li/sirs.zip

---

## 🌟 Features in Detail

### Pretrained Model Zoo
Access pretrained models for various SIRR algorithms:
TODO


<!-- ---

## 📄 Citation

If you find XReflection useful in your research or work, please consider citing:
```bibtex
@misc{xreflection2024,
  title={XReflection: A Toolbox for Single-image Reflection Removal},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-username/XReflection}}
}
``` -->

---
## 🙏 License and Acknowledgement

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE.md) file for details.
The authors would express gratitude to the computational resource support from Google's TPU Research Cloud.






