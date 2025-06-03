
===============================================================================
XReflection - An Easy-to-use Toolbox for Single-image Reflection Removal
===============================================================================

XReflection is a neat toolbox tailored for single-image reflection removal (SIRR). 
We offer state-of-the-art SIRR solutions for training and inference, 
with a high-performance data pipeline, multi-GPU/TPU/NPU support, and more!

📰 News and Updates
==================

- **[Upcoming]** More models are on the way!
- **[2025-05-26]** Release a training/testing pipeline.

💡 Key Features
===============

- All-in-one integration for the state-of-the-art SIRR solutions. We aim to create an out-of-the-box experience for SIRR research.
- Multi-GPU/TPU support via PyTorchLightning.
- Pretrained model zoo.
- Fast data synthesis pipeline.

📝 Introduction
===============

Please visit the documentation for more features and usage.

🚀 Installation
===============

Requirements
------------

Make sure you have the following system dependencies installed:

- Python >= 3.8
- PyTorch >= 1.10
- PyTorchLightning >= 1.5
- CUDA >= 11.2 (for GPU support)

Installation Commands
---------------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-username/XReflection.git
   cd XReflection

   # Install dependencies
   pip install -r requirements.txt

📦 Getting Started
==================

Inference with Pretrained Models
---------------------------------

Run reflection removal on an image:

.. code-block:: python

   from xreflection import inference

   result = inference.run("path_to_your_image.jpg", model_name="default_model")

Training a Model
----------------

.. code-block:: bash

   python tools/train.py --config configs/train_config.yaml

Data Preparation
----------------

Generate synthetic reflection datasets:

.. code-block:: bash

   python tools/data_pipeline.py --input_dir ./raw_images --output_dir ./synthetic_data

🌟 Features in Detail
=====================

Pretrained Model Zoo
--------------------

Access pretrained models for various SIRR algorithms:

+----------------+---------------------+---------------------+
| Model Name     | Description         | Performance Metrics |
+================+=====================+=====================+
| Default Model  | General SIRR        | PSNR: 32.5, SSIM:  |
|                |                     | 0.85                |
+----------------+---------------------+---------------------+
| Enhanced Model | Optimized structure | PSNR: 34.3, SSIM:  |
|                |                     | 0.88                |
+----------------+---------------------+---------------------+

🙏 License and Acknowledgement
==============================

This project is licensed under the Apache License 2.0. See the `LICENSE <LICENSE.md>`_ file for details.

The authors would express gratitude to the computational resource support from Google's TPU Research Cloud.

📧 Contact
==========

If you have any questions, please email **peiyuan_he@tju.edu.cn**