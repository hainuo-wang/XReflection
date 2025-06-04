
===============================================================================
XReflection - An Easy-to-use Toolbox for Single-image Reflection Removal
===============================================================================

XReflection is a neat toolbox tailored for single-image reflection removal (SIRR). 
We offer state-of-the-art SIRR solutions for training and inference, 
with a high-performance data pipeline, multi-GPU/TPU/NPU support, and more!


📰 News and Updates
====================

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

Please visit the `Installation <Installation.html>`_ for more features and usage.

🚀 Installation
===============

Requirements
------------

Make sure you have the following system dependencies installed:

- Python >= 3.10
- PyTorch >= 2.5
- PyTorchLightning >= 2.5
- CUDA >= 12.1 (for GPU support)

Installation Commands
---------------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/hainuo-wang/XReflection.git
   cd XReflection

   # Install dependencies
   pip install -r requirements.txt
   python setup.py develop

📦 Getting Started
==================

Inference with Pretrained Models
---------------------------------

Run reflection removal on an image:

.. code-block:: python

      TODO

Training a Model
----------------

.. code-block:: bash

   python tools/train.py --config configs/train_config.yaml

Data Preparation
----------------

Training dataset
~~~~~~~~~~~~~~~~

- 7,643 images from the `Pascal VOC dataset <http://host.robots.ox.ac.uk/pascal/VOC/>`_, center-cropped as 224 x 224 slices to synthesize training pairs.
- 90 real-world training pairs provided by `Zhang et al. <https://github.com/ceciliavision/perceptual-reflection-removal>`_.
- 200 real-world training pairs provided by `IBCLN <https://github.com/JHL-HUST/IBCLN>`_ (In our training setting 2, † labeled in our paper).


Testing dataset
~~~~~~~~~~~~~~~

- 45 real-world testing images from `CEILNet dataset <https://github.com/fqnchina/CEILNet>`_.
- 20 real testing pairs provided by `Zhang et al. <https://github.com/ceciliavision/perceptual-reflection-removal>`_.
- 20 real testing pairs provided by `IBCLN <https://github.com/JHL-HUST/IBCLN>`_.
- 454 real testing pairs from `SIR² dataset <https://sir2data.github.io/>`_, containing three subsets (i.e., Objects (200), Postcard (199), Wild (55)).

Download all in one from `https://checkpoints.mingjia.li/sirs.zip <https://checkpoints.mingjia.li/sirs.zip>`_

🌟 Features in Detail
=====================

Pretrained Model Zoo
--------------------

Access pretrained models for various SIRR algorithms:

TODO


🙏 License and Acknowledgement
==============================

This project is licensed under the Apache License 2.0. See the `LICENSE <LICENSE.md>`_ file for details.

The authors would express gratitude to the computational resource support from Google's TPU Research Cloud.

📧 Contact
==========

If you have any questions, please email **peiyuan_he@tju.edu.cn**

.. toctree::
   :maxdepth: 2
   :hidden:

   Installation
   api