# KPAC-Net (Retrained Version)


![License CC BY-NC](https://img.shields.io/badge/License-GNU_AGPv3-yellowgreen.svg?style=flat)
![Ubuntu](https://img.shields.io/badge/Ubuntu-16.0.4%20&%2018.0.4-blue.svg?style=plastic)
![Python](https://img.shields.io/badge/Python-3.6-yellowgreen.svg?style=plastic)
![CUDA](https://img.shields.io/badge/CUDA-10.2%20-yellowgreen.svg?style=plastic)
![TensorFlow 1.13.1](https://img.shields.io/badge/tensorflow-1.13.1-yellowgreen.svg?style=plastic)
![TensorLayer 1.11.1](https://img.shields.io/badge/tensorlayer-1.11.1-yellowgreen.svg?style=plastic)

This repository offers the retrained version of KPAC network using the training strategy mentioned in the following paper.

> **[Learning to Deblur using Light Field Generated and Real Defocused Images](https://arxiv.org/pdf/2204.00367.pdf)**<br>
> Lingyan Ruan<sup>\*</sup>, Bin Chen<sup>\*</sup>, Jizhou Li, Miuling Lam （\* equal contribution）<br>
> IEEE Computer Vision and Pattern Recognition (**CVPR Oral**) 2022

<p align="left">
  <a>
    <img src="./assets/performance_gain.png" width="50%" alt="teaser figure">
  </a><br>
</p>

<!-- <img src="./assets/performance_gain.png" width="50%" alt="teaser figure"> -->

The orignial KPAC-Net can be found **[Here](https://github.com/HyeongseokSon1/KPAC)** .

## Code Here

### Prerequisites

Notes: the code may also work with other library versions that didn't specify here.

Table 4 in our main paper.

#### 1. Installation

Clone this project to your local machine

```bash
$ git clone https://github.com/lingyanruan/KPAC-Net.git
$ cd KPAC-Net
```

#### 2. Pre-trained models

Download and unzip [pretrained weights](https://drive.google.com/drive/folders/1KZp0t9v7bHUyIcAQzZHfVFjAA0LUrHy3?usp=sharing) under `./ModelCheckpoints/`:

#### 3. Datasets

Download and unzip test sets [DPDD](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) under folder `./DPD/`

#### 4. Command Line

```shell
python Eval.py
```

## Contact

Should you have any questions, please open an issue or contact me [lyruanruan@gmail.com](mailto:lyruanruan@gmail.com)

## License

This software is being made available under the terms in the [LICENSE](LICENSE) file.

