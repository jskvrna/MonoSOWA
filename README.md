<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

[![HF Competition][hf-shield]][hf-url]

[![arXiv][arxiv-shield]][arxiv-url]

<!-- ... existing definitions ... -->
[license-url]: https://github.com/jskvrna/MonoSOWA/blob/master/LICENSE.txt

<!-- Add these new definitions -->
[hf-shield]: https://img.shields.io/badge/HuggingFace%20Competition-1st%20Place-blue
[hf-url]: https://huggingface.co/spaces/usm3d/S23DR2025
[arxiv-shield]: https://img.shields.io/badge/arXiv-2405.12345-b31b1b.svg
[arxiv-url]: https://arxiv.org/abs/2501.09481

# MonoSOWA

Official implementation of **MonoSOWA**: **S**calable m**o**nocular 3D Object detector **W**ithout human **A**nnotations.

<!-- TABLE OF CONTENTS -->
<details open>
    <summary>Table of Contents</summary>
    <ol>
        <li>
            <a href="#Abstract">Abstract</a>
        </li>
        <li>
            <a href="#getting-started">Getting Started</a>
            <ul>
                <li><a href="#prerequisites">Prerequisites</a></li>
                <li><a href="#installation">Installation</a></li>
            </ul>
        </li>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#roadmap">Roadmap</a></li>
        <li><a href="#contributing">Contributing</a></li>
        <li><a href="#license">License</a></li>
        <li><a href="#contact">Contact</a></li>
        <li><a href="#acknowledgments">Acknowledgments</a></li>
    </ol>
</details>

<!-- ABSTRACT -->
## Abstract

Inferring object 3D position and orientation from a single RGB camera is a foundational task in computer vision with many important applications. Traditionally, 3D object detection methods are trained in a fully-supervised setup, requiring LiDAR and vast amounts of human annotations, which are laborious, costly, and do not scale well with the ever-increasing amounts of data being captured.

We present a novel method to train a 3D object detector from a single RGB camera without domain-specific human annotations, making orders of magnitude more data available for training. The method uses newly proposed Local Object Motion Model to disentangle object movement source between subsequent frames, is approximately 700 times faster than previous work and compensates camera focal length differences to aggregate multiple datasets.

The method is evaluated on three public datasets, where despite using no human labels, it outperforms prior work by a significant margin. It also shows its versatility as a pre-training tool for fully-supervised training and shows that combining pseudo-labels from multiple datasets can achieve comparable accuracy to using human labels from a single dataset.

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

We recommend creating two distinct virtual environments, one for pseudo-labelling and second for MonoDETR, due to conflicts of packages.

Clone the repo

```sh
git clone https://github.com/jskvrna/MonoSOWA.git
```

### Installation of Pseudo labelling pipeline

1.  Create a virtual environment
    ```sh
    python3 -m venv pseudo_labelling
    source pseudo_labelling/bin/activate
    ```
2.  Install dependencies from `requirements.txt`
    ```sh
    cd MonoSOWA/pseudo_label_generator
    pip install -r requirements.txt
    ```
3.  Install Detectron2
    ```sh
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```
4.  Build PyTorch3D from source (recommended)
    ```sh
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"
    ```

### Installation of MonoDETR
This gets little bit tricky. MultiScaleDeformableAttetion, which is required by MonoDETR, requires PyTorch version of lower than 2.0. I have tested it with PyTorch 1.13.1 and it works fine. However, as the MSDA is compiled it requires the nvcc (cuda toolkit) to have the same version as the pytorch has been compiled with. This makes it little bit harder to install it. 

It is worth noting that our used Canonical Objects Space can be simply implemented to any off-the-shelf Monocular Detector.

1.  Create a virtual environment
    ```sh
    deactivate
    cd ../../
    python3 -m venv monodetr
    source monodetr/bin/activate
    ```
2.  Install dependencies from `requirements.txt`
    ```sh
    cd MonoSOWA/MonoDETR
    pip install -r requirements.txt
    ```
3. Compile the deformable attention
    ```sh
    cd lib/models/monodetr/ops/
    bash make.sh

    cd ../../../..
    ```



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/jskvrna/MonoSOWA.svg?style=for-the-badge
[contributors-url]: https://github.com/jskvrna/MonoSOWA/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/jskvrna/MonoSOWA.svg?style=for-the-badge
[forks-url]: https://github.com/jskvrna/MonoSOWA/network/members
[stars-shield]: https://img.shields.io/github/stars/jskvrna/MonoSOWA.svg?style=for-the-badge
[stars-url]: https://github.com/jskvrna/MonoSOWA/stargazers
[issues-shield]: https://img.shields.io/github/issues/jskvrna/MonoSOWA.svg?style=for-the-badge
[issues-url]: https://github.com/jskvrna/MonoSOWA/issues
[license-shield]: https://img.shields.io/github/license/jskvrna/MonoSOWA.svg?style=for-the-badge
[license-url]: https://github.com/jskvrna/MonoSOWA/blob/master/LICENSE.txt
[product-screenshot]: images/screenshot.png
[PyTorch.org]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[NumPy.org]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[OpenCV.org]: https://img.shields.io/badge/opencv-%235C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org/
