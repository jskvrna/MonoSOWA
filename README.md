<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

[![arXiv][arxiv-shield]][arxiv-url]

[license-url]: https://github.com/jskvrna/MonoSOWA/blob/master/LICENSE.txt

[arxiv-shield]: https://img.shields.io/badge/arXiv-2405.12345-b31b1b.svg
[arxiv-url]: https://arxiv.org/abs/2501.09481

# MonoSOWA

Official implementation of **MonoSOWA**: **S**calable m**o**nocular 3D Object detector **W**ithout human **A**nnotations.
<p align="center">
    <img src="monosowa_intro.png" alt="MonoSOWA Overview" width="80%">
</p>
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
        <li>
            <a href="#usage">Usage</a>
            <ul>
                <li><a href="#pseudo-labelling-pipeline">Pseudo-labelling Pipeline</a></li>
                <li><a href="#monodetr-training">MonoDETR Training</a></li>
            </ul>
        </li>
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

To get a local copy up and running, follow these simple steps.

### Prerequisites

We recommend creating two separate virtual environments: one for pseudo-labelling and another for MonoDETR, due to potential package conflicts.

Clone the repository:

```sh
git clone https://github.com/jskvrna/MonoSOWA.git
```

### Installation of Pseudo-Labelling Pipeline

1.  Create a virtual environment:
    ```sh
    python3 -m venv pseudo_labelling
    source pseudo_labelling/bin/activate
    ```
2.  Install dependencies from `requirements.txt`:
    ```sh
    cd MonoSOWA/pseudo_label_generator
    pip install -r requirements.txt
    ```
3.  Install Detectron2:
    ```sh
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```
4.  Build PyTorch3D from source (recommended):
    ```sh
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"
    ```

### Installation of MonoDETR

This step is a bit more involved. MultiScaleDeformableAttention, required by MonoDETR, needs a PyTorch version lower than 2.0. We have tested it with PyTorch 1.13.1, which works well. Since MSDA is compiled, the CUDA toolkit (nvcc) version must match the version PyTorch was compiled with. This can make installation more challenging.

Note: Our Canonical Objects Space can be implemented in any off-the-shelf monocular detector.

1.  Create a virtual environment:
    ```sh
    deactivate
    cd ../../
    python3 -m venv monodetr
    source monodetr/bin/activate
    ```
2.  Install dependencies from `requirements.txt`:
    ```sh
    cd MonoSOWA/MonoDETR
    pip install -r requirements.txt
    ```
3.  Compile the deformable attention module:
    ```sh
    cd lib/models/monodetr/ops/
    bash make.sh

    cd ../../../..
    ```

<!-- USAGE EXAMPLES -->
## Usage

Our method fully supports both KITTI and KITTI-360 datasets.

MonoDETR expects input in the KITTI format. For this reason, we provide the script `k360_to_k.py`, which converts KITTI-360 with pseudo-labels to the MonoDETR KITTI format.

To use the Waymo Open Perception Dataset, we provide a script that converts it into KITTI-360 format. Only the front camera is supported, and labels are extracted only if present in the front camera. See `waymo_to_kitti_projected.py` for more details.

### Pseudo-Labelling Pipeline

**Dataset Preparation**

1. KITTI - https://www.cvlibs.net/datasets/kitti/
    
    a. Download the [3D Object Detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) data and the [Raw Data](https://www.cvlibs.net/datasets/kitti/raw_data.php) (complete sequences).

    b. Unpack the `object_detection` data to match the following directory structure:
    ```text
    object_detection/
    ├── training/
    │   ├── calib/
    │   ├── image_2/
    │   ├── image_3/
    │   ├── label_2/
    │   └── velodyne/
    └── testing/
        ├── calib/
        ├── image_2/
        ├── image_3/
        └── velodyne/
    ```

    c. Unpack the `raw_data` data to match the following directory structure:

    ```text
    raw_data/
    ├── 2011_09_26/
    │   ├── 2011_09_26_drive_0001_sync/
    │   │   ├── image_00/
    │   │   ├── image_01/
    │   │   ├── image_02/
    │   │   ├── image_03/
    │   │   ├── oxts/
    │   │   └── velodyne_points/
    │   ├── ...
    │   ├── calib_cam_to_cam.txt
    │   ├── calib_imu_to_velo.txt
    │   └── calib_velo_to_cam.txt
    ├── 2011_09_28/
    ├── 2011_09_29/
    ├── 2011_09_30/
    └── 2011_10_03/
    ```

2. KITTI-360 - https://www.cvlibs.net/datasets/kitti-360/

    a. Download the [Perspective Images for Train & Val](https://www.cvlibs.net/datasets/kitti-360/download.php), [Calibrations](https://www.cvlibs.net/datasets/kitti-360/download.php), and [Vehicle Poses](https://www.cvlibs.net/datasets/kitti-360/download.php).

    b. Unpack the `Perspective Images for Train & Val`, `Calibrations`, and `Vehicle Poses` as follows:

    ```text
    KITTI-360/
    ├── 2013_05_28_drive_0000_sync/
    │   ├── image_00/
    │   └── image_01/
    ├── 2013_05_28_drive_0002_sync/
    ├── 2013_05_28_drive_0003_sync/
    ├── 2013_05_28_drive_0004_sync/
    ├── 2013_05_28_drive_0005_sync/
    ├── 2013_05_28_drive_0006_sync/
    ├── 2013_05_28_drive_0007_sync/
    ├── 2013_05_28_drive_0009_sync/
    ├── 2013_05_28_drive_0010_sync/
    ├── calibration/
    └── data_poses/
    ```

**Pseudo-Label Creator**

1. Update `pseudo_label_generator/3d/configs/config.yaml`. Most importantly:
    ```text
    kitti_path: path to object_detection
    all_dataset_path: path to either KITTI raw_data or the KITTI-360 folder
    detectron_config: path to the mvitv2 config
    merged_frames_path: path to the output folder for intermediate files
    labels_path: path to the output folder for final labels
    ```

2. Generate relative transformations between frames (CPU only):

    KITTI:
    ```sh 
    cd pseudo_label_generator/3d/scrip
    python main.py --config ../configs/config.yaml --dataset kitti --action transformations
    ```

    KITTI-360:
    ```sh 
    cd pseudo_label_generator/3d/scrip
    python main.py --config ../configs/config.yaml --dataset all --action transformations
    ```

3. Generate pseudo-lidar with Metric3Dv2 (GPU recommended):

    KITTI:
    ```sh 
    cd pseudo_label_generator/3d/scrip
    python main.py --config ../configs/config.yaml --dataset kitti --action lidar_scans
    ```

    KITTI-360:
    ```sh 
    cd pseudo_label_generator/3d/scrip
    python main.py --config ../configs/config.yaml --dataset all --action lidar_scans
    ```
4. Generate 2D masks via MViTv2 (GPU recommended):

    KITTI:
    ```sh 
    cd pseudo_label_generator/3d/scrip
    python main.py --config ../configs/config.yaml --dataset kitti --action mask_tracking
    ```

    KITTI-360:
    ```sh 
    cd pseudo_label_generator/3d/scrip
    python main.py --config ../configs/config.yaml --dataset all --action mask_tracking
    ```

5. Perform frame aggregation and optimization (CPU only, parallelization recommended):

    KITTI:
    ```sh 
    cd pseudo_label_generator/3d/scrip
    python main.py --config ../configs/config.yaml --dataset kitti --action optimization
    ```

    KITTI-360:
    ```sh 
    cd pseudo_label_generator/3d/scrip
    python main.py --config ../configs/config.yaml --dataset all --action optimization
    ```
### MonoDETR Training

We describe training for the KITTI dataset, as preparing KITTI-360 data requires additional steps (MonoDETR only supports KITTI format). Feel free to contact us if you need assistance.

**Dataset Preparation**

1. Copy the `object_detection` part of the KITTI dataset and ImageSets into `MonoDETR/data/KITTI/` with the official labels, using the following structure:

    ```text
    MonoDETR/data/KITTI/
    ├── ImageSets/
    │   ├── train.txt
    │   └── test.txt
    ├── training/
    │   ├── calib/
    │   ├── image_2/
    │   ├── image_3/
    │   ├── label_2/
    │   └── velodyne/
    └── testing/
        ├── calib/
        ├── image_2/
        ├── image_3/
        └── velodyne/
    ```

2. Replace human ground-truth training samples with pseudo-labels, but keep the validation set. Also, prepare the labels for training:
    ```sh
    cd pseudo_label_generator/
    python label_replacer.py MonoDETR/data/KITTI path_to_pseudo_labels
    python label_preparation.py MonoDETR/data/KITTI
    ```

**Model Training**

To train the model, use:
```sh
python train.py --config configs/monodetr.yaml
```

**Evaluation**

To evaluate a trained model:
```sh
python test.py --config configs/monodetr.yaml
```
