# KITTI 3D Object Detection with OpenPCDet

A 3D object detection implementation using the PointPillar algorithm on the KITTI dataset, built with OpenPCDet framework.

## Overview

This project implements 3D object detection for autonomous driving scenarios using point cloud data from the KITTI dataset. The model is trained to detect three classes: Car, Pedestrian, and Cyclist.

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- OpenPCDet framework
- KITTI dataset

## Installation

1. Clone the OpenPCDet repository:
```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python setup.py develop
```

3. Download and prepare the KITTI dataset following the OpenPCDet documentation.

## Training Configuration

- Model: PointPillar
- Dataset: KITTI
- Batch size: 4
- Epochs: 10
- Optimizer: Adam OneCycle
- Learning rate: 0.003
- Classes: Car, Pedestrian, Cyclist

## Usage

Run the Jupyter notebook to train and evaluate the model:

```bash
jupyter notebook Kitti_OpenPCDet.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Model building and training
- Evaluation on validation set
- Visualization of results

## Results

The model achieves the following Average Precision (AP) on the KITTI validation set:

### AP BEV (Bird's Eye View) R40

| Class      | Easy  | Moderate | Hard  |
|------------|-------|----------|-------|
| Car        | 91.44 | 85.23    | 84.15 |
| Pedestrian | 61.49 | 56.36    | 52.04 |
| Cyclist    | 80.27 | 63.27    | 58.82 |

### AP 2D (Image) R40

| Class      | Easy  | Moderate | Hard  |
|------------|-------|----------|-------|
| Car        | 95.28 | 91.38    | 90.15 |
| Pedestrian | 66.66 | 62.80    | 59.13 |
| Cyclist    | 87.86 | 72.70    | 68.62 |

### Recall Metrics

- RCNN Recall @ IoU 0.3: 93.88%
- RCNN Recall @ IoU 0.5: 87.83%
- RCNN Recall @ IoU 0.7: 61.45%


## Acknowledgments

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) - Open source point cloud detection toolbox
- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/) - Dataset provider

## License

This project follows the license terms of the OpenPCDet framework.
