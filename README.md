# KITTI 3D Object Detection with OpenPCDet

A 3D object detection implementation using the PointPillar algorithm on the KITTI dataset, built with the OpenPCDet framework. This project demonstrates training, evaluation, and visualization of 3D bounding boxes for autonomous driving scenarios.

## Overview

This project implements 3D object detection for autonomous driving using point cloud data from the KITTI dataset. The model detects three object classes: **Car**, **Pedestrian**, and **Cyclist** using LiDAR point clouds projected into a bird's-eye view representation.

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- OpenPCDet framework
- KITTI dataset
- Additional libraries: numpy, opencv-python, matplotlib, tqdm

## Installation

### 1. Clone and Setup OpenPCDet

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
pip install -r requirements.txt
python setup.py develop
```

### 2. Download KITTI Dataset

Follow the [OpenPCDet KITTI dataset documentation](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to download and prepare the dataset.

## Implementation Details

### Training Configuration

The notebook implements training with the following setup:

- **Model**: PointPillar
- **Dataset**: KITTI 3D Object Detection
- **Batch Size**: 4
- **Number of Epochs**: 80
- **Optimizer**: Adam with OneCycleLR scheduler
- **Base Learning Rate**: 0.003
- **Weight Decay**: 0.01
- **Mixed Precision Training**: Enabled (AMP with GradScaler)
- **Classes**: Car, Pedestrian, Cyclist

## Usage

### Training

Run the Jupyter notebook to train the model:

```bash
jupyter notebook Kitti_OpenPCDet.ipynb
```

The notebook contains the following main sections:
1. Environment setup and configuration loading
2. Dataset preparation and dataloader creation
3. Model building (PointPillar architecture)
4. Optimizer and scheduler setup
5. Training loop with AMP
6. Model checkpoint saving
7. Evaluation on validation set
8. Visualization of predictions

### Inference and Visualization

The notebook includes a custom visualization function that:
- Loads trained checkpoints
- Performs inference on test samples
- Projects 3D bounding boxes onto camera images
- Displays predictions with confidence scores

Example usage:
```python
visualize_kitti_prediction(model, test_set, index=56, score_thresh=0.5)
```

## Results

### Training Metrics

The model was trained for 10 epochs with the following characteristics:
- Training loss tracked per epoch
- Validation metrics computed after each epoch
- Best model checkpoint saved at epoch 6

### Performance Metrics

Average Precision (AP) results on KITTI validation set:

#### AP BEV (Bird's Eye View) R40
| Class      | Easy  | Moderate | Hard  |
|------------|-------|----------|-------|
| Car        | 91.44 | 85.23    | 84.15 |
| Pedestrian | 61.49 | 56.36    | 52.04 |
| Cyclist    | 80.27 | 63.27    | 58.82 |

#### AP 3D R40
| Class      | Easy  | Moderate | Hard  |
|------------|-------|----------|-------|
| Car        | 95.28 | 91.38    | 90.15 |
| Pedestrian | 66.66 | 62.80    | 59.13 |
| Cyclist    | 87.86 | 72.70    | 68.62 |

#### Recall Metrics
- **RCNN Recall @ IoU 0.3**: 93.88%
- **RCNN Recall @ IoU 0.5**: 87.83%
- **RCNN Recall @ IoU 0.7**: 61.45%

### Visualization Examples

The notebook demonstrates inference results with 3D bounding boxes projected onto camera images, showing:
- Accurate car detection with high confidence scores
- Pedestrian detection in various scenarios
- Cyclist detection with proper orientation

Example visualizations:

<img width="1182" height="371" alt="image" src="https://github.com/user-attachments/assets/e439e12f-c48b-45a4-9fd0-89f1c4d33f38" />
<img width="1182" height="371" alt="image" src="https://github.com/user-attachments/assets/fc264b5c-8abe-49e1-b26e-6a8c16f683c1" />
<img width="1182" height="370" alt="image" src="https://github.com/user-attachments/assets/be049774-d32c-4a0d-8c45-a9e6fae1c50e" />
<img width="1182" height="372" alt="image" src="https://github.com/user-attachments/assets/b5ea57da-0a3a-4343-98eb-1c6ca9103c93" />

## Code Structure

The notebook is organized into the following sections:

1. **Setup and Configuration** - Environment setup, path configuration, and YAML config loading
2. **Data Preparation** - Dataset loading, train/val split, and dataloader creation
3. **Model Building** - PointPillar model instantiation
4. **Training Setup** - Optimizer, scheduler, and mixed precision setup
5. **Training Loop** - Main training loop with loss tracking and checkpoint saving
6. **Evaluation** - Validation metrics computation and performance analysis
7. **Inference** - Loading best checkpoint and running predictions
8. **Visualization** - Custom visualization functions for 3D bounding boxes
9. **Metrics Analysis** - Training curves and performance plots

## Acknowledgments

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) - Open source point cloud detection toolbox
- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/) - Dataset provider
- PointPillar paper: [Lang et al., CVPR 2019](https://arxiv.org/abs/1812.05784)

## License

This project follows the license terms of the OpenPCDet framework.
