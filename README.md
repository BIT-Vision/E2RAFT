# e2flow

This repository contains the official codes for our paper:

```Preserving Motion Detail in the Dark: Event-enhanced Optical Flow Estimation via Recurrent Feature Fusion```

## Requirements

You can install the Python and Pytorch environment by running the following commands:

```
conda create --name e2flow
conda activate e2flow
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Then, you can install the necessary Python libraries using the following command:

```
pip install -r requirements.txt
```

## Checkpoints

You can get checkpoint of our model from [e2flow-chairsDark](https://drive.google.com/drive/folders/14lrhoKdycVyfgtUlWHTkn6RV6JwNpNS6).

Meanwhile, we have gathered some published model weights for comparative experiments. You can get them from  [Checkpoints](https://drive.google.com/drive/folders/1rW_M4aqLmHve7GN19sC0BZeTpzF96olM).

We suggest organizing the checkpoints as follows:

```
|- checkpoints
|    |- e2flow
|    |    |- e2flow-chairsDark.pth
|    |- raft
|    |    |- chairs.pth
|    |    |- things.pth
|    |- ...
```

## Data Preparation

### FlyingChairs-Dark

You can download the [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs) dataset, and then generate simulated data based on the simulation baseline proposed in the paper. 

You can also directly download simulated data from (Link).

Completely, the dataset should be organized as following format:

```
|- train
|    |- dark
|    |    |- 0000000-img_0.npy
|    |    |- 0000000-img_1.npy
|    |    |- ...
|    |    |- 0022231-img_0.npy
|    |    |- 0022231-img_0.npy
|    |- 0000000-flow_01.flo
|    |- 0000000-img_0.png
|    |- 0000000-img_1.png
|    |- ...
|    |- 0022231-img_1.png
|- val
|    |- dark
|    |    |- 0000000-img_0.npy
|    |    |- ...
|    |    |- 0000639-img_1.npy
|    |- 0000000-flow_01.flo
|    |- ...
|    |- 0000639-img_1.png
|- voxels_train_b5_pn.hdf5
|- voxels_val_b5_pn.hdf5
```

### MVSEC
You can download [MVSEC](https://daniilidis-group.github.io/mvsec/) in hdf5 format from [Google Drive](https://drive.google.com/drive/folders/1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV).

The dataset should be organized as following format:

```
|- data_hdf5
|    |- indoor_flying1_data.hdf5
|    |- indoor_flying1_gt.hdf5
|    |- indoor_flying1_gt_index.hdf5
|    |- ...
|    |- outdoor_day2_data.hdf5
|    |- outdoor_day2_gt.hdf5
|    |- outdoor_day2_gt_index.hdf5
```

###  Real-world low light dataset

You can get it from [RealData](https://pan.baidu.com/s/1HTlGbnVEaLctz-lZsnU6WQ?pwd=5g2t).

## Usage

### Training

### Testing
