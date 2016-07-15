DeepCompare training code
==========

This is training code for CVPR2015 paper "Learning to Compare Image Patches via Convolutional Neural Networks". http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zagoruyko_Learning_to_Compare_2015_CVPR_paper.pdf

The original code was rewritten to support:
* parallel data augmentation via torchnet
* Multi-GPU
* half precision (fp16) training
* CPU training

# Required rocks

```
luarocks install torchnet
luarocks install cudnn
```

Cudnn is optional.

Preprocessing code depends on OpenCV. To install it follow https://github.com/VisionLabs/torch-opencv

# Data preprocessing

First download Brown dataset http://www.cs.ubc.ca/~mbrown/patchdata/patchdata.html
Then run preprocessing with this script https://gist.github.com/szagoruyko/569c2b713a2c40629ecbe75fb3c9d980

```
th create_dataset_file.lua notredame/
th create_dataset_file.lua yosemite/
th create_dataset_file.lua liberty/
```

This will convert the data to torch format for faster loading and compute per-patch mean.

# Testing existing models

```
testOnly=true train_set=liberty/data.t7 test_set=notredame/data.t7 model=2ch_liberty.t7 th train.lua
```

Only FPR95 value will be printed, to access TPR and FPR values see `fpr95meter.tpr` and `fpr95meter.fpr` tensors.
It is easy to save them in torch/numpy/matlab format to visualize or compare with other methods.

To test on CPU additionally set `data_type=torch.FloatTensor`

# Training

```
train_set=liberty/data.t7 test_set=notredame/data.t7 model=siam th train.lua
```

I prefer to save logs in a separate folder, here is the script:

```bash
export train_set=/opt/datasets/daisy/notredame/data.t7
export test_set=/opt/datasets/daisy/liberty/data.t7

export save_folder=logs/deepcompare_${model}_$RANDOM$RANDOM
mkdir -p $save_folder

th train.lua | tee $save_folder/log.txt
```

To train on multuple GPUs set: `nGPU=4`

To train in half-presicion set: `data_type=torch.CudaHalfTensor`
