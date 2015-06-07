Code for CVPR15 paper "Learning to Compare Image Patches via Convolutional Neural Networks"
-----
This package allows researches to apply the described networks to match image patches and extract corresponding patches.

We tried to make the code as easy to use as possible. The original models were trained with Torch ( http://torch.ch ) and we release them in Torch7 and binary formats with C++ bindings which do not require Torch installation. Thus we provide example code how to use the models in Torch, MATLAB and with OpenCV http://opencv.org

CREDITS, LICENSE, CITATION
-----

Copyright © 2015 Ecole des Ponts, Universite Paris-Est

All Rights Reserved. A license to use and copy this software and its documentation solely for your internal 
research and evaluation  purposes, without fee and without a signed licensing agreement, is hereby granted 
upon your download of the software, through which you agree to the following: 
1)  the above copyright notice, this paragraph and the following three paragraphs will prominently appear 
in all internal copies and modifications; 
2)  no rights to sublicense or further distribute this software are granted; 
3) no rights to modify this software are granted; and 
4) no rights to assign this license are granted.   

Please Contact Prof. Nikos Komodakis,
6 Avenue Blaise Pascal - Cite Descartes, Champs-sur-Marne, 77455 Marne-la-Vallee cedex 2, France for commercial licensing opportunities, or for further distribution, modification or license rights.

Created by Sergey Zagoruyko and Nikos Komodakis. http://imagine.enpc.fr/~komodakn/

Please cite the paper below if you use this code in your research.

Sergey Zagoruyko, Nikos Komodakis, 
"Learning to Compare Image Patches via Convolutional Neural Networks". http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zagoruyko_Learning_to_Compare_2015_CVPR_paper.pdf, bib:

```
@InProceedings{Zagoruyko_2015_CVPR,
	author = {Zagoruyko, Sergey and Komodakis, Nikos},
	title = {Learning to Compare Image Patches via Convolutional Neural Networks},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2015}
}
```

### Models

We provide the models in 3 formats, two are Torch7 "nn" and "cudnn" formats and one is binary format with weights only. The table from the paper is here for convenience.

**All models expect input patches to be in [0;1] range before mean subtraction.**

| Train set | Test set | 2ch | 2ch2stream | 2chdeep | siam | siam2stream |
| --- |  --- | :---: |  :---: |  :---: |  :---: |  :---: |
| yosemite | notredame | 2.74 | **2.11** | 2.43 | 5.62 | 5.23 |
| yosemite | liberty | 8.59 | **7.2** | 7.4 | 13.48 | 11.34 |
| notredame | yosemite | 6.04 | **4.09** | 4.38 | 13.23 | 10.44 |
| notredame | liberty | 6.04 | 4.85 | **4.56** | 8.77 | 6.45 |
| liberty | yosemite | 7 | **5** | 6.18 | 14.76 | 9.39 |
| liberty | notredame | 2.76 | **1.9** | 2.77 | 4.04 | 2.82 |

To save time downloading models one by one run the script (downloads ~375MB):

```
./download_pack.sh
```

#### nn format

Models in nn format can be loaded and used without CUDA support in Torch. To enable CUDA support ```model:cuda()``` call required.

| Train Set | 2ch | 2ch2stream | 2chdeep | siam | siam2stream |
| --- |  :---: |  :---: |  :---: |  :---: |  :---: |
| yosemite | [3.49 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_yosemite_nn.t7) | [9.74 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_yosemite_nn.t7) | [4.15 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_yosemite_nn.t7) | [7.95 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_yosemite_nn.t7) | [22.36 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_yosemite_nn.t7) |
| notredame | [3.49 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_notredame_nn.t7) | [9.74 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_notredame_nn.t7) | [4.15 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_notredame_nn.t7) | [7.95 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_notredame_nn.t7) | [22.36 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_notredame_nn.t7) |
| liberty | [3.49 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_liberty_nn.t7) | [9.74 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_liberty_nn.t7) | [4.15 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_liberty_nn.t7) | [7.95 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_liberty_nn.t7) | [22.36 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_liberty_nn.t7) |

#### cudnn format

Models in cudnn format are faster, but need a special library from NVIDIA. Check https://github.com/soumith/cudnn.torch 

| Train Set | 2ch | 2ch2stream | 2chdeep | siam | siam2stream |
| --- |  :---: |  :---: |  :---: |  :---: |  :---: |
| yosemite | [3.75 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_yosemite_cudnn.t7) | [9.85 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_yosemite_cudnn.t7) | [4.15 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_yosemite_cudnn.t7) | [5.62 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_yosemite_cudnn.t7) | [17.46 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_yosemite_cudnn.t7) |
| notredame | [3.75 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_notredame_cudnn.t7) | [9.85 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_notredame_cudnn.t7) | [4.15 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_notredame_cudnn.t7) | [5.62 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_notredame_cudnn.t7) | [17.71 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_notredame_cudnn.t7) |
| liberty | [3.75 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_liberty_cudnn.t7) | [9.85 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_liberty_cudnn.t7) | [4.15 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_liberty_cudnn.t7) | [5.62 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_liberty_cudnn.t7) | [17.71 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_liberty_cudnn.t7) |

#### binary format

| Train Set | 2ch | 2ch2stream | 2chdeep | siam | siam2stream |
| --- |  --- |  --- |  --- |  --- |  --- |
| yosemite | [3.48 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_yosemite.bin) | [8.98 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_yosemite.bin) | [4.13 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_yosemite.bin) | [4.47 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_yosemite.bin) | [13.17 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_yosemite.bin) |
| notredame | [3.48 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_notredame.bin) | [8.98 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_notredame.bin) | [4.13 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_notredame.bin) | [4.47 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_notredame.bin) | [13.17 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_notredame.bin) |
| liberty | [3.48 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_liberty.bin) | [8.98 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_liberty.bin) | [4.13 MB](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_liberty.bin) | [4.47 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_liberty.bin) | [13.17 MB](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_liberty.bin) |

### Torch

To install torch follow http://torch.ch/
Check torch folder for examples.
Match patches on CPU:

```lua
require 'nn'

N = 76  -- the number of patches to match
patches = torch.rand(N,2,64,64):float()

-- load the network
net = torch.load'../networks/2ch/2ch_liberty_nn.t7'

-- in place mean subtraction
local p = patches:view(N,2,64*64)
p:add(-p:mean(3):expandAs(p))

-- get the output similarities
output = net:forward(patches)
```

In fact, nn and cudnn are not the only backends for Torch, for a big number of patches it might be faster to use cuda-convnet2 or SpatialConvolutionFFT from facebook.

### C++ API

The code was tested to work in Linux (Ubuntu 14.04) and OS X 10.10, although we release all the source code to enable usage in other operating systems.

We release CUDA code for now, CPU code might be added in the future. To install it you need to have CUDA 7.0 with the up-to-date CUDA driver.
Clone and compile this repository it with:

```
git clone --recursive https://github.com/szagoruyko/cvpr15deepmatch
cd cvpr15deepmatch
mkdir build; cd build;
cmake .. -DCMAKE_INSTALL_PREFIX=../install
make -j4 install
```

Then you will have ```loadNetwork``` function defined in src/loader.h, which expects the state and the path to a network in binary format on input. A simple example:

```c++
THCState *state = (THCState*)malloc(sizeof(THCState));
THCudaInit(state);

cunn::Sequential::Ptr net = loadNetwork(state, "networks/siam/siam_notredame.bin");

THCudaTensor *input = THCudaTensor_newWithSize4d(state, 128, 2, 64, 64);
THCudaTensor *output = net->forward(input); // output is 128x1 similarity score tensor
```

Only 2D and 4D tensors accepted on input.

Again, **all binary models expect input patches to be in [0;1] range before mean subtraction.**

After you build everything and download the networks run test with ```run_test.sh```. It will download a small test_data.bin file.

### MATLAB

Building Matlab bindings requires a little bit of user intervention. Open matlab/make.m file in Matlab and put your paths to Matlab and include/lib paths of TH and THC, then run ```>> make```. Mex file will be created. 

To initialize the interface do

```
deepcompare('init', 'networks/2ch/2ch_notredame.bin');
```
To reset do

```
deepcompare('reset')
```

To propagate through the network:

```
deepcompare('forward', A)
```
```A``` can be 2D, 3D or 4D array, which is converted inside to 2D or 4D array (Matlab is col-major and Torch is row-major so the array is transposed):

| #dim | matlab dim | torch dim |
| -- | -- | -- |
| 2d | N x B | B x N |
| 3d | 64 x 64 x N | 1 x N x 64 x 64 |
| 4d | 64 x 64 x N x B | B x N x 64 x 64 |

2D or 4D tensor is returned. In case of full network propagation for example the output will be 2D: 1 x B, if input was B x 2 x 64 x 64.

To set the number of GPU to be used (the numbering starts from 1):

```
deepcompare('set_device', 2)
```
Print the network structure:

```
deepcompare('print')
```


### OpenCV

OpenCV example is here to demonstrate how to use the deep CNN models to match image patches, how to preprocess the patches and use the proposed API.

Depends on OpenCV 3.0. To build the example do

```
cd build;
cmake -DWITH_OPENCV=ON -DOpenCV_DIR=/opt/opencv .; make -j8
```
Here ```/opt/opencv``` has to be a folder where OpenCV is built. If you have it installed, you don't need to add it, ```-DWITH_OPENCV=ON``` will be enough.

