Code for CVPR15 paper "Learning to Compare Image Patches via Convolutional Neural Networks"
-----
This package allows researches to apply the described networks to match image patches and extract corresponding patches.

We tried to make the code as easy to use as possible. The original models were trained with Torch ( http://torch.ch ) and we release the models in Torch7 format, and in addition to that we have C++ bindings which do not require Torch installation. Thus we provide example code how to use the models in MATLAB and with OpenCV http://opencv.org

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
6 Avenue Blaise Pascal - Cite Descartes, Champs-sur-Marne, 77455 Marne-la-Vallee cedex 2, France, Office B003
Tel : +33164152173, Fax: +33164152186
for commercial licensing opportunities, or for further distribution, modification or license rights.

Created by Sergey Zagoruyko and Nikos Komodakis. http://imagine.enpc.fr/~komodakn/

Please cite the paper below if you use this code in your research.

Sergey Zagoruyko, Nikos Komodakis
"Learning to Compare Image Patches via Convolutional Neural Networks",
ArXiv:1504.03641. http://arxiv.org/abs/1504.03641

Installation
-----
The code was tested to work in Linux (Ubuntu 14.04) and OS X 10.10, although we release all the source code to enable usage in other operating systems.

### Models

We provide the models in 3 formats, two are Torch7 "nn" and "cudnn" formats and one is binary format with weights only.

#### nn format

Models in nn format can be loaded and used without CUDA support in Torch. To enable CUDA support ```model:cuda()``` call required.

| Train Set | 2ch | 2ch2stream | 2chdeep | siam | siam2stream |
| --- |  --- |  --- |  --- |  --- |  --- |
| yosemite | [nn (3.49 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_yosemite_nn.t7) | [nn (9.74 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_yosemite_nn.t7) | [nn (4.15 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_yosemite_nn.t7) | [nn (7.95 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_yosemite_nn.t7) | [nn (22.36 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_yosemite_nn.t7) |
| notredame | [nn (3.49 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_notredame_nn.t7) | [nn (9.74 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_notredame_nn.t7) | [nn (4.15 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_notredame_nn.t7) | [nn (7.95 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_notredame_nn.t7) | [nn (22.36 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_notredame_nn.t7) |
| liberty | [nn (3.49 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_liberty_nn.t7) | [nn (9.74 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_liberty_nn.t7) | [nn (4.15 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_liberty_nn.t7) | [nn (7.95 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_liberty_nn.t7) | [nn (22.36 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_liberty_nn.t7) |

#### cudnn format

Model in cudnn format are faster, but need a special library from NVIDIA. Check https://github.com/soumith/cudnn.torch 

| Train Set | 2ch | 2ch2stream | 2chdeep | siam | siam2stream |
| --- |  --- |  --- |  --- |  --- |  --- |
| yosemite | [cudnn (3.75 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_yosemite_cudnn.t7) | [cudnn (9.85 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_yosemite_cudnn.t7) | [cudnn (4.15 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_yosemite_cudnn.t7) | [cudnn (5.62 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_yosemite_cudnn.t7) | [cudnn (17.46 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_yosemite_cudnn.t7) |
| notredame | [cudnn (3.75 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_notredame_cudnn.t7) | [cudnn (9.85 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_notredame_cudnn.t7) | [cudnn (4.15 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_notredame_cudnn.t7) | [cudnn (5.62 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_notredame_cudnn.t7) | [cudnn (17.71 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_notredame_cudnn.t7) |
| liberty | [cudnn (3.75 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2ch/2ch_liberty_cudnn.t7) | [cudnn (9.85 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2ch2stream/2ch2stream_liberty_cudnn.t7) | [cudnn (4.15 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/2chdeep/2chdeep_liberty_cudnn.t7) | [cudnn (5.62 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/siam/siam_liberty_cudnn.t7) | [cudnn (17.71 MB)](https://dl.dropboxusercontent.com/u/44617616/networks/siam2stream/siam2stream_liberty_cudnn.t7) |

#### binary format

Same table, not here yet.

### C++ code

We release CUDA code for the time being, CPU code might be added in the future. To install it you need to have CUDA 7.0 with the up-to-date CUDA driver.
Clone and compile this repository it with:

```
git clone --recursive https://github.com/szagoruyko/cvpr15deepmatch
cd cvpr15deepmatch
mkdir build; cd build;
cmake .. -DCMAKE_INSTALL_PREFIX=../install
make -j4 install
```