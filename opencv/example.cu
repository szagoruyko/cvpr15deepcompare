// Copyright 2015 Sergey Zagoruyko, Nikos Komodakis
// Ecole des Ponts ParisTech, Universite Paris-Est
//
// The software is free to use only for non-commercial purposes.
// IF YOU WOULD LIKE TO USE IT FOR COMMERCIAL PURPOSES, PLEASE CONTACT
// Prof. Nikos Komodakis (nikos.komodakis@enpc.fr)
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <THC/THC.h>
#include <cunn.h>

// defined by network architecture
#define M 64

// defined in loader.cpp
cunn::Sequential::Ptr
loadNetwork(THCState* state, const char* filename);

// Given an image an coordinates+sizes of detected points
// extract corresponding image patches with OpenCV functions
// input image is in [0 255] range,
// the patches are divided by 255 and mean-normalized
// Note: depending on the type of applications you might want to use 
// orientation of detected region or inrease the cropped bounding box by some
// constant
// Another note: this is of course not the fastest way to extracted features.
// The ultimate would be to use CUDA texture memory and process all the features
// from am image in parallel
void extractPatches(const cv::Mat& image,
    const std::vector<cv::KeyPoint>& kp,
    std::vector<cv::Mat>& patches)
{
  for(auto &it : kp)
  {
    cv::Mat patch(M, M, CV_32F);
    cv::Mat buf;
    cv::getRectSubPix(image, cv::Size(it.size, it.size), it.pt, buf);
    cv::Scalar m = cv::mean(buf);
    cv::resize(buf, patch, cv::Size(M,M));
    patch.convertTo(patch, CV_32F, 1./255.);
    patch = patch.isContinuous() ? patch : patch.clone();
    patches.push_back(patch - m[0]/255.);
  }
}

// Copy extracted patches to CUDA memory and run the network
// One has to keep mind that GPU memory is limited and extracting too many patches
// at once might cause troubles
// So if you need to extract a lot of patches, an efficient way would be to
// devide the set in smaller equal parts and preallocate CPU and GPU memory
void extractDescriptors(THCState *state,
    cunn::Sequential::Ptr net,
    const std::vector<cv::Mat>& patches,
    cv::Mat& descriptors)
{
  size_t N = patches.size();
  THFloatTensor *buffer = THFloatTensor_newWithSize4d(N, 1, M, M);
  float *data = THFloatTensor_data(buffer);

  for(size_t i = 0; i < N; ++i)
    memcpy(data + i*M*M, patches[i].data, sizeof(float) * M * M);

  THCudaTensor *input = THCudaTensor_newWithSize4d(state, N, 1, M, M);
  THCudaTensor_copyFloat(state, input, buffer);
  THCudaTensor *output = net->forward(input);

  THFloatTensor *desc = THFloatTensor_newWithSize2d(N, output->size[1]);
  THFloatTensor_copyCuda(state, desc, output);

  descriptors.create(N, output->size[1], CV_32F);
  memcpy(descriptors.data, THFloatTensor_data(desc), sizeof(float) * N * output->size[1]);

  THCudaTensor_free(state, input);
  THFloatTensor_free(buffer);
  THFloatTensor_free(desc);
}


int main(int argc, char** argv)
{
  cv::Mat im = cv::imread("../beach.jpg");

  THCState *state = (THCState*)malloc(sizeof(THCState));
  THCudaInit(state);

  const char *network_path = "/opt/projects/deepfeat/release/networks/siam2stream/siam2stream_desc_notredame.bin";

  auto net = loadNetwork(state, network_path);

  // Here we set min_area parameter to a bigger value, like that minimal size
  // of a patch will be around 11x11, because the network was trained on bigger patches
  // this parameter is important in practice
  cv::Ptr<cv::MSER> detector = cv::MSER::create(5, 120);
  std::vector<cv::KeyPoint> kp;
  detector->detect(im, kp);
  std::cout << "MSER points detected: " << kp.size() << std::endl;

  cv::Mat im_gray;
  cv::cvtColor(im, im_gray, cv::COLOR_BGR2GRAY);

  std::vector<cv::Mat> patches;
  extractPatches(im_gray, kp, patches);

  cv::Mat descriptors;
  extractDescriptors(state, net, patches, descriptors);

  for(auto &it : kp)
    cv::circle(im, cv::Point(it.pt.x, it.pt.y), it.size, cv::Scalar(255,0,0));

  cv::imshow("desc", descriptors);
  //cv::imshow("im", im);
  cv::waitKey();
  THCudaShutdown(state);

  return 0;
}
