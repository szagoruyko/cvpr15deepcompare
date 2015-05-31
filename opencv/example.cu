// Copyright 2015 Sergey Zagoruyko, Nikos Komodakis
// sergey.zagoruyko@imagine.enpc.fr, nikos.komodakis@enpc.fr
// Ecole des Ponts ParisTech, Universite Paris-Est, IMAGINE
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
#include "loader.h"

// defined by network architecture
#define M 64

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
    // increase the size of the region to include some context
    cv::getRectSubPix(image, cv::Size(it.size*1.3, it.size*1.3), it.pt, buf);
    cv::Scalar m = cv::mean(buf);
    cv::resize(buf, patch, cv::Size(M,M));
    patch.convertTo(patch, CV_32F, 1./255.);
    patch = patch.isContinuous() ? patch : patch.clone();
    // mean subtraction is crucial!
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

  // initialize 4D CUDA tensor and copy patches into it
  THCudaTensor *input = THCudaTensor_newWithSize4d(state, N, 1, M, M);
  THCudaTensor_copyFloat(state, input, buffer);

  // propagate through the network
  THCudaTensor *output = net->forward(input);

  // copy descriptors back
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
  THCState *state = (THCState*)malloc(sizeof(THCState));
  THCudaInit(state);

  const char *network_path = "/opt/projects/deepfeat/release/networks/siam/siam_desc_notredame.bin";
  auto net = loadNetwork(state, network_path);

  // load the images
  cv::Mat ima = cv::imread("../100_7100.JPG");
  cv::Mat imb = cv::imread("../100_7101.JPG");

  if(ima.empty() || imb.empty())
  {
    std::cout << "images not found\n";
    return 1;
  }

  cv::Mat ima_gray, imb_gray;
  cv::cvtColor(ima, ima_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(imb, imb_gray, cv::COLOR_BGR2GRAY);

  // Here we set min_area parameter to a bigger value, like that minimal size
  // of a patch will be around 11x11, because the network was trained on bigger patches
  // this parameter is important in practice
  cv::Ptr<cv::MSER> detector = cv::MSER::create(5, 120);
  std::vector<cv::KeyPoint> kpa, kpb;
  detector->detect(ima_gray, kpa);
  detector->detect(imb_gray, kpb);
  std::cout << "image A MSER points detected: " << kpa.size() << std::endl;
  std::cout << "image B MSER points detected: " << kpb.size() << std::endl;

  std::vector<cv::Mat> patches_a, patches_b;
  extractPatches(ima_gray, kpa, patches_a);
  extractPatches(imb_gray, kpb, patches_b);

  cv::Mat descriptors_a, descriptors_b;
  extractDescriptors(state, net, patches_a, descriptors_a);
  extractDescriptors(state, net, patches_b, descriptors_b);

  cv::FlannBasedMatcher matcher;
  std::vector<cv::DMatch> matches;
  matcher.match( descriptors_a, descriptors_b, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_a.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );


  std::vector<cv::DMatch> good_matches;
  for( int i = 0; i < descriptors_a.rows; i++ )
  { if( matches[i].distance <= max(4*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
  }

  //-- Draw only "good" matches
  cv::Mat img_matches;
  cv::drawMatches( ima, kpa, imb, kpb,
               good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  for(auto &it : kpa)
    cv::circle(ima, cv::Point(it.pt.x, it.pt.y), it.size, cv::Scalar(255,255,0));
  for(auto &it : kpb)
    cv::circle(imb, cv::Point(it.pt.x, it.pt.y), it.size, cv::Scalar(255,255,0));

  cv::imshow("matches", img_matches);
  cv::imshow("keypoints image 1", ima);
  cv::imshow("keypoints image 2", imb);
  cv::waitKey();
  THCudaShutdown(state);

  return 0;
}
