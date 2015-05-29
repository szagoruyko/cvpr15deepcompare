// Copyright 2015 Sergey Zagoruyko, Nikos Komodakis
// sergey.zagoruyko@imagine.enpc.fr, nikos.komodakis@enpc.fr
// Ecole des Ponts ParisTech, Universite Paris-Est, IMAGINE
//
// The software is free to use only for non-commercial purposes.
// IF YOU WOULD LIKE TO USE IT FOR COMMERCIAL PURPOSES, PLEASE CONTACT
// Prof. Nikos Komodakis (nikos.komodakis@enpc.fr)
#include <iostream>
#include <cunn.h>
#include "loader.h"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"


/*
void printTensorSize(THCState *state, THCudaTensor *tensor)
{
  long ndim = THCudaTensor_nDimension(state, tensor);
  for(long i = 0; i<ndim; ++i)
    std::cout << tensor->size[i] << " ";
  std::cout << std::endl;
}

void printTensor(THCState* state, THCudaTensor *t)
{
  THFloatTensor *ft = THFloatTensor_newWithSize1d(THCudaTensor_nElement(state, t));
  THFloatTensor_copyCuda(state, ft, t);

  float* data = THFloatTensor_data(ft);
  for (long i=0; i<THFloatTensor_nElement(ft); ++i)
    std::cout << data[i] << " ";
  std::cout << std::endl;

  THFloatTensor_free(ft);
}
*/


int main(int argc, char** argv)
{
  THCState *state = (THCState*)malloc(sizeof(THCState));
  THCudaInit(state);

  FILE *f = fopen("test_data.bin", "rb");
  int num = 0;
  int bs = 0; 
  fread(&num, sizeof(int), 1, f);
  fread(&bs, sizeof(int), 1, f);

  // read test input
  THCudaTensor* input = THCudaTensor_newWithSize4d(state, bs,2,64,64);
  THFloatTensor *finput = THFloatTensor_newWithSize4d(bs,2,64,64);
  fread(THFloatTensor_data(finput), sizeof(float) * THFloatTensor_nElement(finput), 1, f);
  THCudaTensor_copyFloat(state, input, finput);

  for(int i=0; i < num; ++i)
  {
    // read network bin file path
    int namesize = 0;
    fread(&namesize, sizeof(int), 1, f);
    std::vector<unsigned char> name_v(namesize);
    fread(name_v.data(), sizeof(char), namesize, f);
    std::string name(name_v.begin(), name_v.end());

    // read test output
    int v[2];
    fread(v, sizeof(int), 2, f);
    THFloatTensor *foutput = THFloatTensor_newWithSize2d(v[0], v[1]);
    fread(THFloatTensor_data(foutput), sizeof(float)*v[0]*v[1], 1, f);

    THCudaTensor* ref_output = THCudaTensor_newWithSize2d(state, v[0], v[1]);
    THCudaTensor_copyFloat(state, ref_output, foutput);

    // load the network
    auto net = loadNetwork(state, name.c_str());

    THCudaTensor* output;
   
    if(name.find("_desc") != std::string::npos)
    {
      // for desc we have to do input:select(2,1)
      THCudaTensor* patch_src = THCudaTensor_newSelect(state, input, 1, 0);
      THCudaTensor* patch = THCudaTensor_newWithSize4d(state, bs, 1, 64, 64);

      THCudaTensor_copy(state, patch, patch_src);

      output = net->forward(patch);

      THCudaTensor_free(state, patch);
      THCudaTensor_free(state, patch_src);
    }
    else
      output = net->forward(input);

    if(THCudaTensor_dist(state, ref_output, output, 1) == 0)
      std::cout << ANSI_COLOR_GREEN << "PASSED" << ANSI_COLOR_RESET;
    else
      std::cout << ANSI_COLOR_RED << "FAILED" << ANSI_COLOR_RESET;
    std::cout << "	" << name << std::endl;

    THCudaTensor_free(state, ref_output);
    THFloatTensor_free(foutput);
  }

  fclose(f);

  THFloatTensor_free(finput);
  THCudaTensor_free(state, input);

  THCudaShutdown(state);

  return 0;
}
