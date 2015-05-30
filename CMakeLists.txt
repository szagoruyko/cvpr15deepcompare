CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

ADD_SUBDIRECTORY(cunnproduction)

FIND_PACKAGE(CUDA 7.0 REQUIRED)
FIND_PACKAGE(Torch REQUIRED)

add_definitions(-std=c++11)
set(CUDA_NVCC_FLAGS -std=c++11)

INCLUDE_DIRECTORIES(
  "${Torch_INSTALL_INCLUDE}/THC"
  ${OpenCV_INCLUDE_DIRS}
  cunnproduction
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -fPIC")

CUDA_ADD_EXECUTABLE(test_networks test.cpp loader.cpp)
CUDA_ADD_LIBRARY(wrapper SHARED wrapper.cpp loader.cpp)

LINK_DIRECTORIES("${Torch_INSTALL_LIB}")
TARGET_LINK_LIBRARIES(test_networks
  cunnproduction
  TH THC)
