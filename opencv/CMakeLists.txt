CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CUDA 4.0 REQUIRED)
FIND_PACKAGE(Torch REQUIRED)

add_definitions(-std=c++11)
set(CUDA_NVCC_FLAGS -std=c++11)

INCLUDE_DIRECTORIES(
  "${Torch_INSTALL_INCLUDE}/THC"
  ${OpenCV_INCLUDE_DIRS}
  /opt/rocks/cunn-release/
)

CUDA_ADD_EXECUTABLE(example example.cu ../loader.cpp)

LINK_DIRECTORIES("${Torch_INSTALL_LIB}")
TARGET_LINK_LIBRARIES(example
  /opt/rocks/cunn-release/build/libcunnproduction.so
  TH THC
  opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_features2d)
