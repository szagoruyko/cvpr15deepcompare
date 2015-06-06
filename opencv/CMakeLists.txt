CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/cunnproduction
)

CUDA_ADD_EXECUTABLE(example example.cu ../src/loader.cu)

TARGET_LINK_LIBRARIES(example
  cunnproduction
  TH THC
  opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_features2d)
