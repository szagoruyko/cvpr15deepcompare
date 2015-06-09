CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

ADD_SUBDIRECTORY(cunnproduction)

FIND_PACKAGE(CUDA 7.0 REQUIRED)
FIND_PACKAGE(Torch REQUIRED)

ADD_DEFINITIONS(-std=c++11 -fPIC)

INCLUDE_DIRECTORIES(
  ${Torch_INSTALL_INCLUDE}/THC
  ${Torch_INSTALL_INCLUDE}/TH
  ${CUDA_INCLUDE_DIRS}
  cunnproduction
)

LINK_DIRECTORIES("${Torch_INSTALL_LIB}")
ADD_LIBRARY(loader_static STATIC src/loader.cpp)
ADD_LIBRARY(wrapper_static STATIC src/wrapper.cpp)

ADD_EXECUTABLE(test_networks src/test.cpp)

IF(WITH_OPENCV)
  ADD_SUBDIRECTORY(opencv)
ENDIF()

TARGET_LINK_LIBRARIES(test_networks cunnproduction_static TH THC loader_static)
TARGET_LINK_LIBRARIES(loader_static TH THC cunnproduction_static)
TARGET_LINK_LIBRARIES(wrapper_static TH THC cunnproduction_static loader_static)
