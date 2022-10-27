cmake_minimum_required(VERSION 3.23)
project(TensorFlowTest)

set(CMAKE_CXX_STANDARD 14)

add_executable(TensorFlowTest test2.cpp tensorrt_utils.cpp tensorrt_utils.h)

#Specify path to TensorRT


find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(TRT_LIB /usr/lib/x86_64-linux-gnu)
set(TRT_INCLUDE /usr/include/x86_64-linux-gnu)
set(CUDA_INC_DIR /usr/local/cuda/include)

find_library(_NVINFER_LIB nvinfer nvonnxparser HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64 REQUIRED)
set(NVINFER_LIB ${_NVINFER_LIB})

target_link_libraries(TensorFlowTest PRIVATE nvonnxparser ${NVINFER_LIB} ${CUDA_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(TensorFlowTest PRIVATE test_img ${CUDA_INCLUDE_DIRS})