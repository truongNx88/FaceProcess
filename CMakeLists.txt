cmake_minimum_required(VERSION 3.7)
project(FaceProcess)

set(CMAKE_CXX_STANDARD 11)

file(GLOB SOURCES *.cpp )
add_executable(${PROJECT_NAME} ${SOURCES})

# OpenCV libs
#find_package(OpenCV REQUIRED)
# set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/local)
# find_package(OpenCV REQUIRED HINTS ${OpenCV_DIR})
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


# ==================== PATHS TO SPECIFY! ==================== #

# Eigen lib headers
include_directories(${CMAKE_SOURCE_DIR}/tf_Include/tensorflow/include/third_party/eigen3)

# TensorFlow headers
include_directories(${CMAKE_SOURCE_DIR}/tf_Include/tensorflow/include)
#include_directories("/usr/local/include/tf/bazel-genfiles/")
include_directories(${CMAKE_SOURCE_DIR}/tf_Include/tensorflow/include/tensorflow)
include_directories(${CMAKE_SOURCE_DIR}/tf_Include/tensorflow/include/third_party)

# Link TensorFlow libs
target_link_libraries(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/tf_Include/tensorflow/lib/libtensorflow_cc.so)
target_link_libraries(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/tf_Include/tensorflow/lib/libtensorflow_framework.so)
