cmake_minimum_required(VERSION 3.20)
project(test_new_image)

set(CMAKE_CXX_STANDARD 14)

# Pangolin
find_package(Pangolin REQUIRED)
# Eigen
include_directories("/usr/include/eigen3")
# opencv
set(OpenCV_DIR "/home/zed/opencv/opencv/build")
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# 打印消息
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
#message(STATUS "    libraries: ${OpenCV_LIBS}")
#message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(test_new_image main.cpp)

target_link_libraries(test_new_image ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
