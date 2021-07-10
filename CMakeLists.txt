cmake_minimum_required(VERSION 2.4)
project(deeperception C CXX )

add_definitions(-DUSE_OPENCV)

set(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++11)

SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_BUILD_TYPE "Debug")
#add_definitions(-D_DEBUG)

SET(CMAKE_CXX_FLAGS "-fPIC")


add_subdirectory(dpcnn)
add_subdirectory(dpvision)
add_subdirectory(test)
