if(COMMAND cmake_policy)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0015 NEW)
endif(COMMAND cmake_policy)

include_directories(../3rdparty/include)

link_directories(../3rdparty/lib)

aux_source_directory(./ SRCS)

add_executable(test_example ${SRCS})

target_link_libraries(test_example dpcnn MNN dpface opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_videoio dl) 
