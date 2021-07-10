if(COMMAND cmake_policy)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0015 NEW)
endif(COMMAND cmake_policy)

aux_source_directory(./mtcnn MTCNN_SRCS)
aux_source_directory(./ DPFACE_SRCS)

add_library(dpface SHARED ${MTCNN_SRCS} ${DPFACE_SRCS})
add_library(dpface_static STATIC ${MTCNN_SRCS} ${DPFACE_SRCS})
set_target_properties (dpface_static PROPERTIES OUTPUT_NAME "dpface")
set_target_properties (dpface_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

target_link_libraries(dpface dpcnn MNN opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs) 
