if(COMMAND cmake_policy)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0015 NEW)
endif(COMMAND cmake_policy)

include_directories(../3rdparty/include)
include_directories(../common)
include_directories(../mnn_core/include)
include_directories(../mnn_core/)

link_directories(../3rdparty/lib/opencv)
link_directories(../mnn_core/lib)


aux_source_directory(../common COMMON_SRCS)
aux_source_directory(../mnn_core MNNWRAPPER_SRCS)

add_library(dpcnn SHARED ${COMMON_SRCS} ${MNNWRAPPER_SRCS})
add_library(dpcnn_static STATIC ${COMMON_SRCS} ${MNNWRAPPER_SRCS})
set_target_properties (dpcnn_static PROPERTIES OUTPUT_NAME "dpcnn")
set_target_properties (dpcnn_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

target_link_libraries(dpcnn MNN opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs) 
