if(COMMAND cmake_policy)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0015 NEW)
endif(COMMAND cmake_policy)

include_directories(../3rdparty/include)
include_directories(../common)

link_directories(../3rdparty/lib)

add_subdirectory(face_detect)
