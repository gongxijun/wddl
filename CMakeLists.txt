cmake_minimum_required(VERSION 3.8)
project(mxnet_wddl)

set(CMAKE_CXX_STANDARD 11)
#add_definitions(-std=c++11)
find_package( PythonInterp 2.7 REQUIRED )
find_package( PythonLibs 2.7 REQUIRED )

set(LIBS  gflags glog zmq protobuf mxnet )

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include )
aux_source_directory(${CMAKE_CURRENT_BINARY_DIR}/src DIR_LIB_SRCS)

set(SOURCE_FILES ${DIR_LIB_SRCS})
add_executable(mxnet_wddl ${SOURCE_FILES})
target_link_libraries(mxnet_wddl ${LIBS})