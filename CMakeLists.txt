cmake_minimum_required(VERSION 2.8)
project(bar_problem)

#There are lots of scripts with cmake
#for finding external libraries. 
#see /usr/local/share/cmake-2.6/Modules/Find*.cmake for more examples
#find_package(Boost COMPONENTS system filesystem REQUIRED)
#include_directories( ${Boost_INCLUDE_DIRS} )
include_directories(include)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -I /usr/include/eigen3/")
add_subdirectory(include)

#link_directories(libraries)

set( LIBS_TO_LINK Utilities Learning Domains Agents POMDPs)# -llapack
#set( LIBS_TO_LINK POMDPs)# -llapack

add_executable(testMultiNightBarQ testMultiNightBarQ.cpp)
target_link_libraries( testMultiNightBarQ ${LIBS_TO_LINK} )
