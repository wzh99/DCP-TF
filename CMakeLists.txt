cmake_minimum_required(VERSION 3.0)
project(RegTest)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL 1.9 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(TBB REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} ${HDF5_DEFINITIONS} ${TBB_DEFINITIONS})

aux_source_directory(cxx SOURCES)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${HDF5_LIBRARIES}
    tbb)