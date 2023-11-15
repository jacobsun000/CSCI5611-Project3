cmake_minimum_required(VERSION 3.12)

project(pa3)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP)
find_package(FLTK REQUIRED)

set(INCLUDE_PATH
    include/math
    include/neural
    include
)

include_directories(${INCLUDE_PATH})

set(SOURCE_PATH
    src/path.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_PATH})

if(FLTK_FOUND)
  include_directories(${FLTK_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} ${FLTK_LIBRARIES})
endif()

