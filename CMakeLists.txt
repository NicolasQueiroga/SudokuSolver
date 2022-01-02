cmake_minimum_required(VERSION 3.00)

project(SudokuSolver VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -O0")
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS_REQUIRED True)

find_package(OpenCV REQUIRED)
include_directories(
  ${AUXILIAR_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(
	auxiliar
	libs/auxiliar/aux.cpp
	)

target_include_directories(auxiliar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(EXECUTABLE_OUTPUT_PATH "../bin")
add_executable(run.exe src/main.cpp)

target_link_libraries(run.exe PRIVATE auxiliar ${OpenCV_LIBRARIES})
add_dependencies(run.exe auxiliar ${OpenCV_LIBRARIES})


add_executable(test.exe model/src/test.cpp)
add_executable(train.exe model/src/train.cpp)

target_link_libraries(test.exe PRIVATE auxiliar ${OpenCV_LIBRARIES})
target_link_libraries(train.exe PRIVATE auxiliar ${OpenCV_LIBRARIES})
add_dependencies(test.exe auxiliar ${OpenCV_LIBRARIES})
add_dependencies(train.exe auxiliar ${OpenCV_LIBRARIES})