cmake_minimum_required(VERSION 3.16)
project(glim_manual_segmentation_editor LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_WITH_MARCH_NATIVE "Build with -march=native" OFF)
if(BUILD_WITH_MARCH_NATIVE)
  add_definitions(-march=native)
  add_definitions(-DBUILD_WITH_MARCH_NATIVE)
  set(CMAKE_C_FLAGS "-march=native ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
endif()

find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(Eigen3 REQUIRED)
find_package(glim REQUIRED)
find_package(gtsam_points REQUIRED)
find_package(Iridescence REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(spdlog REQUIRED)

add_library(${PROJECT_NAME}_core SHARED
  src/map_cell.cpp
  src/ground_grid_selector.cpp
  src/points_selector.cpp
  src/point_label_layer.cpp
  src/manual_segmentation_editor.cpp
)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(${PROJECT_NAME}_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../glim/thirdparty/json/include
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_core PUBLIC
  Eigen3::Eigen
  glim::glim
  gtsam_points::gtsam_points
  OpenMP::OpenMP_CXX
  spdlog::spdlog
  $<TARGET_NAME_IF_EXISTS:Iridescence::Iridescence>
)
target_link_libraries(${PROJECT_NAME}_core PRIVATE
  ${PCL_LIBRARIES}
)

add_executable(manual_segmentation_editor
  src/manual_segmentation_editor_node.cpp
)
target_link_libraries(manual_segmentation_editor
  ${PROJECT_NAME}_core
  Boost::program_options
)

install(TARGETS ${PROJECT_NAME}_core manual_segmentation_editor
  EXPORT ${PROJECT_NAME}-targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_core)
ament_package()
