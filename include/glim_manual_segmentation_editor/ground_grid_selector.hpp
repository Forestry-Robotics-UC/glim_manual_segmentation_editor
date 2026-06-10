/*
Copyright 2025 Dahlem Center for Machine Learning and Robotics, Freie Universitat Berlin

Ground selection code adapted from dcmlr/groundgrid ros2-jazzy:
https://github.com/dcmlr/groundgrid/tree/ros2-jazzy

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided
with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to
endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <glim/mapping/sub_map.hpp>
#include <spdlog/spdlog.h>

namespace glim_manual_segmentation_editor {

struct GroundGridSelectorConfig {
  float resolution = 0.33f;
  int point_count_cell_variance_threshold = 30;
  float distance_factor = 0.0001f;
  float min_distance_factor = 0.0005f;
  float min_point_height_threshold = 0.3f;
  float min_point_height_obstacle_threshold = 0.1f;
  float outlier_tolerance = 0.1f;
  float min_ground_patch_detection_point_count_threshold = 0.25f;
  float patch_size_change_distance = 20.0f;
  float occupied_cells_decrease_factor = 5.0f;
  float occupied_cells_point_count_factor = 20.0f;
  float horizontal_point_angular_distance = 0.00174532925f;
  float min_distance_squared = 12.0f;
  int max_grid_cells = 5000000;
};

struct GroundGridSelectionResult {
  std::vector<std::uint64_t> point_ids;
  int cell_count = 0;
  float resolution = 0.0f;
  std::size_t input_point_count = 0;
  std::size_t ground_patch_count = 0;
};

/// @brief GroundGrid-derived selector for static GLIM maps.
class GroundGridSelector {
public:
  static GroundGridSelectionResult select_ground_points(
    const std::vector<glim::SubMap::Ptr>& submaps,
    GroundGridSelectorConfig config,
    const std::shared_ptr<spdlog::logger>& logger);
};

}  // namespace glim_manual_segmentation_editor
