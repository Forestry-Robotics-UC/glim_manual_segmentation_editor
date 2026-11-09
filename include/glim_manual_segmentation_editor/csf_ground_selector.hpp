#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <glim/mapping/sub_map.hpp>
#include <spdlog/spdlog.h>

namespace glim_manual_segmentation_editor {

struct CsfGroundSelectorConfig {
  float cloth_resolution = 0.5f;
  float classification_threshold = 0.3f;
  float time_step = 0.1f;
  float height_offset = 1.0f;
  float smoothness = 0.2f;
  float stop_threshold = 0.005f;
  int max_iterations = 500;
  int smoothing_iterations = 2;
  int max_grid_cells = 500000;
};

struct CsfGroundSelectionResult {
  std::vector<std::uint64_t> point_ids;
  int rows = 0;
  int cols = 0;
  float resolution = 0.0f;
  int iterations = 0;
  std::size_t input_point_count = 0;
  std::size_t valid_cell_count = 0;
};

/// @brief Cloth Simulation Filter style ground selector for static point-cloud maps.
class CsfGroundSelector {
public:
  static CsfGroundSelectionResult select_ground_points(
    const std::vector<glim::SubMap::Ptr>& submaps,
    CsfGroundSelectorConfig config,
    const std::shared_ptr<spdlog::logger>& logger);
};

}  // namespace glim_manual_segmentation_editor
