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
#include <glim_manual_segmentation_editor/ground_grid_selector.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace glim_manual_segmentation_editor {
namespace {

using MatrixXfRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct IndexedPoint {
  Eigen::Vector3f point;
  std::uint64_t id = 0;
  int row = 0;
  int col = 0;
};

struct Grid {
  int cell_count = 0;
  float resolution = 0.0f;
  Eigen::Vector2f origin = Eigen::Vector2f::Zero();
  Eigen::Vector2f center = Eigen::Vector2f::Zero();
  float initial_ground_height = 0.0f;

  MatrixXfRowMajor expected_points;
  MatrixXfRowMajor points;
  MatrixXfRowMajor mean_height;
  MatrixXfRowMajor m2;
  MatrixXfRowMajor variance;
  MatrixXfRowMajor min_height;
  MatrixXfRowMajor max_height;
  MatrixXfRowMajor ground;
  MatrixXfRowMajor ground_patch;
};

constexpr float kInitialGroundPatchConfidence = 0.0001f;
constexpr float kGroundPatchConfidenceThreshold = 0.01f;

std::uint64_t pack_point_id(int submap_id, int point_id) {
  return (static_cast<std::uint64_t>(submap_id) << 32) | static_cast<std::uint64_t>(point_id);
}

bool point_to_grid_index(const Grid& grid, const Eigen::Vector3f& point, int& row, int& col) {
  row = static_cast<int>(std::floor((point.x() - grid.origin.x()) / grid.resolution));
  col = static_cast<int>(std::floor((point.y() - grid.origin.y()) / grid.resolution));
  return row >= 0 && row < grid.cell_count && col >= 0 && col < grid.cell_count;
}

void init_expected_points(Grid& grid, const GroundGridSelectorConfig& config) {
  grid.expected_points.resize(grid.cell_count, grid.cell_count);
  const float center = static_cast<float>(grid.cell_count) / 2.0f;
  const float angular_distance = std::max(config.horizontal_point_angular_distance, 1.0e-6f);

  for (int row = 0; row < grid.cell_count; ++row) {
    for (int col = 0; col < grid.cell_count; ++col) {
      const float dist = std::hypot(static_cast<float>(row) - center, static_cast<float>(col) - center);
      grid.expected_points(row, col) = std::atan(1.0f / std::max(dist, 1.0e-3f)) / angular_distance;
    }
  }
}

void init_grid_matrices(Grid& grid) {
  const int n = grid.cell_count;
  grid.points = MatrixXfRowMajor::Zero(n, n);
  grid.mean_height = MatrixXfRowMajor::Zero(n, n);
  grid.m2 = MatrixXfRowMajor::Zero(n, n);
  grid.variance = MatrixXfRowMajor::Zero(n, n);
  grid.min_height = MatrixXfRowMajor::Constant(n, n, std::numeric_limits<float>::max());
  grid.max_height = MatrixXfRowMajor::Constant(n, n, std::numeric_limits<float>::lowest());
  grid.ground = MatrixXfRowMajor::Constant(n, n, grid.initial_ground_height);
  grid.ground_patch = MatrixXfRowMajor::Constant(n, n, kInitialGroundPatchConfidence);
}

void insert_point(Grid& grid, const IndexedPoint& indexed) {
  const int row = indexed.row;
  const int col = indexed.col;
  const float z = indexed.point.z();

  float& count = grid.points(row, col);
  float& mean = grid.mean_height(row, col);
  float& m2 = grid.m2(row, col);

  if (count == 0.0f) {
    mean = z;
  }
  const float delta = z - mean;
  mean += delta / (count + 1.0f);
  m2 += delta * (z - mean);

  grid.min_height(row, col) = std::min(grid.min_height(row, col), z - 0.0001f);
  grid.max_height(row, col) = std::max(grid.max_height(row, col), z);
  count += 1.0f;
}

void compute_variance(Grid& grid) {
  grid.variance = grid.m2.array() / (grid.points.array() + std::numeric_limits<float>::min());
}

void detect_ground_patch(Grid& grid, const GroundGridSelectorConfig& config, int row, int col, int patch_size) {
  const int radius = patch_size / 2;
  const float center = static_cast<float>(grid.cell_count) / 2.0f;

  float points_sum = 0.0f;
  float weighted_variance_sum = 0.0f;
  float weighted_ground_sum = 0.0f;
  float local_min = std::numeric_limits<float>::max();

  for (int drow = -radius; drow <= radius; ++drow) {
    for (int dcol = -radius; dcol <= radius; ++dcol) {
      const int r = row + drow;
      const int c = col + dcol;
      const float count = grid.points(r, c);
      points_sum += count;
      weighted_variance_sum += count * grid.variance(r, c);
      if (count > 0.0f) {
        weighted_ground_sum += count * grid.min_height(r, c);
        local_min = std::min(local_min, grid.min_height(r, c));
      }
    }
  }

  const float expected_point_count = grid.expected_points(row, col);
  const float minimum_points =
    std::max(std::floor(config.min_ground_patch_detection_point_count_threshold * patch_size * expected_point_count), 3.0f);
  if (points_sum < minimum_points || !std::isfinite(local_min)) {
    return;
  }

  const float sqdist =
    (std::pow(static_cast<float>(row) - center, 2.0f) + std::pow(static_cast<float>(col) - center, 2.0f)) * std::pow(grid.resolution, 2.0f);
  const float var_threshold_sq = std::min(
    std::max(sqdist * std::pow(config.distance_factor, 2.0f), std::pow(config.min_distance_factor, 2.0f)),
    std::pow(config.min_distance_factor * 10.0f, 2.0f));

  const float center_variance = grid.variance(row, col);
  const float max_variance = grid.points(row, col) >= static_cast<float>(config.point_count_cell_variance_threshold)
                               ? center_variance
                               : weighted_variance_sum / points_sum;
  const float ground_level = weighted_ground_sum / points_sum;

  float& old_confidence = grid.ground_patch(row, col);
  float& old_ground_height = grid.ground(row, col);
  const float ground_diff = std::max((ground_level - old_ground_height) * (2.0f * old_confidence), 1.0f);

  if (old_confidence > 0.5f && ground_level >= old_ground_height + config.outlier_tolerance) {
    return;
  }

  if (var_threshold_sq > std::pow(max_variance, 2.0f) && max_variance > 0.0f &&
      points_sum > (ground_diff * expected_point_count * patch_size) * config.min_ground_patch_detection_point_count_threshold) {
    const float new_confidence = std::min(points_sum / config.occupied_cells_point_count_factor, 1.0f);
    old_ground_height = (ground_level * new_confidence + old_confidence * old_ground_height * 2.0f) / (new_confidence + old_confidence * 2.0f);
    old_confidence = std::min((points_sum / (config.occupied_cells_point_count_factor * 2.0f) + old_confidence) / 2.0f, 1.0f);
  } else if (local_min < old_ground_height) {
    old_ground_height = local_min;
    old_confidence = std::min(old_confidence + 0.1f, 0.5f);
  }
}

void detect_ground_patches(Grid& grid, const GroundGridSelectorConfig& config) {
  compute_variance(grid);

  const float center = static_cast<float>(grid.cell_count) / 2.0f;
  const float patch_change_sq = std::pow(config.patch_size_change_distance, 2.0f);
  for (int row = 2; row < grid.cell_count - 2; ++row) {
    for (int col = 2; col < grid.cell_count - 2; ++col) {
      const float sqdist =
        (std::pow(static_cast<float>(row) - center, 2.0f) + std::pow(static_cast<float>(col) - center, 2.0f)) * std::pow(grid.resolution, 2.0f);
      detect_ground_patch(grid, config, row, col, sqdist <= patch_change_sq ? 3 : 5);
    }
  }
}

std::size_t count_ground_patches(const Grid& grid) {
  std::size_t count = 0;
  for (int row = 0; row < grid.cell_count; ++row) {
    for (int col = 0; col < grid.cell_count; ++col) {
      if (grid.ground_patch(row, col) > kGroundPatchConfidenceThreshold) {
        ++count;
      }
    }
  }
  return count;
}

bool seed_interpolation_center(Grid& grid) {
  const int center = grid.cell_count / 2;
  if (grid.ground_patch(center, center) > kGroundPatchConfidenceThreshold) {
    grid.ground_patch(center, center) = 1.0f;
    return true;
  }

  int best_row = -1;
  int best_col = -1;
  int best_sqdist = std::numeric_limits<int>::max();
  for (int row = 1; row < grid.cell_count - 1; ++row) {
    for (int col = 1; col < grid.cell_count - 1; ++col) {
      if (grid.ground_patch(row, col) <= kGroundPatchConfidenceThreshold) {
        continue;
      }

      const int sqdist = (row - center) * (row - center) + (col - center) * (col - center);
      if (sqdist < best_sqdist) {
        best_sqdist = sqdist;
        best_row = row;
        best_col = col;
      }
    }
  }

  if (best_row < 0 || best_col < 0) {
    return false;
  }

  grid.ground(center, center) = grid.ground(best_row, best_col);
  grid.ground_patch(center, center) = 1.0f;
  return true;
}

void interpolate_cell(Grid& grid, const GroundGridSelectorConfig& config, int row, int col) {
  if (row <= 0 || col <= 0 || row >= grid.cell_count - 1 || col >= grid.cell_count - 1) {
    return;
  }

  float confidence_sum = 0.0f;
  float weighted_height_sum = 0.0f;
  for (int drow = -1; drow <= 1; ++drow) {
    for (int dcol = -1; dcol <= 1; ++dcol) {
      const float confidence = grid.ground_patch(row + drow, col + dcol);
      confidence_sum += confidence;
      weighted_height_sum += confidence * grid.ground(row + drow, col + dcol);
    }
  }

  if (confidence_sum <= std::numeric_limits<float>::min()) {
    return;
  }

  float& height = grid.ground(row, col);
  float& confidence = grid.ground_patch(row, col);
  const float avg_height = weighted_height_sum / confidence_sum;
  height = (1.0f - confidence) * avg_height + confidence * height;

  const int center = grid.cell_count / 2;
  const float sqdist =
    (std::pow(static_cast<float>(row - center), 2.0f) + std::pow(static_cast<float>(col - center), 2.0f)) * std::pow(grid.resolution, 2.0f);
  if (sqdist > config.min_distance_squared) {
    confidence = std::max(confidence - confidence / config.occupied_cells_decrease_factor, 0.001f);
  }
}

void spiral_ground_interpolation(Grid& grid, const GroundGridSelectorConfig& config) {
  const int center = grid.cell_count / 2;
  for (int ring = center - 1; ring >= 1; --ring) {
    const int side_length = (center - ring) * 2;

    for (short side = 0; side < 2; ++side) {
      for (int pos = ring; pos < ring + side_length; ++pos) {
        const int row = side % 2 ? pos : ring;
        const int col = side % 2 ? ring : pos;
        interpolate_cell(grid, config, row, col);
      }
    }

    const int outer = ring + side_length;
    for (short side = 0; side < 2; ++side) {
      for (int pos = outer; pos >= outer - side_length; --pos) {
        const int row = side % 2 ? pos : outer;
        const int col = side % 2 ? outer : pos;
        interpolate_cell(grid, config, row, col);
      }
    }
  }
}

std::vector<std::uint64_t> classify_ground_points(const Grid& grid, const GroundGridSelectorConfig& config, const std::vector<IndexedPoint>& points) {
  std::vector<std::uint64_t> ground_ids;
  ground_ids.reserve(points.size() / 2);

  const float min_distance_factor = config.min_distance_factor * 5.0f;
  for (const auto& indexed : points) {
    const int row = indexed.row;
    const int col = indexed.col;
    if (row <= 2 || col <= 2 || row >= grid.cell_count - 3 || col >= grid.cell_count - 3) {
      continue;
    }

    const float variance = grid.variance(row, col) + std::numeric_limits<float>::min();
    const float distance = (indexed.point.head<2>() - grid.center).norm();
    const float tolerance = std::max(
      std::min((min_distance_factor * distance) / variance * config.min_point_height_threshold, config.min_point_height_threshold),
      config.min_point_height_obstacle_threshold);

    if (indexed.point.z() <= grid.ground(row, col) + tolerance) {
      ground_ids.emplace_back(indexed.id);
    }
  }

  return ground_ids;
}

}  // namespace

GroundGridSelectionResult GroundGridSelector::select_ground_points(
  const std::vector<glim::SubMap::Ptr>& submaps,
  GroundGridSelectorConfig config,
  const std::shared_ptr<spdlog::logger>& logger) {
  GroundGridSelectionResult result;

  config.resolution = std::max(config.resolution, 0.05f);
  config.max_grid_cells = std::max(config.max_grid_cells, 10000);
  config.point_count_cell_variance_threshold = std::max(config.point_count_cell_variance_threshold, 1);
  config.occupied_cells_decrease_factor = std::max(config.occupied_cells_decrease_factor, 1.0f);
  config.occupied_cells_point_count_factor = std::max(config.occupied_cells_point_count_factor, 1.0f);

  Eigen::Vector2f min_xy(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
  Eigen::Vector2f max_xy(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
  float min_z = std::numeric_limits<float>::max();
  std::size_t total_points = 0;

  for (const auto& submap : submaps) {
    if (!submap || !submap->frame || submap->id < 0) {
      continue;
    }

    total_points += submap->frame->size();
    for (int i = 0; i < submap->frame->size(); ++i) {
      const Eigen::Vector4d world_point_h = submap->T_world_origin * submap->frame->points[i];
      const Eigen::Vector3f world_point = world_point_h.head<3>().cast<float>();
      if (!world_point.allFinite()) {
        continue;
      }

      min_xy = min_xy.cwiseMin(world_point.head<2>());
      max_xy = max_xy.cwiseMax(world_point.head<2>());
      min_z = std::min(min_z, world_point.z());
    }
  }

  result.input_point_count = total_points;
  if (total_points == 0 || !min_xy.allFinite() || !max_xy.allFinite() || !std::isfinite(min_z)) {
    if (logger) {
      logger->warn("GroundGrid selection skipped: no valid map points");
    }
    return result;
  }

  const Eigen::Vector2f map_span = max_xy - min_xy;
  const float padding = std::max(config.resolution * 3.0f, 1.0f);
  const float dimension = std::max(map_span.x(), map_span.y()) + 2.0f * padding;

  float resolution = config.resolution;
  int cell_count = std::max(7, static_cast<int>(std::ceil(dimension / resolution)));
  if (cell_count % 2 == 0) {
    ++cell_count;
  }

  const double requested_cells = static_cast<double>(cell_count) * static_cast<double>(cell_count);
  if (requested_cells > static_cast<double>(config.max_grid_cells)) {
    resolution = dimension / std::sqrt(static_cast<float>(config.max_grid_cells));
    cell_count = std::max(7, static_cast<int>(std::ceil(dimension / resolution)));
    if (cell_count % 2 == 0) {
      ++cell_count;
    }
    if (logger) {
      logger->warn(
        "GroundGrid map would use {:.0f} cells; coarsening resolution from {:.3f}m to {:.3f}m",
        requested_cells,
        config.resolution,
        resolution);
    }
  }

  Grid grid;
  grid.cell_count = cell_count;
  grid.resolution = resolution;
  grid.center = (min_xy + max_xy) * 0.5f;
  grid.origin = grid.center - Eigen::Vector2f::Constant(static_cast<float>(cell_count) * resolution * 0.5f);
  grid.initial_ground_height = min_z;
  init_expected_points(grid, config);
  init_grid_matrices(grid);

  std::vector<IndexedPoint> indexed_points;
  indexed_points.reserve(total_points);
  for (const auto& submap : submaps) {
    if (!submap || !submap->frame || submap->id < 0) {
      continue;
    }

    for (int i = 0; i < submap->frame->size(); ++i) {
      const Eigen::Vector4d world_point_h = submap->T_world_origin * submap->frame->points[i];
      const Eigen::Vector3f world_point = world_point_h.head<3>().cast<float>();
      if (!world_point.allFinite()) {
        continue;
      }

      IndexedPoint indexed;
      indexed.point = world_point;
      indexed.id = pack_point_id(submap->id, i);
      if (!point_to_grid_index(grid, indexed.point, indexed.row, indexed.col)) {
        continue;
      }

      insert_point(grid, indexed);
      indexed_points.emplace_back(indexed);
    }
  }

  detect_ground_patches(grid, config);
  result.ground_patch_count = count_ground_patches(grid);
  result.cell_count = cell_count;
  result.resolution = resolution;

  if (result.ground_patch_count == 0) {
    if (logger) {
      logger->warn("GroundGrid selection found no ground patches; try increasing resolution or lowering minimum point-count threshold");
    }
    return result;
  }

  if (!seed_interpolation_center(grid)) {
    if (logger) {
      logger->warn("GroundGrid selection failed to seed ground interpolation");
    }
    return result;
  }

  spiral_ground_interpolation(grid, config);
  result.point_ids = classify_ground_points(grid, config, indexed_points);
  return result;
}

}  // namespace glim_manual_segmentation_editor
