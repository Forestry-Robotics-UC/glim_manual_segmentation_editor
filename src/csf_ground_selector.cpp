#include <glim_manual_segmentation_editor/csf_ground_selector.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <utility>

namespace glim_manual_segmentation_editor {
namespace {

using MatrixXfRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct IndexedPoint {
  Eigen::Vector3f point;
  std::uint64_t id = 0;
  int row = 0;
  int col = 0;
};

struct ClothGrid {
  int rows = 0;
  int cols = 0;
  float resolution = 0.0f;
  float inversion_height = 0.0f;
  Eigen::Vector2f origin = Eigen::Vector2f::Zero();
  MatrixXfRowMajor terrain;
  MatrixXfRowMajor valid;
  MatrixXfRowMajor cloth;
  MatrixXfRowMajor next_cloth;
};

std::uint64_t pack_point_id(int submap_id, int point_id) {
  return (static_cast<std::uint64_t>(submap_id) << 32) | static_cast<std::uint64_t>(point_id);
}

bool point_to_grid_index(const ClothGrid& grid, const Eigen::Vector3f& point, int& row, int& col) {
  col = static_cast<int>(std::floor((point.x() - grid.origin.x()) / grid.resolution));
  row = static_cast<int>(std::floor((point.y() - grid.origin.y()) / grid.resolution));
  return row >= 0 && row < grid.rows && col >= 0 && col < grid.cols;
}

void fill_empty_terrain(ClothGrid& grid, float fallback_height) {
  std::deque<std::pair<int, int>> pending;
  for (int row = 0; row < grid.rows; ++row) {
    for (int col = 0; col < grid.cols; ++col) {
      if (grid.valid(row, col) != 0.0f) {
        pending.emplace_back(row, col);
      } else {
        grid.terrain(row, col) = 0.0f;
      }
    }
  }

  while (!pending.empty()) {
    const auto [row, col] = pending.front();
    pending.pop_front();

    for (int drow = -1; drow <= 1; ++drow) {
      for (int dcol = -1; dcol <= 1; ++dcol) {
        if (drow == 0 && dcol == 0) {
          continue;
        }

        const int nr = row + drow;
        const int nc = col + dcol;
        if (nr < 0 || nc < 0 || nr >= grid.rows || nc >= grid.cols || grid.valid(nr, nc) != 0.0f) {
          continue;
        }

        grid.terrain(nr, nc) = grid.terrain(row, col);
        grid.valid(nr, nc) = 1.0f;
        pending.emplace_back(nr, nc);
      }
    }
  }

  if (grid.valid.sum() == 0.0f) {
    grid.terrain = MatrixXfRowMajor::Constant(grid.rows, grid.cols, fallback_height);
    grid.valid = MatrixXfRowMajor::Constant(grid.rows, grid.cols, 1.0f);
  }
}

float clamp_to_terrain(float cloth_height, float terrain_height) {
  return std::max(cloth_height, terrain_height);
}

int simulate_cloth(ClothGrid& grid, const CsfGroundSelectorConfig& config) {
  const float max_terrain = grid.terrain.maxCoeff();
  grid.cloth = MatrixXfRowMajor::Constant(grid.rows, grid.cols, max_terrain + config.height_offset);
  grid.next_cloth = grid.cloth;

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    MatrixXfRowMajor previous = grid.cloth;

    for (int row = 0; row < grid.rows; ++row) {
      for (int col = 0; col < grid.cols; ++col) {
        grid.cloth(row, col) = clamp_to_terrain(grid.cloth(row, col) - config.time_step, grid.terrain(row, col));
      }
    }

    for (int smooth_iter = 0; smooth_iter < config.smoothing_iterations; ++smooth_iter) {
      grid.next_cloth = grid.cloth;
      for (int row = 0; row < grid.rows; ++row) {
        for (int col = 0; col < grid.cols; ++col) {
          float sum = 0.0f;
          int count = 0;
          for (int drow = -1; drow <= 1; ++drow) {
            for (int dcol = -1; dcol <= 1; ++dcol) {
              if (drow == 0 && dcol == 0) {
                continue;
              }

              const int nr = row + drow;
              const int nc = col + dcol;
              if (nr < 0 || nc < 0 || nr >= grid.rows || nc >= grid.cols) {
                continue;
              }

              sum += grid.cloth(nr, nc);
              ++count;
            }
          }

          if (count == 0) {
            continue;
          }

          const float average = sum / static_cast<float>(count);
          const float smoothed = grid.cloth(row, col) + config.smoothness * (average - grid.cloth(row, col));
          grid.next_cloth(row, col) = clamp_to_terrain(smoothed, grid.terrain(row, col));
        }
      }

      grid.cloth.swap(grid.next_cloth);
    }

    const float max_delta = (grid.cloth - previous).cwiseAbs().maxCoeff();
    if (max_delta < config.stop_threshold) {
      ++iterations;
      break;
    }
  }

  return iterations;
}

std::vector<std::uint64_t>
classify_ground_points(const ClothGrid& grid, const CsfGroundSelectorConfig& config, const std::vector<IndexedPoint>& indexed_points) {
  std::vector<std::uint64_t> ground_ids;
  ground_ids.reserve(indexed_points.size() / 2);

  for (const auto& indexed : indexed_points) {
    const float cloth_original_z = grid.inversion_height - grid.cloth(indexed.row, indexed.col);
    const float distance = indexed.point.z() - cloth_original_z;
    if (std::abs(distance) <= config.classification_threshold) {
      ground_ids.emplace_back(indexed.id);
    }
  }

  return ground_ids;
}

}  // namespace

CsfGroundSelectionResult CsfGroundSelector::select_ground_points(
  const std::vector<glim::SubMap::Ptr>& submaps,
  CsfGroundSelectorConfig config,
  const std::shared_ptr<spdlog::logger>& logger) {
  CsfGroundSelectionResult result;

  config.cloth_resolution = std::max(config.cloth_resolution, 0.05f);
  config.classification_threshold = std::max(config.classification_threshold, 0.01f);
  config.time_step = std::max(config.time_step, 0.001f);
  config.height_offset = std::max(config.height_offset, 0.0f);
  config.smoothness = std::max(0.0f, std::min(config.smoothness, 1.0f));
  config.stop_threshold = std::max(config.stop_threshold, 0.0f);
  config.max_iterations = std::max(config.max_iterations, 1);
  config.smoothing_iterations = std::max(config.smoothing_iterations, 0);
  config.max_grid_cells = std::max(config.max_grid_cells, 10000);

  Eigen::Vector2f min_xy(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
  Eigen::Vector2f max_xy(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
  float min_z = std::numeric_limits<float>::max();
  float max_z = std::numeric_limits<float>::lowest();
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
      max_z = std::max(max_z, world_point.z());
    }
  }

  result.input_point_count = total_points;
  if (total_points == 0 || !min_xy.allFinite() || !max_xy.allFinite() || !std::isfinite(min_z) || !std::isfinite(max_z)) {
    if (logger) {
      logger->warn("CSF selection skipped: no valid map points");
    }
    return result;
  }

  const float padding = std::max(config.cloth_resolution * 3.0f, 1.0f);
  const Eigen::Vector2f span = max_xy - min_xy + Eigen::Vector2f::Constant(2.0f * padding);

  float resolution = config.cloth_resolution;
  int cols = std::max(3, static_cast<int>(std::ceil(span.x() / resolution)) + 1);
  int rows = std::max(3, static_cast<int>(std::ceil(span.y() / resolution)) + 1);

  const double requested_cells = static_cast<double>(rows) * static_cast<double>(cols);
  if (requested_cells > static_cast<double>(config.max_grid_cells)) {
    const float scale = std::sqrt(static_cast<float>(requested_cells) / static_cast<float>(config.max_grid_cells));
    resolution *= scale;
    cols = std::max(3, static_cast<int>(std::ceil(span.x() / resolution)) + 1);
    rows = std::max(3, static_cast<int>(std::ceil(span.y() / resolution)) + 1);

    if (logger) {
      logger->warn(
        "CSF cloth would use {:.0f} cells; coarsening resolution from {:.3f}m to {:.3f}m",
        requested_cells,
        config.cloth_resolution,
        resolution);
    }
  }

  ClothGrid grid;
  grid.rows = rows;
  grid.cols = cols;
  grid.resolution = resolution;
  grid.inversion_height = max_z;
  grid.origin = min_xy - Eigen::Vector2f::Constant(padding);
  grid.terrain = MatrixXfRowMajor::Constant(rows, cols, std::numeric_limits<float>::lowest());
  grid.valid = MatrixXfRowMajor::Zero(rows, cols);

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

      const float inverted_z = grid.inversion_height - world_point.z();
      grid.terrain(indexed.row, indexed.col) = std::max(grid.terrain(indexed.row, indexed.col), inverted_z);
      grid.valid(indexed.row, indexed.col) = 1.0f;
      indexed_points.emplace_back(indexed);
    }
  }

  result.valid_cell_count = static_cast<std::size_t>(grid.valid.sum());
  result.rows = rows;
  result.cols = cols;
  result.resolution = resolution;
  if (indexed_points.empty() || result.valid_cell_count == 0) {
    if (logger) {
      logger->warn("CSF selection skipped: no valid points inside the cloth grid");
    }
    return result;
  }

  fill_empty_terrain(grid, max_z - min_z);
  result.iterations = simulate_cloth(grid, config);
  result.point_ids = classify_ground_points(grid, config, indexed_points);
  return result;
}

}  // namespace glim_manual_segmentation_editor
