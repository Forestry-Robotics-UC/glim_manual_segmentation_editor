/**
 * Code adapted from koide3/glim (MIT https://github.com/koide3/glim/blob/master/LICENSE)
 * https://github.com/koide3/glim/blob/master/include/glim/viewer/editor/points_selector.hpp
 */
#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <glim/mapping/sub_map.hpp>
#include <glim_manual_segmentation_editor/ground_grid_selector.hpp>
#include <glim_manual_segmentation_editor/map_cell.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <gtsam_points/segmentation/min_cut.hpp>
#include <gtsam_points/segmentation/region_growing.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/util/vector3i_hash.hpp>
#include <spdlog/spdlog.h>

namespace guik {
class ModelControl;
}

namespace glim_manual_segmentation_editor {

/// @brief Interactive point selection UI copied out of GLIM so upstream code stays untouched.
class PointsSelector {
public:
  explicit PointsSelector(std::shared_ptr<spdlog::logger> logger);
  ~PointsSelector();

  void clear();
  void set_submaps(const std::vector<glim::SubMap::Ptr>& submaps);
  void draw_ui();

  const std::vector<std::uint64_t>& selected_point_ids_ref() const;
  const std::vector<glk::PointCloudBuffer::Ptr>& submap_drawables_ref() const;
  void clear_selected_points();

private:
  std::vector<std::uint64_t> collect_neighbor_point_ids(const Eigen::Vector3d& point);
  gtsam_points::PointCloudCPU::Ptr collect_submap_points(const std::vector<std::uint64_t>& point_ids);

  void update_cells();
  void select_points_tool();
  void select_points_radius();
  void select_outlier_points_radius();
  void select_points_segmentation();
  void select_ground_points();

private:
  std::vector<glim::SubMap::Ptr> submaps;
  std::vector<glk::PointCloudBuffer::Ptr> submap_drawables;

  std::unordered_map<Eigen::Vector3i, MapCell::Ptr, gtsam_points::Vector3iHash> map_cells;

  std::vector<std::uint64_t> selected_point_ids;

  Eigen::Vector3d picked_point;
  std::unique_ptr<guik::ModelControl> model_control;

  int num_threads;
  float map_cell_resolution;
  int cell_selection_window;

  bool show_preferences_window;
  bool show_picked_point;
  bool show_selection_radius;

  bool show_cells;
  bool show_gizmo;
  int selected_tool;

  float select_radius;
  float outlier_radius_offset;
  int outliner_num_neighbors;
  float outlier_stddev_thresh;

  bool show_segmentation_radius;
  int segmentation_method;
  gtsam_points::MinCutParams min_cut_params;
  gtsam_points::RegionGrowingParams region_growing_params;

  GroundGridSelectorConfig ground_grid_config;

  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim_manual_segmentation_editor
