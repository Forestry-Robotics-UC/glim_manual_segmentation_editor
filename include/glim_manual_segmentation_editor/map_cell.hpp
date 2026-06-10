/**
 * Code adapted from koide3/glim (MIT https://github.com/koide3/glim/blob/master/LICENSE)
 * https://github.com/koide3/glim/blob/master/include/glim/viewer/editor/map_cell.hpp
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace glim_manual_segmentation_editor {

/// @brief A cell storing packed submap point indices for quick spatial queries.
struct MapCell {
public:
  using Ptr = std::shared_ptr<MapCell>;
  using ConstPtr = std::shared_ptr<const MapCell>;

  MapCell(double resolution, const Eigen::Vector3i& coord);
  ~MapCell();

  std::string name() const;

  void clear();
  void add_point(int submap_id, int point_id);
  void add_points(int submap_id, const std::vector<int>& point_ids);

  void remove_submap(int submap_id);
  void remove_submaps(const std::vector<int>& submap_ids);

public:
  const double resolution;
  const Eigen::Vector3i coord;

  std::vector<std::uint64_t> point_ids;
};

}  // namespace glim_manual_segmentation_editor
