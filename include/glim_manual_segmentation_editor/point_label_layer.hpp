#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <glim/mapping/sub_map.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <spdlog/spdlog.h>

namespace glim_manual_segmentation_editor {

struct PointClassInfo {
  int id;
  std::string name;
  Eigen::Vector4f color;
};

enum class PointDisplayMode {
  SEGMENTATION = 0,
  INTENSITY = 1,
  Z = 2,
  FLAT = 3,
};

/// @brief Per-point semantic label layer for loaded GLIM submaps.
class PointLabelLayer {
public:
  explicit PointLabelLayer(std::shared_ptr<spdlog::logger> logger);

  void clear();
  void set_submaps(const std::vector<glim::SubMap::Ptr>& submaps, const std::vector<glk::PointCloudBuffer::Ptr>& drawables);

  bool dirty() const;
  void mark_clean();

  int active_class_id() const;
  void set_active_class_id(int class_id);

  PointClassInfo class_or_default(int class_id) const;
  void set_class_metadata(int class_id, const std::string& name, const Eigen::Vector4f& color);
  const std::vector<PointClassInfo>& classes() const;

  PointDisplayMode display_mode() const;
  void set_display_mode(PointDisplayMode mode);
  Eigen::Vector4f flat_color() const;
  void set_flat_color(const Eigen::Vector4f& color);
  bool has_any_intensities() const;
  bool has_complete_intensities() const;

  std::size_t labeled_count() const;
  std::map<int, std::size_t> class_counts() const;
  const std::vector<std::vector<std::int32_t>>& labels() const;

  bool assign_selected(const std::vector<std::uint64_t>& point_ids);
  bool clear_selected(const std::vector<std::uint64_t>& point_ids);

  bool save(const std::string& path) const;
  bool load(const std::string& path);

private:
  PointClassInfo* find_class(int class_id);
  const PointClassInfo* find_class(int class_id) const;

  Eigen::Vector4f color_for_label(std::int32_t label) const;
  void apply_display_mode();
  void apply_intensity_colors();
  void apply_flat_color();
  void apply_z_color();
  Eigen::Vector2f world_z_range() const;
  void refresh_all_colors();
  void refresh_class_colors(int class_id);
  void update_submap_colors(int submap_id, const std::vector<unsigned int>& point_indices);

  static Eigen::Vector4f default_color(int class_id);
  static std::string default_name(int class_id);

private:
  std::shared_ptr<spdlog::logger> logger;

  std::vector<glim::SubMap::Ptr> submaps;
  std::vector<glk::PointCloudBuffer::Ptr> drawables;
  std::vector<std::vector<std::int32_t>> labels_by_submap;
  std::vector<bool> color_buffers_ready;
  std::vector<PointClassInfo> class_infos;

  int active_id;
  PointDisplayMode current_display_mode;
  Eigen::Vector4f current_flat_color;
  mutable bool z_range_valid;
  mutable Eigen::Vector2f cached_z_range;
  bool any_intensities;
  bool complete_intensities;
  bool has_loaded_map;
  bool is_dirty;
};

}  // namespace glim_manual_segmentation_editor
