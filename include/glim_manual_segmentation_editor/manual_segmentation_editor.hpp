#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <glim/mapping/sub_map.hpp>
#include <glim_manual_segmentation_editor/point_label_layer.hpp>
#include <glim_manual_segmentation_editor/points_selector.hpp>
#include <guik/progress_modal.hpp>
#include <spdlog/spdlog.h>

namespace glim_manual_segmentation_editor {

/// @brief Standalone GLIM map editor for manually assigning semantic point labels.
class ManualSegmentationEditor {
public:
  explicit ManualSegmentationEditor(const std::string& init_map_path);
  ~ManualSegmentationEditor();

  void run();

private:
  void main_menu();
  void ui_callback();
  void draw_label_ui();

  void sync_class_inputs(int class_id);
  std::string default_label_path() const;
  std::string submap_path(const std::string& root, int index) const;

  std::vector<glim::SubMap::Ptr> load_submaps(guik::ProgressInterface& progress, const std::string& map_path);
  bool save_labels(guik::ProgressInterface& progress, const std::string& path);
  bool export_semantic_ply(guik::ProgressInterface& progress, const std::string& path);

private:
  std::shared_ptr<spdlog::logger> logger;
  std::unique_ptr<guik::ProgressModal> progress_modal;

  std::string map_path;
  std::string init_map_path;

  std::vector<glim::SubMap::Ptr> submaps;
  std::unique_ptr<PointsSelector> selector;
  std::unique_ptr<PointLabelLayer> label_layer;

  int ui_class_id;
  std::array<char, 64> ui_class_name;
  std::array<float, 4> ui_class_color;
  int ui_display_mode;
  std::array<float, 4> ui_flat_color;
};

}  // namespace glim_manual_segmentation_editor
