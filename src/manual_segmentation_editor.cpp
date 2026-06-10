#include <glim_manual_segmentation_editor/manual_segmentation_editor.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <glk/io/ply_io.hpp>
#include <guik/progress_modal.hpp>
#include <guik/recent_files.hpp>
#include <guik/spdlog_sink.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <portable-file-dialogs.h>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <glim/mapping/sub_map.hpp>
#include <glim/util/logging.hpp>

namespace glim_manual_segmentation_editor {

ManualSegmentationEditor::ManualSegmentationEditor(const std::string& init_map_path)
: init_map_path(init_map_path),
  ui_class_id(1),
  ui_display_mode(static_cast<int>(PointDisplayMode::SEGMENTATION)) {
  logger = glim::get_default_logger();

  auto viewer = guik::viewer();
  viewer->register_ui_callback("log", guik::create_logger_ui(glim::get_ringbuffer_sink(), 0.8));
  viewer->register_ui_callback("ui_callback", [this] { ui_callback(); });

  progress_modal.reset(new guik::ProgressModal("manual_segmentation_progress"));
  selector.reset(new PointsSelector(logger));
  label_layer.reset(new PointLabelLayer(logger));

  ui_class_name.fill('\0');
  ui_class_color = {0.0f, 0.0f, 0.0f, 1.0f};
  ui_flat_color = {0.70f, 0.70f, 0.70f, 1.0f};
  sync_class_inputs(1);
}

ManualSegmentationEditor::~ManualSegmentationEditor() {}

void ManualSegmentationEditor::run() {
  guik::viewer()->spin();
}

void ManualSegmentationEditor::main_menu() {
  bool start_open_map = false;
  bool start_close_map = false;
  bool start_load_labels = false;
  bool start_save_labels = false;
  bool start_export_ply = false;

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Open New Map")) {
        if (!submaps.empty()) {
          if (pfd::message("Warning", label_layer->dirty() ? "Close the current map? Unsaved labels will be lost." : "Close the current map?").result() ==
              pfd::button::ok) {
            submaps.clear();
            label_layer->clear();
            selector->clear();
            map_path.clear();
            start_open_map = true;
          }
        } else {
          start_open_map = true;
        }
      }

      if (ImGui::MenuItem("Load Labels")) {
        start_load_labels = true;
      }

      if (ImGui::MenuItem("Save Labels")) {
        start_save_labels = true;
      }

      if (ImGui::MenuItem("Export Semantic PLY")) {
        start_export_ply = true;
      }

      if (ImGui::MenuItem("Close Map")) {
        start_close_map = true;
      }

      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }

  if (start_close_map) {
    if (submaps.empty()) {
      logger->warn("No map to close");
    } else if (pfd::message("Warning", label_layer->dirty() ? "Close the map? Unsaved labels will be lost." : "Close the map?").result() == pfd::button::ok) {
      submaps.clear();
      label_layer->clear();
      selector->clear();
      map_path.clear();
    }
  }

  if (start_open_map || !init_map_path.empty()) {
    guik::RecentFiles recent_files("manual_segmentation_editor_open");
    if (init_map_path.empty()) {
      map_path = pfd::select_folder("Select a dump directory", recent_files.most_recent()).result();
    } else {
      map_path = init_map_path;
      init_map_path.clear();
    }

    if (!map_path.empty()) {
      logger->info("Load map from {}", map_path);
      recent_files.push(map_path);
      progress_modal->open<std::vector<glim::SubMap::Ptr>>("open", [this](guik::ProgressInterface& progress) { return load_submaps(progress, map_path); });
    }
  }

  const auto loaded_submaps = progress_modal->run<std::vector<glim::SubMap::Ptr>>("open");
  if (loaded_submaps && !loaded_submaps->empty()) {
    submaps = *loaded_submaps;
    selector->set_submaps(submaps);
    label_layer->set_submaps(submaps, selector->submap_drawables_ref());
    ui_display_mode = static_cast<int>(label_layer->display_mode());
    const Eigen::Vector4f flat_color = label_layer->flat_color();
    ui_flat_color = {flat_color.x(), flat_color.y(), flat_color.z(), flat_color.w()};
    sync_class_inputs(label_layer->active_class_id());
  }

  if (start_load_labels) {
    if (submaps.empty()) {
      logger->warn("No map loaded");
    } else if (label_layer->load(default_label_path())) {
      sync_class_inputs(label_layer->active_class_id());
    }
  }

  if (start_save_labels) {
    if (submaps.empty()) {
      logger->warn("No map loaded");
    } else {
      const std::string path = default_label_path();
      progress_modal->open<bool>("save_labels", [this, path](guik::ProgressInterface& progress) { return save_labels(progress, path); });
    }
  }

  const auto save_labels_result = progress_modal->run<bool>("save_labels");
  if (save_labels_result && *save_labels_result) {
    label_layer->mark_clean();
  }

  if (start_export_ply) {
    if (submaps.empty()) {
      logger->warn("No map loaded");
    } else {
      guik::RecentFiles recent_files("manual_segmentation_editor_export");
      const std::string path = pfd::save_file("Select semantic PLY destination", recent_files.most_recent(), {"PLY", "*.ply"}).result();
      if (!path.empty()) {
        recent_files.push(path);
        progress_modal->open<bool>("export_ply", [this, path](guik::ProgressInterface& progress) { return export_semantic_ply(progress, path); });
      }
    }
  }

  const auto export_result = progress_modal->run<bool>("export_ply");
  if (export_result) {
    if (*export_result) {
      logger->info("Semantic PLY exported");
    } else {
      logger->warn("Failed to export semantic PLY");
    }
  }
}

void ManualSegmentationEditor::ui_callback() {
  main_menu();
  selector->draw_ui();
  draw_label_ui();
}

void ManualSegmentationEditor::draw_label_ui() {
  ImGui::SetNextWindowPos(ImVec2(430.0f, 30.0f), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Semantic Labels", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::End();
    return;
  }

  const auto& selected = selector->selected_point_ids_ref();
  ImGui::Text("Submaps: %ld", submaps.size());
  ImGui::Text("Selected points: %ld", selected.size());
  ImGui::Text("Labeled points: %ld", label_layer->labeled_count());
  ImGui::Text("Label file: %s", default_label_path().c_str());
  if (label_layer->dirty()) {
    ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.25f, 1.0f), "Unsaved label changes");
  }

  ImGui::Separator();
  const char* display_modes[] = {"Segmentation", "Intensity", "Z", "Flat color"};
  int edited_display_mode = ui_display_mode;
  if (ImGui::Combo("Visualization", &edited_display_mode, display_modes, IM_ARRAYSIZE(display_modes))) {
    ui_display_mode = std::max(0, std::min(edited_display_mode, static_cast<int>(IM_ARRAYSIZE(display_modes)) - 1));
    label_layer->set_display_mode(static_cast<PointDisplayMode>(ui_display_mode));
  }

  if (static_cast<PointDisplayMode>(ui_display_mode) == PointDisplayMode::FLAT) {
    if (ImGui::ColorEdit4("Flat color", ui_flat_color.data())) {
      label_layer->set_flat_color(Eigen::Vector4f(ui_flat_color[0], ui_flat_color[1], ui_flat_color[2], ui_flat_color[3]));
    }
  } else if (static_cast<PointDisplayMode>(ui_display_mode) == PointDisplayMode::INTENSITY) {
    if (!label_layer->has_any_intensities()) {
      ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.25f, 1.0f), "No intensities in loaded map");
    } else if (!label_layer->has_complete_intensities()) {
      ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.25f, 1.0f), "Some submaps have no intensities");
    }
  }

  ImGui::Separator();
  int edited_class_id = ui_class_id;
  if (ImGui::InputInt("Class ID", &edited_class_id)) {
    edited_class_id = std::max(0, edited_class_id);
    sync_class_inputs(edited_class_id);
  }

  bool metadata_changed = false;
  if (ui_class_id == 0) {
    ImGui::Text("Class name: unlabeled");
  } else {
    metadata_changed |= ImGui::InputText("Class name", ui_class_name.data(), ui_class_name.size());
    metadata_changed |= ImGui::ColorEdit4("Class color", ui_class_color.data());
  }

  if (metadata_changed) {
    label_layer->set_class_metadata(
      ui_class_id,
      ui_class_name.data(),
      Eigen::Vector4f(ui_class_color[0], ui_class_color[1], ui_class_color[2], ui_class_color[3]));
  }

  if (ImGui::Button("Assign Selected")) {
    label_layer->set_active_class_id(ui_class_id);
    label_layer->set_class_metadata(
      ui_class_id,
      ui_class_name.data(),
      Eigen::Vector4f(ui_class_color[0], ui_class_color[1], ui_class_color[2], ui_class_color[3]));
    label_layer->assign_selected(selected);
  }

  ImGui::SameLine();
  if (ImGui::Button("Clear Selected Labels")) {
    label_layer->clear_selected(selected);
  }

  if (ImGui::Button("Clear Selection")) {
    selector->clear_selected_points();
  }

  ImGui::Separator();
  ImGui::Text("Classes");
  const auto class_counts = label_layer->class_counts();
  for (const auto& [class_id, point_count] : class_counts) {
    const auto cls = label_layer->class_or_default(class_id);
    const ImVec4 color(cls.color.x(), cls.color.y(), cls.color.z(), cls.color.w());
    const std::string color_id = "##class_color_" + std::to_string(cls.id);
    ImGui::ColorButton(color_id.c_str(), color);
    ImGui::SameLine();
    ImGui::Text("%d  %s  (%zu %s)", cls.id, cls.name.c_str(), point_count, point_count == 1 ? "point" : "points");
  }

  ImGui::End();
}

void ManualSegmentationEditor::sync_class_inputs(int class_id) {
  ui_class_id = std::max(0, class_id);
  label_layer->set_active_class_id(ui_class_id);

  const auto cls = label_layer->class_or_default(ui_class_id);
  std::snprintf(ui_class_name.data(), ui_class_name.size(), "%s", cls.name.c_str());
  ui_class_color = {cls.color.x(), cls.color.y(), cls.color.z(), cls.color.w()};
}

std::string ManualSegmentationEditor::default_label_path() const {
  return map_path.empty() ? "semantic_labels.json" : map_path + "/semantic_labels.json";
}

std::string ManualSegmentationEditor::submap_path(const std::string& root, int index) const {
  std::ostringstream oss;
  oss << root << "/" << std::setw(6) << std::setfill('0') << index;
  return oss.str();
}

std::vector<glim::SubMap::Ptr> ManualSegmentationEditor::load_submaps(guik::ProgressInterface& progress, const std::string& map_path) {
  progress.set_title("Load submaps");
  progress.set_text("Now loading");

  std::ifstream ifs(map_path + "/graph.txt");
  if (!ifs) {
    logger->warn("Failed to open {}/graph.txt", map_path);
    return {};
  }

  std::string token;
  int num_submaps;
  ifs >> token >> num_submaps;

  if (token != "num_submaps:") {
    logger->warn("Invalid graph.txt format");
    return {};
  }

  progress.set_maximum(num_submaps);

  std::vector<glim::SubMap::Ptr> loaded(num_submaps);
  for (int i = 0; i < num_submaps; i++) {
    progress.increment();
    loaded[i] = glim::SubMap::load(submap_path(map_path, i));
    if (!loaded[i]) {
      logger->warn("Failed to load submap {}", i);
      return {};
    }

    if (!loaded[i]->frame->has_normals()) {
      auto frame = std::dynamic_pointer_cast<gtsam_points::PointCloudCPU>(loaded[i]->frame);
      if (!frame) {
        logger->warn("Failed to cast frame to PointCloudCPU");
      } else {
        frame->add_normals(gtsam_points::estimate_normals(frame->points, frame->covs, frame->size(), 4));
      }
    }
  }

  logger->info("loaded {} submaps", num_submaps);
  return loaded;
}

bool ManualSegmentationEditor::save_labels(guik::ProgressInterface& progress, const std::string& path) {
  progress.set_title("Save semantic labels");
  progress.set_text(path);
  progress.set_maximum(1);
  progress.increment();
  return label_layer->save(path);
}

bool ManualSegmentationEditor::export_semantic_ply(guik::ProgressInterface& progress, const std::string& path) {
  progress.set_title("Export semantic PLY");
  progress.set_text("Collecting points");
  progress.set_maximum(submaps.size() + 1);

  if (submaps.empty()) {
    logger->warn("No map loaded");
    return false;
  }

  const auto& labels = label_layer->labels();
  std::size_t total_points = 0;
  bool has_intensities = true;
  for (const auto& submap : submaps) {
    if (!submap || !submap->frame) {
      continue;
    }
    total_points += submap->frame->size();
    has_intensities &= submap->frame->has_intensities();
  }

  glk::PLYData ply;
  ply.vertices.reserve(total_points);
  if (has_intensities) {
    ply.intensities.reserve(total_points);
  }

  std::vector<std::int32_t> semantic_classes;
  semantic_classes.reserve(total_points);

  for (const auto& submap : submaps) {
    progress.increment();
    if (!submap || !submap->frame || submap->id < 0 || submap->id >= static_cast<int>(labels.size())) {
      logger->warn("Invalid submap while exporting semantic PLY");
      return false;
    }
    if (labels[submap->id].size() != submap->frame->size()) {
      logger->warn("Label count mismatch for submap {}", submap->id);
      return false;
    }

    for (int i = 0; i < submap->frame->size(); ++i) {
      const Eigen::Vector4d point = submap->T_world_origin * submap->frame->points[i];
      ply.vertices.push_back(point.head<3>().cast<float>());
      if (has_intensities) {
        ply.intensities.push_back(submap->frame->intensities[i]);
      }
      semantic_classes.push_back(labels[submap->id][i]);
    }
  }

  progress.set_text("Writing PLY");
  ply.add_prop<std::int32_t>("class", semantic_classes.data(), semantic_classes.size());
  return glk::save_ply_binary(path, ply);
}

}  // namespace glim_manual_segmentation_editor
