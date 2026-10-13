#include <glim_manual_segmentation_editor/manual_segmentation_editor.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#include <glk/io/ply_io.hpp>
#include <guik/progress_modal.hpp>
#include <guik/recent_files.hpp>
#include <guik/spdlog_sink.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <portable-file-dialogs.h>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <pcl/PCLPointCloud2.h>
#include <pcl/PCLPointField.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <glim/mapping/sub_map.hpp>
#include <glim/util/logging.hpp>

namespace glim_manual_segmentation_editor {
namespace {

std::string to_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string lower_extension(const std::string& path) {
  return to_lower(std::filesystem::path(path).extension().string());
}

bool is_point_cloud_file(const std::string& path) {
  const std::string ext = lower_extension(path);
  return ext == ".ply" || ext == ".pcd";
}

const pcl::PCLPointField* find_field(const pcl::PCLPointCloud2& cloud, std::initializer_list<const char*> names) {
  for (const auto& field : cloud.fields) {
    const std::string field_name = to_lower(field.name);
    for (const auto* name : names) {
      if (field_name == name) {
        return &field;
      }
    }
  }

  return nullptr;
}

std::size_t field_scalar_size(const pcl::PCLPointField& field) {
  switch (field.datatype) {
    case pcl::PCLPointField::BOOL:
    case pcl::PCLPointField::INT8:
    case pcl::PCLPointField::UINT8:
      return 1;
    case pcl::PCLPointField::INT16:
    case pcl::PCLPointField::UINT16:
      return 2;
    case pcl::PCLPointField::INT32:
    case pcl::PCLPointField::UINT32:
    case pcl::PCLPointField::FLOAT32:
      return 4;
    case pcl::PCLPointField::INT64:
    case pcl::PCLPointField::UINT64:
    case pcl::PCLPointField::FLOAT64:
      return 8;
    default:
      return 0;
  }
}

bool is_readable_scalar_field(const pcl::PCLPointCloud2& cloud, const pcl::PCLPointField* field) {
  if (!field) {
    return false;
  }

  const std::size_t scalar_size = field_scalar_size(*field);
  return scalar_size != 0 && (field->count == 0 || field->count >= 1) && field->offset + scalar_size <= cloud.point_step;
}

std::size_t point_data_offset(const pcl::PCLPointCloud2& cloud, std::size_t point_index) {
  if (cloud.width == 0) {
    return point_index * cloud.point_step;
  }

  const std::size_t row = point_index / cloud.width;
  const std::size_t column = point_index % cloud.width;
  const std::size_t row_step = cloud.row_step ? cloud.row_step : static_cast<std::size_t>(cloud.width) * cloud.point_step;
  return row * row_step + column * cloud.point_step;
}

template <typename T>
T read_value(const pcl::PCLPointCloud2& cloud, const pcl::PCLPointField& field, std::size_t point_index) {
  T value{};
  const std::size_t offset = point_data_offset(cloud, point_index) + field.offset;
  if (offset + sizeof(T) > cloud.data.size()) {
    throw std::out_of_range("point field offset outside cloud data");
  }

  std::memcpy(&value, cloud.data.data() + offset, sizeof(T));
  return value;
}

double read_scalar(const pcl::PCLPointCloud2& cloud, const pcl::PCLPointField& field, std::size_t point_index) {
  switch (field.datatype) {
    case pcl::PCLPointField::BOOL:
      return read_value<bool>(cloud, field, point_index) ? 1.0 : 0.0;
    case pcl::PCLPointField::INT8:
      return read_value<std::int8_t>(cloud, field, point_index);
    case pcl::PCLPointField::UINT8:
      return read_value<std::uint8_t>(cloud, field, point_index);
    case pcl::PCLPointField::INT16:
      return read_value<std::int16_t>(cloud, field, point_index);
    case pcl::PCLPointField::UINT16:
      return read_value<std::uint16_t>(cloud, field, point_index);
    case pcl::PCLPointField::INT32:
      return read_value<std::int32_t>(cloud, field, point_index);
    case pcl::PCLPointField::UINT32:
      return read_value<std::uint32_t>(cloud, field, point_index);
    case pcl::PCLPointField::INT64:
      return static_cast<double>(read_value<std::int64_t>(cloud, field, point_index));
    case pcl::PCLPointField::UINT64:
      return static_cast<double>(read_value<std::uint64_t>(cloud, field, point_index));
    case pcl::PCLPointField::FLOAT32:
      return read_value<float>(cloud, field, point_index);
    case pcl::PCLPointField::FLOAT64:
      return read_value<double>(cloud, field, point_index);
    default:
      throw std::runtime_error("unsupported point field type");
  }
}

gtsam_points::PointCloudCPU::Ptr pcl_to_point_cloud_cpu(const pcl::PCLPointCloud2& cloud, const std::shared_ptr<spdlog::logger>& logger) {
  const auto* x_field = find_field(cloud, {"x"});
  const auto* y_field = find_field(cloud, {"y"});
  const auto* z_field = find_field(cloud, {"z"});
  if (!is_readable_scalar_field(cloud, x_field) || !is_readable_scalar_field(cloud, y_field) || !is_readable_scalar_field(cloud, z_field)) {
    logger->warn("Point cloud file is missing readable scalar x/y/z fields");
    return nullptr;
  }

  const auto* intensity_field = find_field(cloud, {"intensity", "intensities", "i", "reflectance", "reflectivity", "scalar_intensity"});
  const auto* normal_x_field = find_field(cloud, {"normal_x", "nx"});
  const auto* normal_y_field = find_field(cloud, {"normal_y", "ny"});
  const auto* normal_z_field = find_field(cloud, {"normal_z", "nz"});
  const bool has_intensity_field = is_readable_scalar_field(cloud, intensity_field);
  const bool has_normal_fields =
    is_readable_scalar_field(cloud, normal_x_field) && is_readable_scalar_field(cloud, normal_y_field) && is_readable_scalar_field(cloud, normal_z_field);

  std::size_t point_count = static_cast<std::size_t>(cloud.width) * cloud.height;
  if (point_count == 0) {
    if (cloud.point_step == 0) {
      logger->warn("Point cloud file has no point count and no point_step");
      return nullptr;
    }
    point_count = cloud.data.size() / cloud.point_step;
  }
  if (point_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    logger->warn("Point cloud contains too many points for the editor: {}", point_count);
    return nullptr;
  }

  std::vector<Eigen::Vector4d> points;
  std::vector<Eigen::Vector4d> normals;
  std::vector<double> intensities;
  points.reserve(point_count);
  if (has_normal_fields) {
    normals.reserve(point_count);
  }
  if (has_intensity_field) {
    intensities.reserve(point_count);
  }

  bool normals_are_valid = has_normal_fields;
  std::size_t skipped = 0;
  try {
    for (std::size_t i = 0; i < point_count; ++i) {
      const double x = read_scalar(cloud, *x_field, i);
      const double y = read_scalar(cloud, *y_field, i);
      const double z = read_scalar(cloud, *z_field, i);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        ++skipped;
        continue;
      }

      points.emplace_back(x, y, z, 1.0);

      if (has_intensity_field) {
        const double intensity = read_scalar(cloud, *intensity_field, i);
        intensities.push_back(std::isfinite(intensity) ? intensity : 0.0);
      }

      if (has_normal_fields) {
        Eigen::Vector3d normal(
          read_scalar(cloud, *normal_x_field, i),
          read_scalar(cloud, *normal_y_field, i),
          read_scalar(cloud, *normal_z_field, i));
        const bool normal_valid = normal.allFinite() && normal.squaredNorm() > 1.0e-12;
        normals_are_valid &= normal_valid;
        if (normal_valid) {
          normal.normalize();
          normals.emplace_back(normal.x(), normal.y(), normal.z(), 0.0);
        } else {
          normals.emplace_back(Eigen::Vector4d::Zero());
        }
      }
    }
  } catch (const std::exception& e) {
    logger->warn("Failed to decode point cloud fields: {}", e.what());
    return nullptr;
  }

  if (points.empty()) {
    logger->warn("Point cloud file contains no finite XYZ points");
    return nullptr;
  }

  auto frame = std::make_shared<gtsam_points::PointCloudCPU>();
  frame->add_points(points);
  if (has_intensity_field) {
    frame->add_intensities(intensities);
  }
  if (normals_are_valid) {
    frame->add_normals(normals);
  } else if (has_normal_fields) {
    logger->warn("Imported normals contain invalid values; normals will be estimated instead");
  }

  if (skipped) {
    logger->warn("Skipped {} non-finite points while importing point cloud", skipped);
  }

  return frame;
}

}  // namespace

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
  bool start_open_cloud = false;
  bool start_close_map = false;
  bool start_load_labels = false;
  bool start_save_labels = false;
  bool start_export_ply = false;

  const auto close_current_source = [this]() {
    if (submaps.empty()) {
      return true;
    }

    if (pfd::message("Warning", label_layer->dirty() ? "Close the current source? Unsaved labels will be lost." : "Close the current source?").result() !=
        pfd::button::ok) {
      return false;
    }

    submaps.clear();
    label_layer->clear();
    selector->clear();
    map_path.clear();
    return true;
  };

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Open GLIM Map Directory")) {
        start_open_map = close_current_source();
      }

      if (ImGui::MenuItem("Open Point Cloud File")) {
        start_open_cloud = close_current_source();
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

      if (ImGui::MenuItem("Close Source")) {
        start_close_map = true;
      }

      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }

  if (start_close_map) {
    if (submaps.empty()) {
      logger->warn("No source to close");
    } else if (close_current_source()) {
      logger->info("Closed current source");
    }
  }

  if (start_open_map || start_open_cloud || !init_map_path.empty()) {
    guik::RecentFiles recent_files("manual_segmentation_editor_open");
    if (init_map_path.empty()) {
      if (start_open_cloud) {
        const auto paths = pfd::open_file("Select a point cloud file", recent_files.most_recent(), {"Point clouds", "*.ply *.pcd", "PLY", "*.ply", "PCD", "*.pcd"}).result();
        map_path = paths.empty() ? "" : paths.front();
      } else {
        map_path = pfd::select_folder("Select a GLIM dump directory", recent_files.most_recent()).result();
      }
    } else {
      map_path = init_map_path;
      init_map_path.clear();
    }

    if (!map_path.empty()) {
      logger->info("Load source from {}", map_path);
      recent_files.push(map_path);
      const std::string source_path = map_path;
      progress_modal->open<std::vector<glim::SubMap::Ptr>>("open", [this, source_path](guik::ProgressInterface& progress) {
        return load_map_source(progress, source_path);
      });
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
  if (map_path.empty()) {
    return "semantic_labels.json";
  }

  std::error_code ec;
  if (std::filesystem::is_directory(map_path, ec)) {
    return map_path + "/semantic_labels.json";
  }

  if (is_point_cloud_file(map_path)) {
    return map_path + ".semantic_labels.json";
  }

  return map_path + "/semantic_labels.json";
}

std::string ManualSegmentationEditor::submap_path(const std::string& root, int index) const {
  std::ostringstream oss;
  oss << root << "/" << std::setw(6) << std::setfill('0') << index;
  return oss.str();
}

void ManualSegmentationEditor::ensure_normals(const gtsam_points::PointCloudCPU::Ptr& frame) {
  if (!frame || frame->has_normals()) {
    return;
  }

  if (frame->has_covs()) {
    frame->add_normals(gtsam_points::estimate_normals(frame->points, frame->covs, frame->size(), 4));
    return;
  }

  if (frame->size() < 3) {
    logger->warn("Not enough points to estimate normals");
    return;
  }

  const int k_neighbors = std::min<int>(20, std::max<int>(2, frame->size() - 1));
  frame->add_normals(gtsam_points::estimate_normals(frame->points, frame->size(), k_neighbors, 4));
}

std::vector<glim::SubMap::Ptr> ManualSegmentationEditor::load_map_source(guik::ProgressInterface& progress, const std::string& path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) {
    return load_glim_submaps(progress, path);
  }

  if (is_point_cloud_file(path)) {
    auto submap = load_point_cloud_submap(progress, path);
    return submap ? std::vector<glim::SubMap::Ptr>{submap} : std::vector<glim::SubMap::Ptr>{};
  }

  logger->warn("Unsupported input source: {}. Select a GLIM dump directory, .ply, or .pcd file.", path);
  return {};
}

std::vector<glim::SubMap::Ptr> ManualSegmentationEditor::load_glim_submaps(guik::ProgressInterface& progress, const std::string& map_path) {
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
        ensure_normals(frame);
      }
    }
  }

  logger->info("loaded {} submaps", num_submaps);
  return loaded;
}

glim::SubMap::Ptr ManualSegmentationEditor::load_point_cloud_submap(guik::ProgressInterface& progress, const std::string& path) {
  progress.set_title("Load point cloud");
  progress.set_text(path);
  progress.set_maximum(3);
  progress.increment();

  pcl::PCLPointCloud2 cloud;
  const std::string ext = lower_extension(path);
  int result = -1;
  if (ext == ".pcd") {
    result = pcl::io::loadPCDFile(path, cloud);
  } else if (ext == ".ply") {
    result = pcl::io::loadPLYFile(path, cloud);
  }

  if (result < 0) {
    logger->warn("Failed to load point cloud file {}", path);
    return nullptr;
  }

  progress.increment();
  auto frame = pcl_to_point_cloud_cpu(cloud, logger);
  if (!frame) {
    return nullptr;
  }

  ensure_normals(frame);
  progress.increment();

  auto submap = std::make_shared<glim::SubMap>();
  submap->id = 0;
  submap->session_id = 0;
  submap->T_world_origin.setIdentity();
  submap->T_origin_endpoint_L.setIdentity();
  submap->T_origin_endpoint_R.setIdentity();
  submap->frame = frame;

  logger->info("loaded point cloud {} with {} points", path, frame->size());
  return submap;
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
