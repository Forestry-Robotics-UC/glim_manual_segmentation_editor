#include <glim_manual_segmentation_editor/point_label_layer.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>

#include <glk/colormap.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <nlohmann/json.hpp>

namespace glim_manual_segmentation_editor {
namespace {
using ColorVector = std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>;
}

PointLabelLayer::PointLabelLayer(std::shared_ptr<spdlog::logger> logger)
: logger(logger),
  active_id(1),
  current_display_mode(PointDisplayMode::SEGMENTATION),
  current_flat_color(0.70f, 0.70f, 0.70f, 1.0f),
  z_range_valid(false),
  cached_z_range(0.0f, 1.0f),
  any_intensities(false),
  complete_intensities(false),
  has_loaded_map(false),
  is_dirty(false) {
  class_infos.push_back({0, default_name(0), default_color(0)});
  class_infos.push_back({1, default_name(1), default_color(1)});
}

void PointLabelLayer::clear() {
  submaps.clear();
  drawables.clear();
  labels_by_submap.clear();
  color_buffers_ready.clear();
  class_infos.clear();
  class_infos.push_back({0, default_name(0), default_color(0)});
  class_infos.push_back({1, default_name(1), default_color(1)});
  active_id = 1;
  z_range_valid = false;
  cached_z_range = Eigen::Vector2f(0.0f, 1.0f);
  any_intensities = false;
  complete_intensities = false;
  has_loaded_map = false;
  is_dirty = false;
}

void PointLabelLayer::set_submaps(const std::vector<glim::SubMap::Ptr>& submaps, const std::vector<glk::PointCloudBuffer::Ptr>& submap_drawables) {
  this->submaps = submaps;
  this->drawables = submap_drawables;
  labels_by_submap.assign(submaps.size(), {});
  color_buffers_ready.assign(submaps.size(), false);
  z_range_valid = false;
  cached_z_range = Eigen::Vector2f(0.0f, 1.0f);
  any_intensities = false;
  complete_intensities = !submaps.empty();

  for (const auto& submap : submaps) {
    if (!submap || submap->id < 0 || submap->id >= static_cast<int>(labels_by_submap.size())) {
      logger->warn("Invalid submap id in label layer");
      complete_intensities = false;
      continue;
    }

    labels_by_submap[submap->id].assign(submap->frame->size(), 0);
    const bool has_intensity = submap->frame && submap->frame->has_intensities();
    any_intensities |= has_intensity;
    complete_intensities &= has_intensity;
  }

  has_loaded_map = !submaps.empty();
  is_dirty = false;
  apply_display_mode();
}

bool PointLabelLayer::dirty() const {
  return is_dirty;
}

void PointLabelLayer::mark_clean() {
  is_dirty = false;
}

int PointLabelLayer::active_class_id() const {
  return active_id;
}

void PointLabelLayer::set_active_class_id(int class_id) {
  active_id = std::max(0, class_id);
  if (!find_class(active_id)) {
    class_infos.push_back(class_or_default(active_id));
  }
}

PointClassInfo PointLabelLayer::class_or_default(int class_id) const {
  class_id = std::max(0, class_id);
  if (const auto* found = find_class(class_id)) {
    return *found;
  }

  return {class_id, default_name(class_id), default_color(class_id)};
}

void PointLabelLayer::set_class_metadata(int class_id, const std::string& name, const Eigen::Vector4f& color) {
  class_id = std::max(0, class_id);
  const PointClassInfo info = class_id == 0 ? PointClassInfo{0, default_name(0), default_color(0)} : PointClassInfo{class_id, name, color};

  auto* found = find_class(class_id);
  if (!found) {
    class_infos.push_back(info);
    refresh_class_colors(class_id);
    is_dirty = true;
    return;
  }

  const bool color_changed = !found->color.isApprox(info.color);
  const bool name_changed = found->name != info.name;
  *found = info;

  if (color_changed) {
    refresh_class_colors(class_id);
  }
  if (color_changed || name_changed) {
    is_dirty = true;
  }
}

const std::vector<PointClassInfo>& PointLabelLayer::classes() const {
  return class_infos;
}

PointDisplayMode PointLabelLayer::display_mode() const {
  return current_display_mode;
}

void PointLabelLayer::set_display_mode(PointDisplayMode mode) {
  current_display_mode = mode;
  apply_display_mode();
}

Eigen::Vector4f PointLabelLayer::flat_color() const {
  return current_flat_color;
}

void PointLabelLayer::set_flat_color(const Eigen::Vector4f& color) {
  current_flat_color = color;
  if (current_display_mode == PointDisplayMode::FLAT) {
    apply_flat_color();
  }
}

bool PointLabelLayer::has_any_intensities() const {
  return any_intensities;
}

bool PointLabelLayer::has_complete_intensities() const {
  return complete_intensities;
}

std::size_t PointLabelLayer::labeled_count() const {
  std::size_t count = 0;
  for (const auto& labels : labels_by_submap) {
    count += std::count_if(labels.begin(), labels.end(), [](std::int32_t label) { return label != 0; });
  }
  return count;
}

std::map<int, std::size_t> PointLabelLayer::class_counts() const {
  std::map<int, std::size_t> counts;
  for (const auto& cls : class_infos) {
    counts.emplace(cls.id, 0);
  }

  for (const auto& labels : labels_by_submap) {
    for (const auto label : labels) {
      const int class_id = label < 0 ? 0 : static_cast<int>(label);
      ++counts[class_id];
    }
  }

  return counts;
}

const std::vector<std::vector<std::int32_t>>& PointLabelLayer::labels() const {
  return labels_by_submap;
}

bool PointLabelLayer::assign_selected(const std::vector<std::uint64_t>& point_ids) {
  if (!has_loaded_map) {
    logger->warn("No map loaded");
    return false;
  }
  if (point_ids.empty()) {
    logger->warn("No points selected");
    return false;
  }

  set_active_class_id(active_id);
  std::vector<std::vector<unsigned int>> updated(labels_by_submap.size());

  for (const auto packed_id : point_ids) {
    const std::uint64_t submap_id = packed_id >> 32;
    const std::uint64_t point_id = packed_id & ((1ull << 32) - 1);
    if (submap_id >= labels_by_submap.size() || point_id >= labels_by_submap[submap_id].size()) {
      logger->warn("Invalid selected point id {}", packed_id);
      continue;
    }

    labels_by_submap[submap_id][point_id] = static_cast<std::int32_t>(active_id);
    updated[submap_id].push_back(static_cast<unsigned int>(point_id));
  }

  for (int submap_id = 0; submap_id < static_cast<int>(updated.size()); ++submap_id) {
    auto& indices = updated[submap_id];
    if (indices.empty()) {
      continue;
    }

    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    update_submap_colors(submap_id, indices);
  }

  is_dirty = true;
  logger->info("Assigned {} selected points to class {}", point_ids.size(), active_id);
  return true;
}

bool PointLabelLayer::clear_selected(const std::vector<std::uint64_t>& point_ids) {
  if (!has_loaded_map) {
    logger->warn("No map loaded");
    return false;
  }
  if (point_ids.empty()) {
    logger->warn("No points selected");
    return false;
  }

  std::vector<std::vector<unsigned int>> updated(labels_by_submap.size());
  for (const auto packed_id : point_ids) {
    const std::uint64_t submap_id = packed_id >> 32;
    const std::uint64_t point_id = packed_id & ((1ull << 32) - 1);
    if (submap_id >= labels_by_submap.size() || point_id >= labels_by_submap[submap_id].size()) {
      logger->warn("Invalid selected point id {}", packed_id);
      continue;
    }

    labels_by_submap[submap_id][point_id] = 0;
    updated[submap_id].push_back(static_cast<unsigned int>(point_id));
  }

  for (int submap_id = 0; submap_id < static_cast<int>(updated.size()); ++submap_id) {
    auto& indices = updated[submap_id];
    if (indices.empty()) {
      continue;
    }

    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    update_submap_colors(submap_id, indices);
  }

  is_dirty = true;
  logger->info("Cleared labels for {} selected points", point_ids.size());
  return true;
}

bool PointLabelLayer::save(const std::string& path) const {
  if (!has_loaded_map) {
    logger->warn("No map loaded");
    return false;
  }

  nlohmann::json root;
  root["version"] = 1;
  root["classes"] = nlohmann::json::array();

  auto classes = class_infos;
  std::sort(classes.begin(), classes.end(), [](const auto& lhs, const auto& rhs) { return lhs.id < rhs.id; });
  for (const auto& cls : classes) {
    root["classes"].push_back({
      {"id", cls.id},
      {"name", cls.name},
      {"color", {cls.color.x(), cls.color.y(), cls.color.z(), cls.color.w()}},
    });
  }

  root["submaps"] = nlohmann::json::array();
  for (const auto& submap : submaps) {
    if (!submap || submap->id < 0 || submap->id >= static_cast<int>(labels_by_submap.size())) {
      logger->warn("Invalid submap while saving labels");
      return false;
    }

    const auto& labels = labels_by_submap[submap->id];
    nlohmann::json encoded = nlohmann::json::array();
    if (!labels.empty()) {
      std::int32_t current = labels.front();
      std::size_t count = 1;
      for (std::size_t i = 1; i < labels.size(); ++i) {
        if (labels[i] == current) {
          ++count;
          continue;
        }

        encoded.push_back({current, count});
        current = labels[i];
        count = 1;
      }
      encoded.push_back({current, count});
    }

    root["submaps"].push_back({
      {"id", submap->id},
      {"size", labels.size()},
      {"labels_rle", encoded},
    });
  }

  std::ofstream ofs(path);
  if (!ofs) {
    logger->warn("Failed to open {} for label saving", path);
    return false;
  }

  ofs << root.dump(2) << std::endl;
  logger->info("Saved semantic labels to {}", path);
  return true;
}

bool PointLabelLayer::load(const std::string& path) {
  if (!has_loaded_map) {
    logger->warn("No map loaded");
    return false;
  }

  std::ifstream ifs(path);
  if (!ifs) {
    logger->warn("Failed to open label file {}", path);
    return false;
  }

  nlohmann::json root;
  try {
    ifs >> root;
  } catch (const std::exception& e) {
    logger->warn("Failed to parse {}: {}", path, e.what());
    return false;
  }

  if (root.value("version", 0) != 1) {
    logger->warn("Unsupported semantic label version in {}", path);
    return false;
  }
  if (!root.contains("submaps") || !root["submaps"].is_array() || root["submaps"].size() != submaps.size()) {
    logger->warn("Label file submap count does not match loaded map");
    return false;
  }

  std::vector<PointClassInfo> loaded_classes;
  loaded_classes.push_back({0, default_name(0), default_color(0)});
  if (root.contains("classes") && root["classes"].is_array()) {
    for (const auto& cls_json : root["classes"]) {
      const int id = cls_json.value("id", 0);
      if (id <= 0) {
        continue;
      }

      Eigen::Vector4f color = default_color(id);
      if (cls_json.contains("color") && cls_json["color"].is_array() && cls_json["color"].size() == 4) {
        color = Eigen::Vector4f(
          cls_json["color"][0].get<float>(),
          cls_json["color"][1].get<float>(),
          cls_json["color"][2].get<float>(),
          cls_json["color"][3].get<float>());
      }

      loaded_classes.push_back({id, cls_json.value("name", default_name(id)), color});
    }
  }

  std::vector<std::vector<std::int32_t>> loaded_labels = labels_by_submap;
  for (auto& labels : loaded_labels) {
    std::fill(labels.begin(), labels.end(), 0);
  }

  std::vector<bool> seen(labels_by_submap.size(), false);
  for (const auto& submap_json : root["submaps"]) {
    const int id = submap_json.value("id", -1);
    if (id < 0 || id >= static_cast<int>(labels_by_submap.size()) || !submaps[id] || submaps[id]->id != id) {
      logger->warn("Label file contains invalid submap id {}", id);
      return false;
    }

    const std::size_t expected_size = labels_by_submap[id].size();
    if (submap_json.value("size", std::size_t{0}) != expected_size) {
      logger->warn("Label size mismatch for submap {}", id);
      return false;
    }

    if (!submap_json.contains("labels_rle") || !submap_json["labels_rle"].is_array()) {
      logger->warn("Missing RLE labels for submap {}", id);
      return false;
    }

    std::vector<std::int32_t> decoded;
    decoded.reserve(expected_size);
    for (const auto& run : submap_json["labels_rle"]) {
      if (!run.is_array() || run.size() != 2) {
        logger->warn("Invalid RLE entry for submap {}", id);
        return false;
      }

      const std::int32_t label = run[0].get<std::int32_t>();
      const std::size_t count = run[1].get<std::size_t>();
      if (decoded.size() + count > expected_size) {
        logger->warn("RLE overrun for submap {}", id);
        return false;
      }
      decoded.insert(decoded.end(), count, label);
    }

    if (decoded.size() != expected_size) {
      logger->warn("RLE size mismatch for submap {}", id);
      return false;
    }

    loaded_labels[id] = std::move(decoded);
    seen[id] = true;
  }

  for (const auto& submap : submaps) {
    if (!submap || submap->id < 0 || submap->id >= static_cast<int>(seen.size()) || !seen[submap->id]) {
      logger->warn("Label file is missing labels for a loaded submap");
      return false;
    }
  }

  class_infos = std::move(loaded_classes);
  labels_by_submap = std::move(loaded_labels);
  apply_display_mode();
  is_dirty = false;
  logger->info("Loaded semantic labels from {}", path);
  return true;
}

PointClassInfo* PointLabelLayer::find_class(int class_id) {
  const auto found = std::find_if(class_infos.begin(), class_infos.end(), [class_id](const auto& cls) { return cls.id == class_id; });
  return found == class_infos.end() ? nullptr : &(*found);
}

const PointClassInfo* PointLabelLayer::find_class(int class_id) const {
  const auto found = std::find_if(class_infos.begin(), class_infos.end(), [class_id](const auto& cls) { return cls.id == class_id; });
  return found == class_infos.end() ? nullptr : &(*found);
}

Eigen::Vector4f PointLabelLayer::color_for_label(std::int32_t label) const {
  if (const auto* found = find_class(label)) {
    return found->color;
  }
  return default_color(label);
}

void PointLabelLayer::apply_display_mode() {
  if (!has_loaded_map) {
    return;
  }

  switch (current_display_mode) {
    case PointDisplayMode::SEGMENTATION:
      refresh_all_colors();
      break;
    case PointDisplayMode::INTENSITY:
      apply_intensity_colors();
      break;
    case PointDisplayMode::Z:
      apply_z_color();
      break;
    case PointDisplayMode::FLAT:
      apply_flat_color();
      break;
  }
}

void PointLabelLayer::apply_intensity_colors() {
  if (!has_loaded_map) {
    return;
  }

  if (!any_intensities) {
    logger->warn("Loaded map has no point intensities; using flat gray fallback");
  } else if (!complete_intensities) {
    logger->warn("Some submaps have no point intensities; using flat gray fallback for those submaps");
  }

  const Eigen::Vector4f fallback(0.55f, 0.55f, 0.55f, 1.0f);
  for (const auto& submap : submaps) {
    if (!submap || !submap->frame || submap->id < 0 || submap->id >= static_cast<int>(drawables.size()) || !drawables[submap->id]) {
      continue;
    }

    if (submap->frame->size() == 0) {
      guik::viewer()->update_drawable("submap_" + std::to_string(submap->id), drawables[submap->id], guik::VertexColor(submap->T_world_origin));
      continue;
    }

    if (submap->frame->has_intensities()) {
      double max_intensity = 0.0;
      for (int i = 0; i < submap->frame->size(); ++i) {
        const double intensity = submap->frame->intensities[i];
        if (std::isfinite(intensity)) {
          max_intensity = std::max(max_intensity, intensity);
        }
      }

      drawables[submap->id]->add_intensity(glk::COLORMAP::TURBO, submap->frame->intensities, submap->frame->size(), max_intensity > 0.0 ? 1.0 / max_intensity : 1.0);
    } else {
      ColorVector colors(submap->frame->size(), fallback);
      if (!colors.empty()) {
        drawables[submap->id]->add_color(colors);
      }
    }

    if (submap->id < static_cast<int>(color_buffers_ready.size())) {
      color_buffers_ready[submap->id] = true;
    }
    guik::viewer()->update_drawable("submap_" + std::to_string(submap->id), drawables[submap->id], guik::VertexColor(submap->T_world_origin));
  }
}

void PointLabelLayer::apply_flat_color() {
  if (!has_loaded_map) {
    return;
  }

  for (const auto& submap : submaps) {
    if (!submap || submap->id < 0 || submap->id >= static_cast<int>(drawables.size()) || !drawables[submap->id]) {
      continue;
    }

    guik::viewer()->update_drawable("submap_" + std::to_string(submap->id), drawables[submap->id], guik::FlatColor(current_flat_color, submap->T_world_origin));
  }
}

void PointLabelLayer::apply_z_color() {
  if (!has_loaded_map) {
    return;
  }

  guik::viewer()->shader_setting().add<Eigen::Vector2f>("z_range", world_z_range());
  for (const auto& submap : submaps) {
    if (!submap || submap->id < 0 || submap->id >= static_cast<int>(drawables.size()) || !drawables[submap->id]) {
      continue;
    }

    guik::viewer()->update_drawable("submap_" + std::to_string(submap->id), drawables[submap->id], guik::Rainbow(submap->T_world_origin));
  }
}

Eigen::Vector2f PointLabelLayer::world_z_range() const {
  if (z_range_valid) {
    return cached_z_range;
  }

  double min_z = std::numeric_limits<double>::max();
  double max_z = std::numeric_limits<double>::lowest();
  for (const auto& submap : submaps) {
    if (!submap || !submap->frame) {
      continue;
    }

    for (int i = 0; i < submap->frame->size(); ++i) {
      const double z = (submap->T_world_origin * submap->frame->points[i]).z();
      if (!std::isfinite(z)) {
        continue;
      }
      min_z = std::min(min_z, z);
      max_z = std::max(max_z, z);
    }
  }

  if (min_z == std::numeric_limits<double>::max() || max_z == std::numeric_limits<double>::lowest()) {
    cached_z_range = Eigen::Vector2f(0.0f, 1.0f);
  } else if (std::abs(max_z - min_z) < 1e-3) {
    cached_z_range = Eigen::Vector2f(static_cast<float>(min_z - 0.5), static_cast<float>(max_z + 0.5));
  } else {
    cached_z_range = Eigen::Vector2f(static_cast<float>(min_z), static_cast<float>(max_z));
  }

  z_range_valid = true;
  return cached_z_range;
}

void PointLabelLayer::refresh_all_colors() {
  for (const auto& submap : submaps) {
    if (!submap || submap->id < 0 || submap->id >= static_cast<int>(labels_by_submap.size())) {
      continue;
    }

    const auto& labels = labels_by_submap[submap->id];
    ColorVector colors(labels.size());
    std::transform(labels.begin(), labels.end(), colors.begin(), [this](std::int32_t label) { return color_for_label(label); });

    if (submap->id < static_cast<int>(drawables.size()) && drawables[submap->id]) {
      if (colors.empty()) {
        guik::viewer()->update_drawable("submap_" + std::to_string(submap->id), drawables[submap->id], guik::VertexColor(submap->T_world_origin));
      } else if (submap->id >= static_cast<int>(color_buffers_ready.size()) || !color_buffers_ready[submap->id]) {
        drawables[submap->id]->add_color(colors);
        if (submap->id < static_cast<int>(color_buffers_ready.size())) {
          color_buffers_ready[submap->id] = true;
        }
      } else {
        std::vector<unsigned int> indices(colors.size());
        std::iota(indices.begin(), indices.end(), 0);
        drawables[submap->id]->update_color_with_indices(colors, indices);
      }
      guik::viewer()->update_drawable("submap_" + std::to_string(submap->id), drawables[submap->id], guik::VertexColor(submap->T_world_origin));
    }
  }
}

void PointLabelLayer::refresh_class_colors(int class_id) {
  if (current_display_mode != PointDisplayMode::SEGMENTATION) {
    return;
  }

  for (int submap_id = 0; submap_id < static_cast<int>(labels_by_submap.size()); ++submap_id) {
    std::vector<unsigned int> indices;
    const auto& labels = labels_by_submap[submap_id];
    for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
      if (labels[i] == class_id) {
        indices.push_back(static_cast<unsigned int>(i));
      }
    }
    update_submap_colors(submap_id, indices);
  }
}

void PointLabelLayer::update_submap_colors(int submap_id, const std::vector<unsigned int>& point_indices) {
  if (current_display_mode != PointDisplayMode::SEGMENTATION) {
    return;
  }

  if (point_indices.empty() || submap_id < 0 || submap_id >= static_cast<int>(labels_by_submap.size()) || submap_id >= static_cast<int>(drawables.size()) ||
      !drawables[submap_id]) {
    return;
  }

  ColorVector colors(point_indices.size());
  for (std::size_t i = 0; i < point_indices.size(); ++i) {
    colors[i] = color_for_label(labels_by_submap[submap_id][point_indices[i]]);
  }
  drawables[submap_id]->update_color_with_indices(colors, point_indices);
}

Eigen::Vector4f PointLabelLayer::default_color(int class_id) {
  if (class_id <= 0) {
    return Eigen::Vector4f(0.45f, 0.45f, 0.45f, 1.0f);
  }

  switch ((class_id - 1) % 12) {
    case 0:
      return Eigen::Vector4f(0.90f, 0.16f, 0.16f, 1.0f);
    case 1:
      return Eigen::Vector4f(0.10f, 0.62f, 0.25f, 1.0f);
    case 2:
      return Eigen::Vector4f(0.16f, 0.40f, 0.92f, 1.0f);
    case 3:
      return Eigen::Vector4f(0.95f, 0.62f, 0.10f, 1.0f);
    case 4:
      return Eigen::Vector4f(0.62f, 0.22f, 0.85f, 1.0f);
    case 5:
      return Eigen::Vector4f(0.00f, 0.68f, 0.70f, 1.0f);
    case 6:
      return Eigen::Vector4f(0.95f, 0.32f, 0.62f, 1.0f);
    case 7:
      return Eigen::Vector4f(0.55f, 0.72f, 0.10f, 1.0f);
    case 8:
      return Eigen::Vector4f(0.26f, 0.72f, 0.95f, 1.0f);
    case 9:
      return Eigen::Vector4f(0.95f, 0.46f, 0.22f, 1.0f);
    case 10:
      return Eigen::Vector4f(0.70f, 0.70f, 0.18f, 1.0f);
    default:
      return Eigen::Vector4f(0.55f, 0.38f, 0.25f, 1.0f);
  }
}

std::string PointLabelLayer::default_name(int class_id) {
  return class_id <= 0 ? "unlabeled" : "class_" + std::to_string(class_id);
}

}  // namespace glim_manual_segmentation_editor
