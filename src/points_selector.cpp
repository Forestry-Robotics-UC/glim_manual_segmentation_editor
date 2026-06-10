/**
 * Code adapted from koide3/glim (MIT https://github.com/koide3/glim/blob/master/LICENSE)
 * https://github.com/koide3/glim/blob/master/src/glim/viewer/editor/points_selector.cpp
 */
#include <glim_manual_segmentation_editor/points_selector.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <thread>

#include <glk/drawable_container.hpp>
#include <glk/html_colors.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/model_control.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <gtsam_points/ann/kdtree2.hpp>
#include <gtsam_points/util/fast_floor.hpp>
#include <gtsam_points/util/normal_estimation.hpp>

namespace glim_manual_segmentation_editor {

PointsSelector::PointsSelector(std::shared_ptr<spdlog::logger> logger) : picked_point(0, 0, 0), logger(logger) {
  num_threads = std::min<int>(std::thread::hardware_concurrency(), 16);
  logger->info("Num threads: {}", num_threads);

  model_control.reset(new guik::ModelControl("model_control"));
  model_control->set_gizmo_operation("UNIVERSAL");

  map_cell_resolution = 2.0f;
  cell_selection_window = 5;

  show_preferences_window = false;
  show_picked_point = false;
  show_selection_radius = false;

  show_cells = false;
  show_gizmo = false;
  selected_tool = 0;

  select_radius = 2.0f;
  outlier_radius_offset = 1.0f;
  outliner_num_neighbors = 10;
  outlier_stddev_thresh = 2.0f;

  show_segmentation_radius = false;
  segmentation_method = 0;

  min_cut_params.foreground_weight = 10.0;
  min_cut_params.foreground_mask_radius = 0.5f;
  min_cut_params.background_mask_radius = 5.0f;

  auto viewer = guik::viewer();
  viewer->register_drawable_filter("manual_segmentation_filter", [this](const std::string& name) {
    if (!show_cells && name == "cells") {
      return false;
    }
    return true;
  });
  viewer->shader_setting().set_point_shape_circle();
  viewer->shader_setting().set_point_scale_metric();
  viewer->shader_setting().set_point_size(0.05f);
}

PointsSelector::~PointsSelector() {
  guik::viewer()->remove_drawable_filter("manual_segmentation_filter");
}

void PointsSelector::clear() {
  submaps.clear();
  map_cells.clear();
  selected_point_ids.clear();
  picked_point.setZero();

  guik::viewer()->clear_drawables();
}

void PointsSelector::set_submaps(const std::vector<glim::SubMap::Ptr>& submaps) {
  this->submaps = submaps;
  submap_drawables.resize(submaps.size());
  std::transform(submaps.begin(), submaps.end(), submap_drawables.begin(), [](const glim::SubMap::Ptr& submap) {
    return guik::viewer()->update_points("submap_" + std::to_string(submap->id), submap->frame->points, submap->frame->size(), guik::Rainbow(submap->T_world_origin));
  });

  update_cells();
}

const std::vector<std::uint64_t>& PointsSelector::selected_point_ids_ref() const {
  return selected_point_ids;
}

const std::vector<glk::PointCloudBuffer::Ptr>& PointsSelector::submap_drawables_ref() const {
  return submap_drawables;
}

void PointsSelector::clear_selected_points() {
  selected_point_ids.clear();
  guik::viewer()->remove_drawable("selected");
}

std::vector<std::uint64_t> PointsSelector::collect_neighbor_point_ids(const Eigen::Vector3d& point) {
  const Eigen::Vector3i center = (point.array() / map_cell_resolution).floor().cast<int>();

  std::vector<MapCell::Ptr> selected_cells;
  const int w = cell_selection_window;
  for (int i = -w; i <= w; i++) {
    for (int j = -w; j <= w; j++) {
      for (int k = -w; k <= w; k++) {
        const Eigen::Vector3i coord = center + Eigen::Vector3i(i, j, k);
        auto it = map_cells.find(coord);
        if (it != map_cells.end()) {
          selected_cells.push_back(it->second);
        }
      }
    }
  }

  std::vector<std::uint64_t> point_ids;
  std::size_t num_points = 0;
  for (const auto& cell : selected_cells) {
    num_points += cell->point_ids.size();
  }
  point_ids.reserve(num_points);

  for (const auto& cell : selected_cells) {
    point_ids.insert(point_ids.end(), cell->point_ids.begin(), cell->point_ids.end());
  }
  logger->info("|selected_cells|={} |selected_points|={}", selected_cells.size(), point_ids.size());

  return point_ids;
}

gtsam_points::PointCloudCPU::Ptr PointsSelector::collect_submap_points(const std::vector<std::uint64_t>& point_ids) {
  if (submaps.empty()) {
    return std::make_shared<gtsam_points::PointCloudCPU>();
  }

  const bool has_normals = submaps[0]->frame->has_normals();
  const bool has_covs = submaps[0]->frame->has_covs();

  auto points = std::make_shared<gtsam_points::PointCloudCPU>();
  points->points_storage.resize(point_ids.size());
  if (has_normals) {
    points->normals_storage.resize(point_ids.size());
  }
  if (has_covs) {
    points->covs_storage.resize(point_ids.size());
  }

#pragma omp parallel for num_threads(num_threads) schedule(guided, 32)
  for (int i = 0; i < static_cast<int>(point_ids.size()); i++) {
    const auto x = point_ids[i];
    const std::uint64_t submap_id = x >> 32;
    const std::uint64_t point_id = x & ((1ull << 32) - 1);

    const auto& submap = submaps[submap_id];

    points->points_storage[i] = submap->T_world_origin * submap->frame->points[point_id];
    if (has_normals) {
      points->normals_storage[i] = submap->T_world_origin.matrix() * submap->frame->normals[point_id];
    }
    if (has_covs) {
      points->covs_storage[i] = submap->T_world_origin.matrix() * submap->frame->covs[point_id] * submap->T_world_origin.matrix().transpose();
    }
  }

  points->num_points = points->points_storage.size();
  points->points = points->points_storage.data();
  points->normals = points->normals_storage.data();
  points->covs = points->covs_storage.data();

  return points;
}

void PointsSelector::update_cells() {
  map_cells.clear();

  logger->info("Update cells");
  std::size_t total_points = 0;
  for (const auto& submap : submaps) {
    if (submap && submap->frame) {
      total_points += submap->frame->size();
    }
  }
  map_cells.reserve(std::max<std::size_t>(map_cells.bucket_count(), total_points / 8));

  const double inv_resolution = 1.0 / map_cell_resolution;
  for (const auto& submap : submaps) {
    if (!submap || !submap->frame) {
      continue;
    }

    for (int i = 0; i < submap->frame->size(); i++) {
      const Eigen::Vector4d pt = submap->T_world_origin * submap->frame->points[i];
      const Eigen::Vector3i coord = gtsam_points::fast_floor(pt * inv_resolution).head<3>();

      auto& cell = map_cells[coord];
      if (!cell) {
        cell = std::make_shared<MapCell>(map_cell_resolution, coord);
      }
      cell->add_point(submap->id, i);
    }
  }

  if (!show_cells) {
    guik::viewer()->remove_drawable("cells");
    logger->info("|cells|={} (visualization hidden)", map_cells.size());
    return;
  }

  auto container = std::make_shared<glk::DrawableContainer>(false);
  for (const auto& cell : map_cells) {
    const auto& coord = cell.first;
    const auto& map_cell = cell.second;

    const Eigen::Vector3d center = coord.cast<double>() * map_cell_resolution + Eigen::Vector3d::Constant(map_cell_resolution * 0.5);
    container->push_back(glk::Primitives::wire_cube(), guik::FlatGreen().translate(center).scale(map_cell_resolution).set_alpha(0.5));
  }

  guik::viewer()->update_drawable("cells", container, guik::FlatGreen().set_alpha(0.5));
  logger->info("|cells|={}", map_cells.size());
}

void PointsSelector::draw_ui() {
  auto viewer = guik::viewer();

  const auto show_note = [](const std::string& text) {
    if (ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::Text("%s", text.c_str());
      ImGui::EndTooltip();
    }
    return false;
  };

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Preferences")) {
        show_preferences_window = true;
      }
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }

  if (show_preferences_window && ImGui::Begin("Preferenecs", &show_preferences_window, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::DragInt("Num threads", &num_threads, 1, 1, std::thread::hardware_concurrency()) || show_note("Number of threads for processing");

    ImGui::Separator();

    auto& settings = viewer->shader_setting();
    auto point_size_ = settings.get<float>("point_size");
    float point_size = point_size_ ? *point_size_ : 10.0f;
    if (ImGui::DragFloat("Point size", &point_size, 0.01f, 0.01f, 100.0f)) {
      settings.set_point_size(point_size);
    }

    if (ImGui::Button("Rectangle") || show_note("Set point shape to rectangle")) {
      settings.set_point_shape_rectangle();
    }
    ImGui::SameLine();
    if (ImGui::Button("Circle") || show_note("Set point shape to circle")) {
      settings.set_point_shape_circle();
    }

    ImGui::End();
  }

  ImGui::SetNextWindowPos(ImVec2(10.0f, 30.0f), ImGuiCond_FirstUseEver);
  if (ImGui::Begin("Editor", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Submaps: %ld  Cells: %ld", submaps.size(), map_cells.size());
    ImGui::Text("Selected points: %ld", selected_point_ids.size());

    ImGui::Separator();
    if (ImGui::BeginMenu("Map cells config")) {
      show_note("Config of map cells for rough neighbor points extraction");

      if (ImGui::Checkbox("Show cells", &show_cells)) {
        if (show_cells) {
          update_cells();
        } else {
          viewer->remove_drawable("cells");
        }
      }
      show_note("Show map cells");
      ImGui::DragFloat("map cell resolution", &map_cell_resolution, 0.1f, 0.1f, 10.0f) || show_note("Map cell resolution [m]");
      ImGui::DragInt("selection window", &cell_selection_window, 1, 1, 10) || show_note("Neighbor cell selection window");
      if (ImGui::MenuItem("Update cells")) {
        update_cells();
      }

      ImGui::EndMenu();
    }

    ImGui::Separator();
    ImGui::Checkbox("##Show_gizmo", &show_gizmo);
    show_note("Show gizmo for selection tool");

    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("Selection tool", &selected_tool, "Box\0Sphere\0")) {
      show_gizmo = true;
    }

    if (ImGui::Button("Reset") || show_note("Reset gizmo size")) {
      Eigen::Matrix4f matrix = model_control->model_matrix();
      matrix.block<3, 3>(0, 0).setIdentity();
      model_control->set_model_matrix(matrix);
      show_gizmo = true;
    }

    ImGui::SameLine();
    if (ImGui::Button("Select points") || (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S)) || show_note("Select points within the selection tool")) {
      select_points_tool();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(Ctrl+S)");

    if (show_gizmo) {
      model_control->draw_gizmo();
    }

    ImGui::Separator();
    if (ImGui::BeginMenu("GroundGrid config")) {
      ImGui::DragFloat("Grid resolution", &ground_grid_config.resolution, 0.01f, 0.05f, 5.0f, "%.2f m");
      show_note("GroundGrid raster cell size");
      ImGui::DragInt("Cell variance point threshold", &ground_grid_config.point_count_cell_variance_threshold, 1, 1, 1000);
      show_note("Use only the center cell variance after this point count");
      ImGui::DragFloat("Distance factor", &ground_grid_config.distance_factor, 0.00001f, 0.0f, 0.01f, "%.6f");
      show_note("Ground patch variance threshold distance factor");
      ImGui::DragFloat("Minimum distance factor", &ground_grid_config.min_distance_factor, 0.00001f, 0.0f, 0.01f, "%.6f");
      show_note("Minimum GroundGrid distance factor");
      ImGui::DragFloat("Ground height threshold", &ground_grid_config.min_point_height_threshold, 0.01f, 0.01f, 2.0f, "%.2f m");
      show_note("Points below ground height plus this threshold are selected as ground");
      ImGui::DragFloat("Obstacle height threshold", &ground_grid_config.min_point_height_obstacle_threshold, 0.01f, 0.01f, 2.0f, "%.2f m");
      show_note("Minimum ground tolerance close to rough/obstacle cells");
      ImGui::DragFloat("Minimum patch point ratio", &ground_grid_config.min_ground_patch_detection_point_count_threshold, 0.01f, 0.01f, 2.0f, "%.2f");
      show_note("GroundGrid patch point-count threshold as a ratio of expected points");
      ImGui::DragFloat("Patch size change distance", &ground_grid_config.patch_size_change_distance, 0.5f, 1.0f, 200.0f, "%.1f m");
      show_note("Use 5x5 patches after this distance from the grid center");
      ImGui::DragFloat("Horizontal angular distance", &ground_grid_config.horizontal_point_angular_distance, 0.0001f, 0.0001f, 0.1f, "%.6f rad");
      show_note("LiDAR horizontal angular resolution used for expected point count");
      ImGui::DragInt("Max grid cells", &ground_grid_config.max_grid_cells, 10000, 10000, 50000000);
      show_note("The selector coarsens the grid if the map would exceed this many cells");
      ImGui::EndMenu();
    }

    if (ImGui::Button("Select Ground Points") || show_note("Automatically select ground points with GroundGrid")) {
      select_ground_points();
    }

    if (ImGui::Button("Unselect points") || show_note("Clear selected points")) {
      clear_selected_points();
    }

    ImGui::End();
  }

  auto pick = viewer->pick_point(1, 3);
  if (pick) {
    picked_point = pick->cast<double>();

    auto neighbor_point_ids = collect_neighbor_point_ids(pick->cast<double>());
    auto selected_points = collect_submap_points(neighbor_point_ids);
    viewer->update_points("selected_cells", selected_points->points, selected_points->size(), guik::FlatOrange().set_alpha(0.8).set_point_scale(2.0f));

    ImGui::OpenPopup("context_menu");
  }

  show_picked_point = false;
  show_selection_radius = false;
  show_segmentation_radius = false;

  if (ImGui::BeginPopup("context_menu", ImGuiWindowFlags_AlwaysAutoResize)) {
    show_picked_point = true;

    ImGui::Text("Point: (%.1f, %.1f, %.1f)", picked_point.x(), picked_point.y(), picked_point.z());
    ImGui::Text("Selected points: %ld", selected_point_ids.size());
    ImGui::Separator();

    if (ImGui::MenuItem("Move camera here")) {
      viewer->lookat(picked_point);
      viewer->reset_center();
    }

    const bool gizmo_menu_open = ImGui::BeginMenu("Selection tool (Gizmo)");
    const bool gizmo_selection_clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);

    if (gizmo_menu_open) {
      if (ImGui::MenuItem("Show gizmo here") || gizmo_selection_clicked) {
        Eigen::Matrix4f matrix = model_control->model_matrix();
        matrix.block<3, 1>(0, 3) = picked_point.cast<float>();
        model_control->set_model_matrix(matrix);
        show_gizmo = true;
        selected_point_ids.clear();
      }

      if (ImGui::MenuItem("Hide gizmo")) {
        show_gizmo = false;
        selected_point_ids.clear();
      }

      if (ImGui::MenuItem("Select")) {
        logger->info("Select points with gizmo clicked");
        select_points_tool();
      }

      ImGui::EndMenu();
    }

    if (gizmo_selection_clicked) {
      ImGui::CloseCurrentPopup();
    }

    const bool radius_menu_open = ImGui::BeginMenu("Radius tools");
    if (radius_menu_open) {
      show_selection_radius = true;
      ImGui::DragFloat("Radius", &select_radius, 0.01f, 0.01f, 100.0f);

      if (ImGui::BeginMenu("Outlier selection")) {
        ImGui::DragFloat("Radius offset", &outlier_radius_offset, 0.01f, 0.01f, 100.0f);
        show_note("Radius offset for statistics computation");
        ImGui::DragInt("Num neighbors", &outliner_num_neighbors, 1, 1, 100);
        show_note("Number of neighbors for statistics computation");
        ImGui::DragFloat("Stddev threshold", &outlier_stddev_thresh, 0.01f, 0.01f, 10.0f);
        show_note("Threshold for outlier detection");
        ImGui::EndMenu();
      }

      if (ImGui::MenuItem("Select outliers within radius")) {
        logger->info("Select outliers in radius clicked");
        select_outlier_points_radius();
      }

      if (ImGui::MenuItem("Select within radius")) {
        logger->info("Select in radius clicked");
        select_points_radius();
      }
      show_note("Select points within the radius");
      ImGui::EndMenu();
    }

    const bool segmentation_menu_open = ImGui::BeginMenu("Segmentation");
    bool segmentation_clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);

    if (segmentation_menu_open) {
      show_segmentation_radius = true;
      ImGui::Combo("Method", &segmentation_method, "MinCut\0RegionGrowing\0");
      show_note(
        "Segmentation method\n"
        "- MinCut: Graph cut based segmentation for extracting an object in a radius\n"
        "- RegionGrowing: Region growing based segmentation for extracting a connected region");

      if (segmentation_method == 0) {
        float distance_sigma = min_cut_params.distance_sigma;
        float angle_sigma = min_cut_params.angle_sigma * 180.0 / M_PI;
        float foreground_mask_radius = min_cut_params.foreground_mask_radius;
        float background_mask_radius = min_cut_params.background_mask_radius;

        ImGui::DragFloat("Distance sigma", &distance_sigma, 0.01f, 0.01f, 10.0f);
        show_note("Disance proximity width [m]");
        ImGui::DragFloat("Angle sigma", &angle_sigma, 0.01f, 0.01f, 180.0f);
        show_note("Angle proximity width [deg]");
        ImGui::DragFloat("Foreground radius", &foreground_mask_radius, 0.01f, 0.01f, 100.0f);
        show_note("Points within this radius are marked as foreground");
        ImGui::DragFloat("Background radius", &background_mask_radius, 0.01f, 0.01f, 100.0f);
        show_note("Points out of this radius are marked as background");
        ImGui::DragInt("K neighbors", &min_cut_params.k_neighbors, 1, 1, 100);
        show_note("Number of neighbors for mincut graph contruction");

        min_cut_params.distance_sigma = distance_sigma;
        min_cut_params.angle_sigma = angle_sigma * M_PI / 180.0;
        min_cut_params.foreground_mask_radius = foreground_mask_radius;
        min_cut_params.background_mask_radius = background_mask_radius;

      } else {
        float distance_threshold = region_growing_params.distance_threshold;
        float angle_threshold = region_growing_params.angle_threshold * 180.0 / M_PI;
        float dilation_radius = region_growing_params.dilation_radius;

        ImGui::DragFloat("Distance threshold", &distance_threshold, 0.01f, 0.01f, 1.0f);
        show_note("Points connection distance threshold [m]");
        ImGui::DragFloat("Angle threshold", &angle_threshold, 0.01f, 0.01f, 180.0f);
        show_note("Points connection angle threshold [deg]");
        ImGui::DragFloat("Dilation radius", &dilation_radius, 0.01f, 0.01f, 100.0f);
        show_note("After performing region growing, the extracted region will be dilated by this radius to fill halls");

        region_growing_params.distance_threshold = distance_threshold;
        region_growing_params.angle_threshold = angle_threshold * M_PI / 180.0;
        region_growing_params.dilation_radius = dilation_radius;
      }

      segmentation_clicked |= ImGui::MenuItem("Segment");
      ImGui::EndMenu();
    }

    if (segmentation_clicked) {
      logger->info("Segmentation clicked");
      select_points_segmentation();
      ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
  } else {
    viewer->remove_drawable("selected_cells");
  }

  if (selected_point_ids.empty()) {
    viewer->remove_drawable("selected");
  }

  if (show_picked_point) {
    viewer->update_sphere("picked_point", guik::FlatRed().translate(picked_point).scale(0.2));
  } else {
    viewer->remove_drawable("picked_point");
  }

  if (show_selection_radius) {
    viewer->update_wire_sphere("selection_radius", guik::FlatRed().translate(picked_point).scale(select_radius).set_alpha(0.5));
  } else {
    viewer->remove_drawable("selection_radius");
  }

  if (show_segmentation_radius && segmentation_method == 0) {
    viewer->update_wire_sphere("mincut_foreground_radius", guik::FlatRed().translate(picked_point).scale(min_cut_params.foreground_mask_radius).set_alpha(0.5));
    viewer->update_wire_sphere("mincut_background_radius", guik::FlatBlue().translate(picked_point).scale(min_cut_params.background_mask_radius).set_alpha(0.5));
  } else {
    viewer->remove_drawable("mincut_foreground_radius");
    viewer->remove_drawable("mincut_background_radius");
  }

  if (show_gizmo) {
    switch (selected_tool) {
      case 0:
        viewer->update_cube("selection_tool", guik::FlatOrange(model_control->model_matrix()).set_alpha(0.5));
        break;
      case 1:
        viewer->update_sphere("selection_tool", guik::FlatOrange(model_control->model_matrix()).set_alpha(0.5));
        break;
    }
  } else {
    viewer->remove_drawable("selection_tool");
  }
}

void PointsSelector::select_points_tool() {
  if (!show_gizmo) {
    show_gizmo = true;
    logger->warn("Gizmo is not shown");
    return;
  }

  const Eigen::Matrix4d inv_model_matrix = model_control->model_matrix().cast<double>().inverse();
  selected_point_ids.clear();

#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
  for (int submap_idx = 0; submap_idx < static_cast<int>(submaps.size()); submap_idx++) {
    const auto& submap = submaps[submap_idx];
    const Eigen::Matrix4d T_local_world = inv_model_matrix * submap->T_world_origin.matrix();

    std::vector<std::uint64_t> local_selected_point_ids;
    local_selected_point_ids.reserve(submap->frame->size() / 4);

    for (int i = 0; i < submap->frame->size(); i++) {
      const Eigen::Vector4d pt = T_local_world * submap->frame->points[i];

      if (selected_tool == 0) {
        if ((pt.array() > Eigen::Array4d(-0.5, -0.5, -0.5, 0.0)).all() && (pt.array() < Eigen::Array4d(0.5, 0.5, 0.5, 2.0)).all()) {
          local_selected_point_ids.emplace_back((static_cast<std::uint64_t>(submap->id) << 32) | i);
        }
      } else {
        if (pt.head<3>().squaredNorm() < 1.0) {
          local_selected_point_ids.emplace_back((static_cast<std::uint64_t>(submap->id) << 32) | i);
        }
      }
    }

#pragma omp critical
    {
      selected_point_ids.insert(selected_point_ids.end(), local_selected_point_ids.begin(), local_selected_point_ids.end());
    }
  }

  if (selected_point_ids.empty()) {
    logger->warn("No points selected");
    return;
  }

  auto selected_points = collect_submap_points(selected_point_ids);
  guik::viewer()->update_points("selected", selected_points->points, selected_points->size(), guik::FlatOrange().set_alpha(0.8).set_point_scale(2.0f));
  show_gizmo = false;
}

void PointsSelector::select_points_radius() {
  const auto neighbor_point_ids = collect_neighbor_point_ids(picked_point);
  if (neighbor_point_ids.empty()) {
    logger->warn("No points selected");
    return;
  }

  const auto neighbor_points = collect_submap_points(neighbor_point_ids);
  selected_point_ids.clear();
  const double sq_thresh = select_radius * select_radius;
  for (int i = 0; i < neighbor_points->size(); i++) {
    const double dist_sq = (neighbor_points->points[i] - picked_point.homogeneous()).squaredNorm();
    if (dist_sq < sq_thresh) {
      selected_point_ids.emplace_back(neighbor_point_ids[i]);
    }
  }

  logger->info("Select points in radius: {} points selected", selected_point_ids.size());
  auto filtered = collect_submap_points(selected_point_ids);
  guik::viewer()->update_points("selected", filtered->points, filtered->size(), guik::FlatOrange().set_alpha(0.8).set_point_scale(2.0f));
}

void PointsSelector::select_outlier_points_radius() {
  const auto neighbor_point_ids = collect_neighbor_point_ids(picked_point);
  const auto neighbor_points = collect_submap_points(neighbor_point_ids);

  if (neighbor_point_ids.empty()) {
    logger->warn("No points selected");
    return;
  }

  constexpr std::uint64_t INLIER = std::numeric_limits<std::uint64_t>::max();
  std::vector<std::uint64_t> point_ids_in_radius;
  auto points_in_radius = std::make_shared<gtsam_points::PointCloudCPU>();

  const double sq_thresh = std::pow(select_radius, 2);
  const double offset_sq_thresh = std::pow(select_radius + outlier_radius_offset, 2);
  for (int i = 0; i < neighbor_points->size(); i++) {
    const double dist_sq = (neighbor_points->points[i] - picked_point.homogeneous()).squaredNorm();
    if (dist_sq < offset_sq_thresh) {
      point_ids_in_radius.emplace_back(dist_sq < sq_thresh ? neighbor_point_ids[i] : INLIER);
      points_in_radius->points_storage.emplace_back(neighbor_points->points[i]);
    }
  }

  points_in_radius->num_points = points_in_radius->points_storage.size();
  points_in_radius->points = points_in_radius->points_storage.data();

  if (points_in_radius->size() < outliner_num_neighbors) {
    logger->warn("Not enough points in radius");
    return;
  }

  const int k = outliner_num_neighbors;
  auto tree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(points_in_radius);
  std::vector<int> neighbors(k * points_in_radius->size());

  for (int i = 0; i < points_in_radius->size(); i++) {
    std::vector<size_t> k_indices(k);
    std::vector<double> k_sq_dists(k);
    tree->knn_search(points_in_radius->points[i].data(), k, k_indices.data(), k_sq_dists.data());
    std::copy(k_indices.begin(), k_indices.end(), neighbors.begin() + i * k);
  }

  auto inliers = gtsam_points::find_inlier_points(points_in_radius, neighbors, k, outlier_stddev_thresh);
  for (int idx : inliers) {
    point_ids_in_radius[idx] = INLIER;
  }

  auto remove_loc = std::remove_if(point_ids_in_radius.begin(), point_ids_in_radius.end(), [](std::uint64_t id) { return id == INLIER; });
  point_ids_in_radius.erase(remove_loc, point_ids_in_radius.end());

  selected_point_ids = point_ids_in_radius;
  logger->info("Select points in radius: {} points selected", selected_point_ids.size());

  auto filtered = collect_submap_points(selected_point_ids);
  guik::viewer()->update_points("selected", filtered->points, filtered->size(), guik::FlatOrange().set_alpha(0.8).set_point_scale(2.0f));
}

void PointsSelector::select_points_segmentation() {
  selected_point_ids.clear();

  const auto neighbor_point_ids = collect_neighbor_point_ids(picked_point);
  const auto neighbor_points = collect_submap_points(neighbor_point_ids);

  if (neighbor_point_ids.empty()) {
    logger->warn("No neighbor points");
    return;
  }

  if (segmentation_method == 0) {
    std::vector<std::uint64_t> filtered_ids;
    const double sq_dist_thresh = std::pow(min_cut_params.background_mask_radius + 1.0, 2);
    for (int i = 0; i < static_cast<int>(neighbor_point_ids.size()); i++) {
      const double sq_dist = (neighbor_points->points[i] - picked_point.homogeneous()).squaredNorm();
      if (sq_dist < sq_dist_thresh) {
        filtered_ids.emplace_back(neighbor_point_ids[i]);
      }
    }

    auto filtered = collect_submap_points(filtered_ids);
    filtered->add_normals(gtsam_points::estimate_normals(filtered->points, filtered->size(), 20, num_threads));
    auto filtered_tree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(filtered);

    min_cut_params.num_threads = num_threads;
    const auto result = gtsam_points::min_cut(*filtered, *filtered_tree, picked_point.homogeneous(), min_cut_params);

    for (int i = 0; i < static_cast<int>(result.cluster_indices.size()); i++) {
      selected_point_ids.emplace_back(filtered_ids[result.cluster_indices[i]]);
    }
  } else {
    auto filtered = collect_submap_points(neighbor_point_ids);
    auto kdtree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(filtered);

    region_growing_params.num_threads = num_threads;
    auto rg = gtsam_points::region_growing_init(*filtered, *kdtree, picked_point.homogeneous(), region_growing_params);
    gtsam_points::region_growing_update(rg, *filtered, *kdtree, region_growing_params);

    for (int i = 0; i < static_cast<int>(rg.cluster_indices.size()); i++) {
      selected_point_ids.emplace_back(neighbor_point_ids[rg.cluster_indices[i]]);
    }
  }

  auto selected_points = collect_submap_points(selected_point_ids);
  guik::viewer()->update_points("selected", selected_points->points, selected_points->size(), guik::FlatOrange().set_alpha(0.8).set_point_scale(2.0f));
  logger->info("Select points by segmentation: {} points selected", selected_points->size());
}

void PointsSelector::select_ground_points() {
  selected_point_ids.clear();

  if (submaps.empty()) {
    logger->warn("No map loaded");
    guik::viewer()->remove_drawable("selected");
    return;
  }

  const auto result = GroundGridSelector::select_ground_points(submaps, ground_grid_config, logger);
  selected_point_ids = result.point_ids;

  if (selected_point_ids.empty()) {
    logger->warn(
      "Select ground points: no points selected ({} input points, {} ground patches, {}x{} grid at {:.3f}m)",
      result.input_point_count,
      result.ground_patch_count,
      result.cell_count,
      result.cell_count,
      result.resolution);
    guik::viewer()->remove_drawable("selected");
    return;
  }

  auto selected_points = collect_submap_points(selected_point_ids);
  guik::viewer()->update_points("selected", selected_points->points, selected_points->size(), guik::FlatOrange().set_alpha(0.8).set_point_scale(2.0f));
  logger->info(
    "Select ground points: {} points selected from {} input points ({} ground patches, {}x{} grid at {:.3f}m)",
    selected_points->size(),
    result.input_point_count,
    result.ground_patch_count,
    result.cell_count,
    result.cell_count,
    result.resolution);
}

}  // namespace glim_manual_segmentation_editor
