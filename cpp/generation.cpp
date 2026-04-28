#include "placement/generation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr double kMinMacroArea = 100.0;
constexpr double kMaxMacroArea = 10000.0;
constexpr double kStandardCellHeight = 1.0;
constexpr int64_t kMinStandardCellPins = 3;
constexpr int64_t kMaxStandardCellPins = 6;
constexpr double kPinSize = 0.1;
constexpr double kTwoPi = 6.28318530717958647692;

int64_t featureIndex(placement::CellFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

int64_t featureIndex(placement::PinFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

}  // namespace

namespace placement {

using namespace torch::indexing;

PlacementProblem generatePlacementInput(
    int num_macros,
    int num_std_cells,
    const torch::Device& device,
    bool verbose) {
    const int64_t macro_count = num_macros;
    const int64_t std_cell_count = num_std_cells;
    const int64_t total_cells = macro_count + std_cell_count;
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto long_options = torch::TensorOptions().dtype(torch::kInt64).device(device);

    auto macro_areas =
        torch::rand({macro_count}, float_options) * (kMaxMacroArea - kMinMacroArea) +
        kMinMacroArea;

    const auto standard_areas =
        torch::tensor({1.0F, 2.0F, 3.0F}, float_options);
    const auto std_area_indices =
        torch::randint(0, standard_areas.size(0), {std_cell_count}, long_options);
    auto std_cell_areas = standard_areas.index_select(0, std_area_indices);
    auto areas = torch::cat({macro_areas, std_cell_areas});

    auto macro_widths = torch::sqrt(macro_areas);
    auto macro_heights = torch::sqrt(macro_areas);
    auto std_cell_widths = std_cell_areas / kStandardCellHeight;
    auto std_cell_heights =
        torch::full({std_cell_count}, kStandardCellHeight, float_options);
    auto cell_widths = torch::cat({macro_widths, std_cell_widths});
    auto cell_heights = torch::cat({macro_heights, std_cell_heights});

    auto num_pins_per_cell = torch::zeros({total_cells}, long_options);
    for (int64_t i = 0; i < macro_count; ++i) {
        const int64_t sqrt_area =
            static_cast<int64_t>(std::sqrt(macro_areas[i].item<double>()));
        num_pins_per_cell.index_put_(
            {i},
            torch::randint(sqrt_area, 2 * sqrt_area + 1, {1}, long_options)[0]);
    }
    if (std_cell_count > 0) {
        num_pins_per_cell.index_put_(
            {Slice(macro_count, total_cells)},
            torch::randint(
                kMinStandardCellPins,
                kMaxStandardCellPins + 1,
                {std_cell_count},
                long_options));
    }

    auto cell_features = torch::zeros({total_cells, 6}, float_options);
    cell_features.index_put_({Slice(), featureIndex(CellFeatureIdx::Area)}, areas);
    cell_features.index_put_(
        {Slice(), featureIndex(CellFeatureIdx::NumPins)},
        num_pins_per_cell.to(torch::kFloat32));
    cell_features.index_put_({Slice(), featureIndex(CellFeatureIdx::X)}, 0.0);
    cell_features.index_put_({Slice(), featureIndex(CellFeatureIdx::Y)}, 0.0);
    cell_features.index_put_(
        {Slice(), featureIndex(CellFeatureIdx::Width)},
        cell_widths);
    cell_features.index_put_(
        {Slice(), featureIndex(CellFeatureIdx::Height)},
        cell_heights);

    const int64_t total_pins = num_pins_per_cell.sum().item<int64_t>();
    auto pin_features = torch::zeros({total_pins, 7}, float_options);

    int64_t pin_idx = 0;
    for (int64_t cell_idx = 0; cell_idx < total_cells; ++cell_idx) {
        const int64_t n_pins = num_pins_per_cell[cell_idx].item<int64_t>();
        const double cell_width = cell_widths[cell_idx].item<double>();
        const double cell_height = cell_heights[cell_idx].item<double>();
        const double margin = kPinSize / 2.0;

        torch::Tensor pin_x;
        torch::Tensor pin_y;
        if (cell_width > 2.0 * margin && cell_height > 2.0 * margin) {
            pin_x = torch::rand({n_pins}, float_options) * (cell_width - 2.0 * margin) +
                    margin;
            pin_y = torch::rand({n_pins}, float_options) * (cell_height - 2.0 * margin) +
                    margin;
        } else {
            pin_x = torch::full({n_pins}, cell_width / 2.0, float_options);
            pin_y = torch::full({n_pins}, cell_height / 2.0, float_options);
        }

        const auto rows = Slice(pin_idx, pin_idx + n_pins);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::CellIdx)}, cell_idx);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::PinX)}, pin_x);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::PinY)}, pin_y);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::X)}, pin_x);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::Y)}, pin_y);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::Width)}, kPinSize);
        pin_features.index_put_({rows, featureIndex(PinFeatureIdx::Height)}, kPinSize);

        pin_idx += n_pins;
    }

    auto pin_to_cell = torch::zeros({total_pins}, long_options);
    pin_idx = 0;
    for (int64_t cell_idx = 0; cell_idx < total_cells; ++cell_idx) {
        const int64_t n_pins = num_pins_per_cell[cell_idx].item<int64_t>();
        pin_to_cell.index_put_({Slice(pin_idx, pin_idx + n_pins)}, cell_idx);
        pin_idx += n_pins;
    }

    std::vector<std::pair<int64_t, int64_t>> edges;
    std::vector<std::unordered_set<int64_t>> adjacency(total_pins);
    for (int64_t pin = 0; pin < total_pins; ++pin) {
        const int64_t num_connections = torch::randint(1, 4, {1}, long_options)[0].item<int64_t>();
        for (int64_t connection = 0; connection < num_connections; ++connection) {
            const int64_t other_pin =
                torch::randint(0, total_pins, {1}, long_options)[0].item<int64_t>();
            if (other_pin == pin || adjacency[pin].contains(other_pin)) {
                continue;
            }
            edges.emplace_back(std::min(pin, other_pin), std::max(pin, other_pin));
            adjacency[pin].insert(other_pin);
            adjacency[other_pin].insert(pin);
        }
    }

    torch::Tensor edge_list;
    if (edges.empty()) {
        edge_list = torch::zeros({0, 2}, long_options);
    } else {
        std::vector<int64_t> flat_edges;
        flat_edges.reserve(edges.size() * 2);
        for (const auto& edge : edges) {
            flat_edges.push_back(edge.first);
            flat_edges.push_back(edge.second);
        }
        edge_list = torch::from_blob(
                        flat_edges.data(),
                        {static_cast<int64_t>(edges.size()), 2},
                        torch::TensorOptions().dtype(torch::kInt64))
                        .clone()
                        .to(device);
    }

    if (verbose) {
        std::cout << "\nGenerated placement data:\n";
        std::cout << "  Total cells: " << total_cells << "\n";
        std::cout << "  Total pins: " << total_pins << "\n";
        std::cout << "  Total edges: " << edge_list.size(0) << "\n";
        const double avg_edges_per_pin =
            total_pins == 0 ? 0.0 : 2.0 * static_cast<double>(edge_list.size(0)) / total_pins;
        std::cout << "  Average edges per pin: " << avg_edges_per_pin << "\n";
    }

    return {cell_features, pin_features, edge_list};
}

void initializeCellPositions(torch::Tensor& cell_features, double spread_scale) {
    const int64_t total_cells = cell_features.size(0);
    const double total_area =
        cell_features.index({Slice(), featureIndex(CellFeatureIdx::Area)})
            .sum()
            .item<double>();
    const double spread_radius =
        std::max(std::sqrt(total_area) * spread_scale, 1.0);
    const auto options = cell_features.options();

    const auto angles = torch::rand({total_cells}, options) * kTwoPi;
    const auto radii = torch::rand({total_cells}, options) * spread_radius;
    cell_features.index_put_(
        {Slice(), featureIndex(CellFeatureIdx::X)},
        radii * torch::cos(angles));
    cell_features.index_put_(
        {Slice(), featureIndex(CellFeatureIdx::Y)},
        radii * torch::sin(angles));
}

}  // namespace placement
