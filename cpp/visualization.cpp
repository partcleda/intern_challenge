#include "placement/visualization.h"

#include "placement/metrics.h"
#include "placement/types.h"

#define WITHOUT_NUMPY
#include <matplotlibcpp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace torch::indexing;
namespace plt = matplotlibcpp;

#ifndef PLACEMENT_PYTHON_SITE_PACKAGES
#define PLACEMENT_PYTHON_SITE_PACKAGES ""
#endif

int64_t featureIndex(placement::CellFeatureIdx idx) {
    return static_cast<int64_t>(idx);
}

struct CellRect {
    double center_x = 0.0;
    double center_y = 0.0;
    double width = 0.0;
    double height = 0.0;
};

struct PanelData {
    torch::Tensor cells;
    std::vector<CellRect> rects;
    placement::OverlapMetrics metrics;
    double min_x = -10.0;
    double max_x = 10.0;
    double min_y = -10.0;
    double max_y = 10.0;
};

torch::Tensor prepareCellFeatures(const torch::Tensor& cell_features) {
    if (!cell_features.defined()) {
        throw std::invalid_argument("cell feature tensor must be defined");
    }
    if (cell_features.dim() != 2) {
        throw std::invalid_argument("cell feature tensor must be two-dimensional");
    }
    if (cell_features.size(0) > 0 &&
        cell_features.size(1) <= featureIndex(placement::CellFeatureIdx::Height)) {
        throw std::invalid_argument(
            "cell feature tensor must contain area, pins, x, y, width, and height");
    }

    return cell_features.detach()
        .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64))
        .contiguous();
}

double scalarAt(const torch::Tensor& cells, int64_t row, placement::CellFeatureIdx idx) {
    return cells.index({row, featureIndex(idx)}).item<double>();
}

bool isFinite(double value) {
    return std::isfinite(value);
}

PanelData buildPanelData(const torch::Tensor& cell_features) {
    PanelData panel;
    panel.cells = prepareCellFeatures(cell_features);
    panel.metrics = placement::calculateOverlapMetrics(panel.cells);

    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    const int64_t num_cells = panel.cells.size(0);
    panel.rects.reserve(static_cast<std::size_t>(num_cells));
    for (int64_t index = 0; index < num_cells; ++index) {
        CellRect rect;
        rect.center_x = scalarAt(panel.cells, index, placement::CellFeatureIdx::X);
        rect.center_y = scalarAt(panel.cells, index, placement::CellFeatureIdx::Y);
        rect.width = scalarAt(panel.cells, index, placement::CellFeatureIdx::Width);
        rect.height = scalarAt(panel.cells, index, placement::CellFeatureIdx::Height);
        panel.rects.push_back(rect);

        if (isFinite(rect.center_x) && isFinite(rect.center_y)) {
            min_x = std::min(min_x, rect.center_x);
            max_x = std::max(max_x, rect.center_x);
            min_y = std::min(min_y, rect.center_y);
            max_y = std::max(max_y, rect.center_y);
        }
    }

    if (min_x <= max_x && min_y <= max_y) {
        constexpr double kMargin = 10.0;
        panel.min_x = min_x - kMargin;
        panel.max_x = max_x + kMargin;
        panel.min_y = min_y - kMargin;
        panel.max_y = max_y + kMargin;
    }

    return panel;
}

std::string formatTitle(const std::string& title, const placement::OverlapMetrics& metrics) {
    std::ostringstream output;
    output << title << "\nOverlaps: " << metrics.overlap_count
           << ", Total Overlap Area: " << std::fixed << std::setprecision(2)
           << metrics.total_overlap_area;
    return output.str();
}

void drawCell(const CellRect& rect) {
    if (!isFinite(rect.center_x) || !isFinite(rect.center_y) || !isFinite(rect.width) ||
        !isFinite(rect.height) || rect.width <= 0.0 || rect.height <= 0.0) {
        return;
    }

    const double left = rect.center_x - rect.width / 2.0;
    const double right = rect.center_x + rect.width / 2.0;
    const double bottom = rect.center_y - rect.height / 2.0;
    const double top = rect.center_y + rect.height / 2.0;
    const std::vector<double> xs = {left, right, right, left, left};
    const std::vector<double> ys = {bottom, bottom, top, top, bottom};

    if (!plt::fill(xs, ys, {{"color", "lightblue"}})) {
        throw std::runtime_error("Call to fill() failed.");
    }
    if (!plt::plot(xs, ys, "b-")) {
        throw std::runtime_error("Call to plot() failed.");
    }
}

void setEqualAspect() {
    PyObject* axes = PyObject_CallObject(
        matplotlibcpp::detail::_interpreter::get().s_python_function_gca,
        matplotlibcpp::detail::_interpreter::get().s_python_empty_tuple);
    if (axes == nullptr) {
        throw std::runtime_error("Call to gca() failed.");
    }

    PyObject* result = PyObject_CallMethod(
        axes,
        const_cast<char*>("set_aspect"),
        "ss",
        "equal",
        "box");
    Py_DECREF(axes);
    if (result == nullptr) {
        throw std::runtime_error("Call to set_aspect() failed.");
    }
    Py_DECREF(result);
}

void drawPanel(const PanelData& panel, const std::string& title) {
    for (const CellRect& rect : panel.rects) {
        drawCell(rect);
    }

    plt::title(formatTitle(title, panel.metrics));
    setEqualAspect();
    plt::grid(true);
    plt::xlim(panel.min_x, panel.max_x);
    plt::ylim(panel.min_y, panel.max_y);
}

void selectSubplot(long plot_number) {
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyLong_FromLong(plot_number));

    PyObject* result = PyObject_CallObject(
        matplotlibcpp::detail::_interpreter::get().s_python_function_subplot,
        args);
    Py_DECREF(args);
    if (result == nullptr) {
        throw std::runtime_error("Call to subplot() failed.");
    }
    Py_DECREF(result);
}

void setEnvVar(const std::string& name, const std::string& value, bool overwrite) {
#ifdef _WIN32
    if (!overwrite && std::getenv(name.c_str()) != nullptr) {
        return;
    }
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), overwrite ? 1 : 0);
#endif
}

void prependEnvPath(const std::string& name, const std::string& path) {
    if (path.empty()) {
        return;
    }

    const char* current_value = std::getenv(name.c_str());
    if (current_value == nullptr || std::string(current_value).empty()) {
        setEnvVar(name, path, true);
        return;
    }

    const std::string current(current_value);
    if (current == path) {
        return;
    }

#ifdef _WIN32
    constexpr char kPathSeparator = ';';
#else
    constexpr char kPathSeparator = ':';
#endif
    const std::string path_with_separator = path + kPathSeparator;
    const std::string separator_with_path =
        std::string(1, kPathSeparator) + path + kPathSeparator;
    const bool ends_with_path =
        current.size() > path.size() &&
        current.compare(current.size() - path.size(), path.size(), path) == 0 &&
        current[current.size() - path.size() - 1] == kPathSeparator;
    if (current.rfind(path_with_separator, 0) == 0 ||
        current.find(separator_with_path) != std::string::npos || ends_with_path) {
        return;
    }

    setEnvVar(name, path + kPathSeparator + current, true);
}

void configureHeadlessBackend() {
    prependEnvPath("PYTHONPATH", PLACEMENT_PYTHON_SITE_PACKAGES);

    const std::filesystem::path config_dir =
        std::filesystem::temp_directory_path() / "matplotlib-cpp";
    const std::filesystem::path cache_dir =
        std::filesystem::temp_directory_path() / "matplotlib-cpp-cache";
    std::filesystem::create_directories(config_dir);
    std::filesystem::create_directories(cache_dir);

    setEnvVar("MPLBACKEND", "Agg", true);
    setEnvVar("MPLCONFIGDIR", config_dir.string(), true);
    setEnvVar("XDG_CACHE_HOME", cache_dir.string(), false);
}

void closePlotIgnoringErrors() {
    try {
        plt::close();
    } catch (...) {
    }
}

}  // namespace

namespace placement {

void plotPlacement(
    const torch::Tensor& initial_cell_features,
    const torch::Tensor& final_cell_features,
    const std::filesystem::path& output_path) {
    configureHeadlessBackend();

    const PanelData initial_panel = buildPanelData(initial_cell_features);
    const PanelData final_panel = buildPanelData(final_cell_features);

    const std::filesystem::path parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    try {
        plt::figure_size(1600, 800);
        selectSubplot(121);
        drawPanel(initial_panel, "Initial Placement");
        selectSubplot(122);
        drawPanel(final_panel, "Final Placement");
        plt::tight_layout();
        plt::save(output_path.string());
        plt::close();
    } catch (const std::exception& error) {
        if (PyErr_Occurred() != nullptr) {
            PyErr_Print();
        }
        closePlotIgnoringErrors();
        throw std::runtime_error(
            "Unable to render placement image to " + output_path.string() +
            ": " + error.what());
    } catch (...) {
        if (PyErr_Occurred() != nullptr) {
            PyErr_Print();
        }
        closePlotIgnoringErrors();
        throw std::runtime_error(
            "Unable to render placement image to " + output_path.string());
    }
}

}  // namespace placement
