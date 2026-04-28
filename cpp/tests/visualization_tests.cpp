#include "placement/visualization.h"

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to read visualization output");
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

}  // namespace

void visualizationWritesPngWithExpectedContent() {
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);

    const auto initial_cell_features = torch::tensor(
        {
            {4.0F, 1.0F, 0.0F, 0.0F, 2.0F, 2.0F},
            {4.0F, 1.0F, 1.0F, 0.0F, 2.0F, 2.0F},
        },
        float_options);
    const auto final_cell_features = torch::tensor(
        {
            {4.0F, 1.0F, 0.0F, 0.0F, 2.0F, 2.0F},
            {4.0F, 1.0F, 5.0F, 0.0F, 2.0F, 2.0F},
        },
        float_options);

    const std::filesystem::path output_path =
        std::filesystem::temp_directory_path() / "placement_cpp_visualization_tests" /
        "nested" / "tiny_placement.png";
    std::filesystem::remove(output_path);

    placement::plotPlacement(initial_cell_features, final_cell_features, output_path);

    expect(std::filesystem::exists(output_path), "visualization output exists");
    expect(std::filesystem::file_size(output_path) > 0, "visualization output is nonempty");

    const std::string content = readFile(output_path);
    expect(content.size() > 8, "visualization output has png header bytes");
    expect(
        static_cast<unsigned char>(content[0]) == 0x89 && content.substr(1, 3) == "PNG",
        "visualization output is a png");
}
