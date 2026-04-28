cmake_minimum_required(VERSION 3.25)

project(placement_cpp LANGUAGES CXX)

include(CTest)

option(PLACEMENT_ENABLE_COVERAGE "Enable LLVM source-based coverage for unit tests" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(PLACEMENT_ENABLE_COVERAGE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "PLACEMENT_ENABLE_COVERAGE requires a Clang-compatible compiler")
  endif()

  find_program(LLVM_PROFDATA_EXECUTABLE llvm-profdata)
  if(NOT LLVM_PROFDATA_EXECUTABLE)
    execute_process(
      COMMAND xcrun -f llvm-profdata
      OUTPUT_VARIABLE LLVM_PROFDATA_EXECUTABLE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE LLVM_PROFDATA_RESULT
    )
    if(NOT LLVM_PROFDATA_RESULT EQUAL 0)
      message(FATAL_ERROR "Unable to find llvm-profdata")
    endif()
  endif()

  find_program(LLVM_COV_EXECUTABLE llvm-cov)
  if(NOT LLVM_COV_EXECUTABLE)
    execute_process(
      COMMAND xcrun -f llvm-cov
      OUTPUT_VARIABLE LLVM_COV_EXECUTABLE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE LLVM_COV_RESULT
    )
    if(NOT LLVM_COV_RESULT EQUAL 0)
      message(FATAL_ERROR "Unable to find llvm-cov")
    endif()
  endif()
endif()

function(enable_placement_coverage target_name)
  if(PLACEMENT_ENABLE_COVERAGE)
    target_compile_options(
      ${target_name}
      PRIVATE
        -fprofile-instr-generate
        -fcoverage-mapping
        -O0
        -g
    )
    target_link_options(${target_name} PRIVATE -fprofile-instr-generate)
  endif()
endfunction()

# LibTorch is provided by the active Python environment rather than vcpkg.
# Prefer the repo-local virtualenv when present, then ask Python where the
# installed torch package exposes its CMake config files.
set(_repo_venv_python "${CMAKE_CURRENT_LIST_DIR}/../.venv/bin/python")
if(NOT Python3_EXECUTABLE AND EXISTS "${_repo_venv_python}")
  set(Python3_EXECUTABLE "${_repo_venv_python}" CACHE FILEPATH "Python interpreter")
endif()
if(NOT Python3_EXECUTABLE)
  find_program(_path_python3 NAMES python3 python)
  if(_path_python3)
    execute_process(
      COMMAND "${_path_python3}" -c "import torch"
      RESULT_VARIABLE _path_python3_torch_result
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if(_path_python3_torch_result EQUAL 0)
      set(Python3_EXECUTABLE "${_path_python3}" CACHE FILEPATH "Python interpreter")
    endif()
  endif()
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TORCH_CMAKE_PREFIX_RESULT
)
if(NOT TORCH_CMAKE_PREFIX_RESULT EQUAL 0)
  message(FATAL_ERROR "Unable to import torch from ${Python3_EXECUTABLE}. Install torch in .venv or set Python3_EXECUTABLE.")
endif()
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYTHON_SITE_PACKAGES_RESULT
)
if(NOT PYTHON_SITE_PACKAGES_RESULT EQUAL 0 OR PYTHON_SITE_PACKAGES STREQUAL "")
  message(FATAL_ERROR "Unable to detect Python site-packages from ${Python3_EXECUTABLE}.")
endif()
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import matplotlib"
  RESULT_VARIABLE MATPLOTLIB_IMPORT_RESULT
  OUTPUT_QUIET
  ERROR_QUIET
)
if(NOT MATPLOTLIB_IMPORT_RESULT EQUAL 0)
  message(FATAL_ERROR "Unable to import matplotlib from ${Python3_EXECUTABLE}. Install matplotlib in the selected Python environment.")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

find_package(Torch CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(SQLite3 REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h" REQUIRED)

add_library(
  placement_core
  benchmark.cpp
  generation.cpp
  losses.cpp
  metrics.cpp
  sqlite_utils.cpp
  training.cpp
  visualization.cpp
)
target_include_directories(placement_core PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")
target_include_directories(placement_core SYSTEM PRIVATE "${MATPLOTLIB_CPP_INCLUDE_DIRS}")
target_compile_definitions(
  placement_core
  PRIVATE
    PLACEMENT_PYTHON_SITE_PACKAGES="${PYTHON_SITE_PACKAGES}"
    PLACEMENT_REPO_ROOT="${CMAKE_CURRENT_LIST_DIR}/.."
)
target_link_libraries(
  placement_core
  PUBLIC "${TORCH_LIBRARIES}" Python3::Python
  PRIVATE SQLite::SQLite3
)
target_compile_features(placement_core PUBLIC cxx_std_20)
enable_placement_coverage(placement_core)

add_executable(placement placement.cpp)
target_compile_definitions(
  placement
  PRIVATE
    PLACEMENT_REPO_ROOT="${CMAKE_CURRENT_LIST_DIR}/.."
)
target_link_libraries(placement PRIVATE placement_core CLI11::CLI11)
target_compile_features(placement PRIVATE cxx_std_20)
enable_placement_coverage(placement)

add_executable(placement_test test.cpp)
target_link_libraries(placement_test PRIVATE placement_core CLI11::CLI11)
target_compile_features(placement_test PRIVATE cxx_std_20)
enable_placement_coverage(placement_test)

if(BUILD_TESTING)
  add_executable(
    placement_unit_tests
    tests/metrics_tests.cpp
    tests/visualization_tests.cpp
  )
  target_link_libraries(placement_unit_tests PRIVATE placement_core)
  target_compile_features(placement_unit_tests PRIVATE cxx_std_20)
  enable_placement_coverage(placement_unit_tests)
  add_test(NAME placement_unit_tests COMMAND placement_unit_tests)

  if(PLACEMENT_ENABLE_COVERAGE)
    add_custom_target(
      placement_coverage
      COMMAND
        "${CMAKE_COMMAND}"
        -DTEST_EXECUTABLE=$<TARGET_FILE:placement_unit_tests>
        -DLLVM_PROFDATA="${LLVM_PROFDATA_EXECUTABLE}"
        -DLLVM_COV="${LLVM_COV_EXECUTABLE}"
        -DSOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        -DCOVERAGE_DIR="${CMAKE_BINARY_DIR}/coverage"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunCoverage.cmake"
      DEPENDS placement_unit_tests
      USES_TERMINAL
    )
  endif()
endif()

if(MSVC)
  target_compile_options(placement_core PRIVATE /W4)
  target_compile_options(placement PRIVATE /W4)
  target_compile_options(placement_test PRIVATE /W4)
  if(TARGET placement_unit_tests)
    target_compile_options(placement_unit_tests PRIVATE /W4)
  endif()
else()
  target_compile_options(placement_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(placement PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(placement_test PRIVATE -Wall -Wextra -Wpedantic)
  if(TARGET placement_unit_tests)
    target_compile_options(placement_unit_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()
