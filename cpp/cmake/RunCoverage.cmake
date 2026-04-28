foreach(required_var TEST_EXECUTABLE LLVM_PROFDATA LLVM_COV SOURCE_DIR COVERAGE_DIR)
  if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required coverage variable: ${required_var}")
  endif()
endforeach()

set(raw_profile "${COVERAGE_DIR}/placement_unit_tests.profraw")
set(profile_data "${COVERAGE_DIR}/placement_unit_tests.profdata")
set(report_file "${COVERAGE_DIR}/placement_unit_tests.txt")
set(html_dir "${COVERAGE_DIR}/html")

file(REMOVE_RECURSE "${COVERAGE_DIR}")
file(MAKE_DIRECTORY "${COVERAGE_DIR}")

execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "LLVM_PROFILE_FILE=${raw_profile}"
    "${TEST_EXECUTABLE}"
  RESULT_VARIABLE test_result
)
if(NOT test_result EQUAL 0)
  message(FATAL_ERROR "placement_unit_tests failed during coverage run")
endif()

execute_process(
  COMMAND "${LLVM_PROFDATA}" merge -sparse "${raw_profile}" -o "${profile_data}"
  RESULT_VARIABLE profdata_result
)
if(NOT profdata_result EQUAL 0)
  message(FATAL_ERROR "llvm-profdata failed")
endif()

set(covered_sources
  "${SOURCE_DIR}/benchmark.cpp"
  "${SOURCE_DIR}/generation.cpp"
  "${SOURCE_DIR}/losses.cpp"
  "${SOURCE_DIR}/metrics.cpp"
  "${SOURCE_DIR}/training.cpp"
  "${SOURCE_DIR}/visualization.cpp"
)

execute_process(
  COMMAND
    "${LLVM_COV}" report
    "${TEST_EXECUTABLE}"
    "-instr-profile=${profile_data}"
    ${covered_sources}
  OUTPUT_VARIABLE coverage_report
  RESULT_VARIABLE report_result
)
if(NOT report_result EQUAL 0)
  message(FATAL_ERROR "llvm-cov report failed")
endif()
file(WRITE "${report_file}" "${coverage_report}")

execute_process(
  COMMAND
    "${LLVM_COV}" show
    "${TEST_EXECUTABLE}"
    "-instr-profile=${profile_data}"
    "-format=html"
    "-output-dir=${html_dir}"
    ${covered_sources}
  RESULT_VARIABLE html_result
)
if(NOT html_result EQUAL 0)
  message(FATAL_ERROR "llvm-cov html report failed")
endif()

message(STATUS "Coverage text report: ${report_file}")
message(STATUS "Coverage HTML report: ${html_dir}/index.html")
message("${coverage_report}")
