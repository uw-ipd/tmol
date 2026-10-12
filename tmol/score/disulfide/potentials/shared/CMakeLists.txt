cmake_minimum_required(VERSION 3.24...3.31)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(tmol_disulfide_shared_cpu LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  option(TMOL_BUILD_TESTS "Build the framework-neutral CPU test" ON)
endif()

set(TMOL_SHARED_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../..")

add_library(tmol_disulfide_shared_cpu SHARED
  ../shared_cpu.cpp
)
target_include_directories(tmol_disulfide_shared_cpu
  PUBLIC ${TMOL_SHARED_SOURCE_ROOT}
  PRIVATE
  ${TMOL_SHARED_SOURCE_ROOT}/tmol/extern
)
target_compile_options(tmol_disulfide_shared_cpu PRIVATE
  -O3 -fno-math-errno
)
target_compile_definitions(tmol_disulfide_shared_cpu PRIVATE
  TMOL_DISULFIDE_CPU_BUILD
)
set_target_properties(tmol_disulfide_shared_cpu PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  POSITION_INDEPENDENT_CODE ON
)

# Installation is deliberately deferred until the framework-adapter/package
# owner selects one wheel owner for this library and its transitive headers.

if(TMOL_BUILD_TESTS)
  enable_testing()
  add_executable(test_disulfide_shared_cpu
    ${TMOL_SHARED_SOURCE_ROOT}/tmol/tests/score/disulfide/test_shared_cpu.cpp
  )
  target_include_directories(test_disulfide_shared_cpu PRIVATE
    ${TMOL_SHARED_SOURCE_ROOT}
    ${TMOL_SHARED_SOURCE_ROOT}/tmol/extern
  )
  target_link_libraries(test_disulfide_shared_cpu PRIVATE
    tmol_disulfide_shared_cpu
  )
  add_test(NAME disulfide_shared_cpu COMMAND test_disulfide_shared_cpu)
endif()
