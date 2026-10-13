cmake_minimum_required(VERSION 3.24...3.31)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(tmol_disulfide_shared_cpu LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  option(TMOL_BUILD_TESTS "Build the framework-neutral CPU test" ON)
  option(TMOL_BUILD_CUDA "Build the framework-neutral CUDA library/test" OFF)
endif()

if(TMOL_BUILD_CUDA)
  enable_language(CUDA)
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

# Installation is deliberately deferred until the base ``tmol`` distribution
# is framework-neutral and can be the single owner of this library.  Do not
# copy it into either framework adapter: that would give coexistence installs
# two independently upgradeable copies of the same common ABI.

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND NOT TMOL_OPS_CUDA)
  install(TARGETS tmol_disulfide_shared_cpu LIBRARY DESTINATION tmol)
endif()

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

if(TMOL_BUILD_CUDA)
  add_library(tmol_disulfide_shared_cuda SHARED
    ../shared_cuda.cu
  )
  target_include_directories(tmol_disulfide_shared_cuda
    PUBLIC ${TMOL_SHARED_SOURCE_ROOT}
    PRIVATE
    ${TMOL_SHARED_SOURCE_ROOT}/tmol/extern
  )
  target_compile_definitions(tmol_disulfide_shared_cuda PRIVATE
    TMOL_DISULFIDE_CUDA_BUILD
  )
  target_compile_options(tmol_disulfide_shared_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  set_target_properties(tmol_disulfide_shared_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    POSITION_INDEPENDENT_CODE ON
  )

  if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
     AND TMOL_OPS_CUDA)
    install(TARGETS tmol_disulfide_shared_cuda LIBRARY DESTINATION "${_TMOL_NATIVE_PACKAGE}")
  endif()

  if(TMOL_BUILD_TESTS)
    add_executable(test_disulfide_shared_cuda
      ${TMOL_SHARED_SOURCE_ROOT}/tmol/tests/score/disulfide/test_shared_cuda.cu
    )
    set_target_properties(test_disulfide_shared_cuda PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
    )
    target_include_directories(test_disulfide_shared_cuda PRIVATE
      ${TMOL_SHARED_SOURCE_ROOT}
      ${TMOL_SHARED_SOURCE_ROOT}/tmol/extern
    )
    target_link_libraries(test_disulfide_shared_cuda PRIVATE
      tmol_disulfide_shared_cpu tmol_disulfide_shared_cuda
    )
    add_test(NAME disulfide_shared_cuda COMMAND test_disulfide_shared_cuda)
  endif()
endif()
