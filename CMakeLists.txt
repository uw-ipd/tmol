cmake_minimum_required(VERSION 3.18)

option(TMOL_ENABLE_CUDA "Build CUDA extensions (disable for CPU-only)" ON)

# ═══════════════════════════════════════════════════════════════════════════════
# Detect CUDA and choose languages
# ═══════════════════════════════════════════════════════════════════════════════

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER AND TMOL_ENABLE_CUDA)
  # ── CUDA architecture selection — MUST be set before project() so that
  # enable_language(CUDA) uses our architectures, not the nvcc defaults.
  # CMAKE_CUDA_ARCHITECTURES can be set via:
  #   -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"  (from scikit-build-core -C flags)
  #   TORCH_CUDA_ARCH_LIST env var (PyTorch format: "7.0 8.0 9.0+PTX")
  #   or falls back to a sensible default.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
      set(_ARCH_STR "$ENV{TORCH_CUDA_ARCH_LIST}")
      string(REPLACE " " ";" _ARCH_LIST "${_ARCH_STR}")
      set(_CLEAN_ARCHS "")
      foreach(_A ${_ARCH_LIST})
        string(FIND "${_A}" "+PTX" _HAS_PTX)
        string(REPLACE "+PTX" "" _A "${_A}")
        string(REPLACE "." "" _A "${_A}")
        if(_HAS_PTX GREATER_EQUAL 0)
          list(APPEND _CLEAN_ARCHS "${_A}")
        else()
          list(APPEND _CLEAN_ARCHS "${_A}-real")
        endif()
      endforeach()
      set(CMAKE_CUDA_ARCHITECTURES ${_CLEAN_ARCHS} CACHE STRING "CUDA architectures")
    else()
      set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures")
    endif()
  endif()

  project(tmol LANGUAGES CXX CUDA)
  set(TMOL_HAS_CUDA TRUE)
  message(STATUS "tmol: CUDA enabled, CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
else()
  project(tmol LANGUAGES CXX)
  set(TMOL_HAS_CUDA FALSE)
  message(STATUS "tmol: CUDA not found or disabled; building CPU-only extensions")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# Dependencies
# ═══════════════════════════════════════════════════════════════════════════════

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Save Python_INCLUDE_DIRS NOW — find_package(Torch) / Caffe2 may overwrite it.
set(_TMOL_PYTHON_INCLUDE_DIRS "${Python_INCLUDE_DIRS}")
if(NOT _TMOL_PYTHON_INCLUDE_DIRS)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE _TMOL_PYTHON_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()
message(STATUS "tmol: Python include dirs = ${_TMOL_PYTHON_INCLUDE_DIRS}")

if(TMOL_HAS_CUDA)
  # Save CMAKE_CUDA_ARCHITECTURES before find_package(Torch) — Caffe2 sets it to OFF.
  get_property(_TMOL_CUDA_ARCHITECTURES CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY VALUE)
  message(STATUS "tmol: _TMOL_CUDA_ARCHITECTURES (from cache) = ${_TMOL_CUDA_ARCHITECTURES}")

  # Set TORCH_CUDA_ARCH_LIST env var BEFORE find_package(Torch) so that
  # Caffe2's cuda.cmake uses our architectures instead of defaulting to
  # all archs nvcc supports (which includes sm_52 that lacks double atomicAdd).
  set(_TORCH_ARCH_STR "")
  foreach(_A ${CMAKE_CUDA_ARCHITECTURES})
    string(REPLACE "-real" "" _A "${_A}")
    string(LENGTH "${_A}" _ALEN)
    if(_ALEN EQUAL 3)
      string(SUBSTRING "${_A}" 0 2 _MAJOR)
      string(SUBSTRING "${_A}" 2 -1 _MINOR)
      set(_TORCH_ARCH_STR "${_TORCH_ARCH_STR} ${_MAJOR}.${_MINOR}")
    elseif(_ALEN EQUAL 2)
      string(SUBSTRING "${_A}" 0 1 _MAJOR)
      string(SUBSTRING "${_A}" 1 -1 _MINOR)
      set(_TORCH_ARCH_STR "${_TORCH_ARCH_STR} ${_MAJOR}.${_MINOR}")
    endif()
  endforeach()
  string(STRIP "${_TORCH_ARCH_STR}" _TORCH_ARCH_STR)
  set(ENV{TORCH_CUDA_ARCH_LIST} "${_TORCH_ARCH_STR}")
  message(STATUS "tmol: Setting TORCH_CUDA_ARCH_LIST=${_TORCH_ARCH_STR} for find_package(Torch)")
endif()

find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
message(STATUS "tmol: TORCH_PYTHON_LIBRARY = ${TORCH_PYTHON_LIBRARY}")

if(TMOL_HAS_CUDA)
  # Caffe2's TorchConfig.cmake unconditionally appends the -gencode flags that
  # PyTorch was originally built with. Replace them with our architectures.
  string(REGEX REPLACE "-gencode[;= ]arch=compute_[0-9]+,code=sm_[0-9]+" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  string(REGEX REPLACE "-gencode[;= ]arch=compute_[0-9]+,code=compute_[0-9]+" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  string(REGEX REPLACE " +" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)

  set(_TMOL_GENCODE_FLAGS "")
  foreach(_A ${_TMOL_CUDA_ARCHITECTURES})
    string(REPLACE "-real" "" _A_CLEAN "${_A}")
    if("${_A}" STREQUAL "${_A_CLEAN}")
      set(_TMOL_GENCODE_FLAGS "${_TMOL_GENCODE_FLAGS} -gencode arch=compute_${_A_CLEAN},code=sm_${_A_CLEAN} -gencode arch=compute_${_A_CLEAN},code=compute_${_A_CLEAN}")
    else()
      set(_TMOL_GENCODE_FLAGS "${_TMOL_GENCODE_FLAGS} -gencode arch=compute_${_A_CLEAN},code=sm_${_A_CLEAN}")
    endif()
  endforeach()
  string(STRIP "${_TMOL_GENCODE_FLAGS}" _TMOL_GENCODE_FLAGS)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${_TMOL_GENCODE_FLAGS}")
  message(STATUS "tmol: CMAKE_CUDA_FLAGS after fixing gencode = ${CMAKE_CUDA_FLAGS}")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# Compiler flags — match PyTorch's ABI and set tmol defaults
# ═══════════════════════════════════════════════════════════════════════════════

# TORCH_CXX_FLAGS contains -D_GLIBCXX_USE_CXX11_ABI=0 or =1
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(TMOL_HAS_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Get torch version for compile-time macros
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import torch; v=torch.__version__.split('.'); print(f'{v[0]};{v[1]}')"
  OUTPUT_VARIABLE _TORCH_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(GET _TORCH_VERSION 0 TORCH_VERSION_MAJOR)
list(GET _TORCH_VERSION 1 TORCH_VERSION_MINOR)

if(TMOL_HAS_CUDA)
  set(TMOL_NVCC_THREADS "4" CACHE STRING "Threads per nvcc invocation")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# Include directories
# ═══════════════════════════════════════════════════════════════════════════════

set(TMOL_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/tmol/extern
  ${TORCH_INCLUDE_DIRS}
  ${_TMOL_PYTHON_INCLUDE_DIRS}
)

# NVTX includes (from pip nvidia-nvtx package or CUDA toolkit)
if(TMOL_HAS_CUDA)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c
      "import nvidia.nvtx, os; print(os.path.join(nvidia.nvtx.__path__[0], 'include'))"
    OUTPUT_VARIABLE _NVTX_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _NVTX_RESULT
  )
  if(_NVTX_RESULT EQUAL 0 AND EXISTS "${_NVTX_INCLUDE}")
    list(APPEND TMOL_INCLUDE_DIRS "${_NVTX_INCLUDE}")
  else()
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      list(APPEND TMOL_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
    endif()
  endif()
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

# Common compile options for CUDA extensions
function(tmol_set_cuda_flags TARGET)
  target_compile_options(${TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -w -DWITH_CUDA -DWITH_NVTX>
    $<$<COMPILE_LANGUAGE:CUDA>:
      -O3 -w
      --expt-extended-lambda
      --expt-relaxed-constexpr
      -DWITH_NVTX -DWITH_CUDA
      --threads=${TMOL_NVCC_THREADS}
      -DTORCH_VERSION_MAJOR=${TORCH_VERSION_MAJOR}
      -DTORCH_VERSION_MINOR=${TORCH_VERSION_MINOR}
    >
  )
  target_include_directories(${TARGET} PRIVATE ${TMOL_INCLUDE_DIRS})
  target_link_libraries(${TARGET} PRIVATE ${TORCH_LIBRARIES})
endfunction()

# Common compile options for CPU-only extensions
function(tmol_set_cpp_flags TARGET)
  target_compile_options(${TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -w -DWITH_NVTX>
  )
  target_include_directories(${TARGET} PRIVATE ${TMOL_INCLUDE_DIRS})
  target_link_libraries(${TARGET} PRIVATE ${TORCH_LIBRARIES})
endfunction()

# Create a pybind11 CUDA extension and install it
function(tmol_add_pybind_cuda_ext TARGET_NAME INSTALL_DIR EXT_NAME)
  pybind11_add_module(${TARGET_NAME} MODULE ${ARGN})
  target_compile_definitions(${TARGET_NAME} PRIVATE TORCH_EXTENSION_NAME=${EXT_NAME})
  tmol_set_cuda_flags(${TARGET_NAME})
  target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY})
  set_target_properties(${TARGET_NAME} PROPERTIES
    OUTPUT_NAME "${EXT_NAME}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${INSTALL_DIR}"
  )
  install(TARGETS ${TARGET_NAME} DESTINATION "${INSTALL_DIR}")
endfunction()

# Create a pybind11 CPU-only extension and install it
function(tmol_add_pybind_cpp_ext TARGET_NAME INSTALL_DIR EXT_NAME)
  pybind11_add_module(${TARGET_NAME} MODULE ${ARGN})
  target_compile_definitions(${TARGET_NAME} PRIVATE TORCH_EXTENSION_NAME=${EXT_NAME})
  tmol_set_cpp_flags(${TARGET_NAME})
  target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY})
  set_target_properties(${TARGET_NAME} PROPERTIES
    OUTPUT_NAME "${EXT_NAME}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${INSTALL_DIR}"
  )
  install(TARGETS ${TARGET_NAME} DESTINATION "${INSTALL_DIR}")
endfunction()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. tmol._C — monolithic TORCH_LIBRARY extension (14 op namespaces)
# ═══════════════════════════════════════════════════════════════════════════════

set(_C_SOURCES
  # io/details/compiled
  tmol/io/details/compiled/compiled.ops.cpp
  tmol/io/details/compiled/gen_pose_leaf_atoms.cpu.cpp
  tmol/io/details/compiled/resolve_his_taut.cpu.cpp
  # kinematics/compiled
  tmol/kinematics/compiled/compiled_ops.cpp
  tmol/kinematics/compiled/compiled.cpu.cpp
  # optimization/compiled
  tmol/optimization/compiled/lbfgs.ops.cpp
  tmol/optimization/compiled/lbfgs.cpu.cpp
  # pack/compiled
  tmol/pack/compiled/compiled.ops.cpp
  tmol/pack/compiled/compiled.cpu.cpp
  # pack/rotamer/dunbrack
  tmol/pack/rotamer/dunbrack/compiled.ops.cpp
  tmol/pack/rotamer/dunbrack/compiled.cpu.cpp
  # pose/compiled/apsp
  tmol/pose/compiled/apsp_vestibule.ops.cpp
  tmol/pose/compiled/apsp.cpu.cpp
  # score/backbone_torsion
  tmol/score/backbone_torsion/potentials/compiled.ops.cpp
  tmol/score/backbone_torsion/potentials/backbone_torsion_pose_score.cpu.cpp
  # score/cartbonded
  tmol/score/cartbonded/potentials/compiled.ops.cpp
  tmol/score/cartbonded/potentials/cartbonded_pose_score.cpu.cpp
  # score/constraint
  tmol/score/constraint/potentials/compiled.ops.cpp
  tmol/score/constraint/potentials/constraint_score.cpu.cpp
  # score/disulfide
  tmol/score/disulfide/potentials/compiled.ops.cpp
  tmol/score/disulfide/potentials/disulfide_pose_score.cpu.cpp
  # score/dunbrack
  tmol/score/dunbrack/potentials/compiled.ops.cpp
  tmol/score/dunbrack/potentials/dunbrack_pose_score.cpu.cpp
  # score/elec
  tmol/score/elec/potentials/compiled.ops.cpp
  tmol/score/elec/potentials/elec_pose_score.cpu.cpp
  # score/hbond
  tmol/score/hbond/potentials/compiled.ops.cpp
  tmol/score/hbond/potentials/hbond_pose_score.cpu.cpp
  # score/ljlk
  tmol/score/ljlk/potentials/compiled.ops.cpp
  tmol/score/ljlk/potentials/ljlk_pose_score.cpu.cpp
  # score/lk_ball
  tmol/score/lk_ball/potentials/compiled.ops.cpp
  tmol/score/lk_ball/potentials/lk_ball_pose_score.cpu.cpp
  tmol/score/lk_ball/potentials/gen_pose_waters.cpu.cpp
)

if(TMOL_HAS_CUDA)
  list(APPEND _C_SOURCES
    tmol/io/details/compiled/gen_pose_leaf_atoms.cuda.cu
    tmol/io/details/compiled/resolve_his_taut.cuda.cu
    tmol/kinematics/compiled/compiled.cuda.cu
    tmol/optimization/compiled/lbfgs.cuda.cu
    tmol/pack/compiled/compiled.cuda.cu
    tmol/pack/rotamer/dunbrack/compiled.cuda.cu
    tmol/pose/compiled/apsp.cuda.cu
    tmol/score/backbone_torsion/potentials/backbone_torsion_pose_score.cuda.cu
    tmol/score/cartbonded/potentials/cartbonded_pose_score.cuda.cu
    tmol/score/constraint/potentials/constraint_score.cuda.cu
    tmol/score/disulfide/potentials/disulfide_pose_score.cuda.cu
    tmol/score/dunbrack/potentials/dunbrack_pose_score.cuda.cu
    tmol/score/elec/potentials/elec_pose_score.cuda.cu
    tmol/score/hbond/potentials/hbond_pose_score.cuda.cu
    tmol/score/ljlk/potentials/ljlk_pose_score.cuda.cu
    tmol/score/lk_ball/potentials/lk_ball_pose_score.cuda.cu
    tmol/score/lk_ball/potentials/gen_pose_waters.cuda.cu
  )
endif()

add_library(_C MODULE ${_C_SOURCES})

# _C is a MODULE library loaded by torch, not a regular Python extension.
# No pybind11 entry point — ops register via TORCH_LIBRARY at dlopen time.
set_target_properties(_C PROPERTIES
  PREFIX ""
  OUTPUT_NAME "_C"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tmol"
)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
  OUTPUT_VARIABLE _PY_EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set_target_properties(_C PROPERTIES SUFFIX "${_PY_EXT_SUFFIX}")

if(TMOL_HAS_CUDA)
  tmol_set_cuda_flags(_C)
else()
  tmol_set_cpp_flags(_C)
endif()
install(TARGETS _C DESTINATION tmol)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Production pybind11 modules
# ═══════════════════════════════════════════════════════════════════════════════

# bspline (CPU-only)
tmol_add_pybind_cpp_ext(
  bspline_compiled "tmol/numeric/bspline_compiled" "_compiled"
  tmol/numeric/bspline_compiled/bspline.pybind.cpp
)

# inverse kinematics (requires CUDA)
if(TMOL_HAS_CUDA)
  tmol_add_pybind_cuda_ext(
    compiled_inverse_kin "tmol/kinematics/compiled" "_compiled_inverse_kin"
    tmol/kinematics/compiled/compiled_inverse_kin.cpp
    tmol/kinematics/compiled/compiled.cpu.cpp
    tmol/kinematics/compiled/compiled.cuda.cu
  )
endif()

# cubic hermite polynomial (CPU-only)
tmol_add_pybind_cpp_ext(
  cubic_hermite "tmol/score/common" "_cubic_hermite_polynomial"
  tmol/score/common/_cubic_hermite_polynomial.cpp
)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Test extensions (optional, behind TMOL_BUILD_TESTS)
# ═══════════════════════════════════════════════════════════════════════════════

option(TMOL_BUILD_TESTS "Build test C++/CUDA extensions" OFF)

if(TMOL_BUILD_TESTS)

  # --- pybind CPU-only test extensions ---

  tmol_add_pybind_cpp_ext(
    test_hbond_ext "tmol/tests/score/hbond/potentials" "_ext"
    tmol/tests/score/hbond/potentials/compiled.pybind.cpp
  )

  tmol_add_pybind_cpp_ext(
    test_ljlk_ext "tmol/tests/score/ljlk/potentials" "_ext"
    tmol/tests/score/ljlk/potentials/compiled.pybind.cpp
  )

  tmol_add_pybind_cpp_ext(
    test_lkball_ext "tmol/tests/score/lk_ball/potentials" "_ext"
    tmol/tests/score/lk_ball/potentials/compiled.pybind.cpp
  )

  tmol_add_pybind_cpp_ext(
    test_polynomial_ext "tmol/tests/score/common/polynomial" "_ext"
    tmol/tests/score/common/polynomial/polynomial.pybind.cpp
  )

  tmol_add_pybind_cpp_ext(
    test_geom_ext "tmol/tests/score/common/geom" "_ext"
    tmol/tests/score/common/geom/geom.pybind.cpp
  )

  tmol_add_pybind_cpp_ext(
    test_uaid_util "tmol/tests/score/common" "_uaid_util"
    tmol/tests/score/common/uaid_util.pybind.cc
  )

  tmol_add_pybind_cpp_ext(
    test_tensor_struct "tmol/tests/utility/tensor" "_tensor_struct"
    tmol/tests/utility/tensor/tensor_struct.cpp
  )

  tmol_add_pybind_cpp_ext(
    test_tensor_accessor "tmol/tests/utility/tensor" "_tensor_accessor"
    tmol/tests/utility/tensor/tensor_accessor.cpp
  )

  tmol_add_pybind_cpp_ext(
    test_tensor_collection "tmol/tests/utility/tensor" "_tensor_collection"
    tmol/tests/utility/tensor/tensor_collection.cpp
  )

  tmol_add_pybind_cpp_ext(
    test_in_place_heap "tmol/tests/utility/datastructures" "_in_place_heap"
    tmol/tests/utility/datastructures/in_place_heap.cpp
  )

  # --- pybind CUDA test extensions (skipped in CPU-only builds) ---

  if(TMOL_HAS_CUDA)
    tmol_add_pybind_cuda_ext(
      test_geom_cuda "tmol/tests/score/common/geom" "_ext_cuda"
      tmol/tests/score/common/geom/geom.cu
    )

    tmol_add_pybind_cuda_ext(
      test_bonded_atom "tmol/tests/score/bonded_atom" "_ext"
      tmol/tests/score/bonded_atom/test.pybind.cpp
      tmol/tests/score/bonded_atom/test_cpu.cpp
      tmol/tests/score/bonded_atom/test_cuda.cu
    )

    tmol_add_pybind_cuda_ext(
      test_dispatch "tmol/tests/score/common/dispatch" "_ext"
      tmol/tests/score/common/dispatch/test.pybind.cpp
      tmol/tests/score/common/dispatch/test_cpu.cpp
      tmol/tests/score/common/dispatch/test_cuda.cu
    )

    tmol_add_pybind_cuda_ext(
      test_warp_segreduce "tmol/tests/score/common" "_warp_segreduce"
      tmol/tests/score/common/warp_segreduce.cpp
      tmol/tests/score/common/warp_segreduce.cuda.cu
    )

    tmol_add_pybind_cuda_ext(
      test_warp_stride_reduce "tmol/tests/score/common" "_warp_stride_reduce"
      tmol/tests/score/common/warp_stride_reduce.cpp
      tmol/tests/score/common/warp_stride_reduce.cuda.cu
    )

    tmol_add_pybind_cuda_ext(
      test_segscan "tmol/tests/kinematics/segscan" "_ext"
      tmol/tests/kinematics/segscan/segscan.cu
    )

    tmol_add_pybind_cuda_ext(
      test_dunbrack_ext "tmol/tests/pack/rotamer/dunbrack" "_ext"
      tmol/tests/pack/rotamer/dunbrack/compiled.pybind.cpp
      tmol/tests/pack/rotamer/dunbrack/test_cpu.cpp
      tmol/tests/pack/rotamer/dunbrack/test_cuda.cu
    )
  endif()

endif()
