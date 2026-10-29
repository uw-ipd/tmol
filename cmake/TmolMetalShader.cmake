# Turn a `.metal.in` into a C string literal the module can hand to torch's
# runtime shader compiler.
#
# Metal shaders are normally built offline by `xcrun metal`, part of the Xcode
# Metal toolchain, which is not present on every machine that can run Metal.
# tmol compiles its shaders at runtime instead, which needs the source as a
# string -- and runtime compilation cannot resolve `#include`, so the source
# has to be flattened first. That is what the preprocessor pass below does:
# the same headers the CPU and CUDA builds use are inlined, with
# __METAL_VERSION__ defined so the portable layer selects its Metal spellings.
# The `metal_*` includes resolve to empty stubs here; the real ones are
# supplied by the Metal front end at runtime.
function(tmol_embed_metal_shader IN_FILE OUT_HEADER SYMBOL)
  get_filename_component(_stem "${IN_FILE}" NAME_WE)
  set(_raw "${CMAKE_CURRENT_BINARY_DIR}/${_stem}.raw.metal")
  set(_flat "${CMAKE_CURRENT_BINARY_DIR}/${_stem}.flat.metal")
  get_filename_component(_out_dir "${OUT_HEADER}" DIRECTORY)

  # The flattened output depends on every header the .metal.in pulls in, not
  # just the .metal.in itself. Without the depfile, editing a kernel header
  # leaves a stale shader embedded in the module -- which fails at run time,
  # inside the Metal compiler, rather than at build time.
  set(_dep "${CMAKE_CURRENT_BINARY_DIR}/${_stem}.metal.d")

  add_custom_command(
    OUTPUT "${_flat}"
    COMMAND "${CMAKE_CXX_COMPILER}" -E -x c++ -std=c++17
            -D__METAL_VERSION__=320
            -I "${CMAKE_SOURCE_DIR}/cmake/mslstub"
            -I "${CMAKE_SOURCE_DIR}"
            -I "${CMAKE_SOURCE_DIR}/tmol/extern"
            -MD -MF "${_dep}" -MT "${_flat}"
            "${IN_FILE}" -o "${_raw}"
    COMMAND "${CMAKE_COMMAND}" -DIN=${_raw} -DOUT=${_flat}
            -P "${CMAKE_SOURCE_DIR}/cmake/StripLineMarkers.cmake"
    DEPENDS "${IN_FILE}"
    DEPFILE "${_dep}"
    COMMENT "Flattening ${_stem} for runtime Metal compilation"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${OUT_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_out_dir}"
    COMMAND "${CMAKE_COMMAND}" -DIN=${_flat} -DOUT=${OUT_HEADER}
            -DSYMBOL=${SYMBOL}
            -P "${CMAKE_SOURCE_DIR}/cmake/EmbedMetalSource.cmake"
    DEPENDS "${_flat}"
    COMMENT "Embedding ${_stem} shader source"
    VERBATIM
  )
endfunction()
