if (NOT DEFINED C8_SOURCE_DIR)
  message (FATAL_ERROR "C8_SOURCE_DIR is required")
endif ()

set (
  generic_runner_sources
  "${C8_SOURCE_DIR}/corsika/gpu/em/detail/CudaEmRunSession.hpp"
  "${C8_SOURCE_DIR}/corsika/gpu/em/detail/CudaHybridCascadeRunner.hpp"
  "${C8_SOURCE_DIR}/src/gpu/em/CudaEmRunSession.cpp"
  )

set (
  forbidden_direct_include_pattern
  "#[ \t]*include[^\n]*(CLI|yaml|applications|air_shower|ice_cascade|CoREAS|ZHS|FLUKA|fluka)"
  )

foreach (source IN LISTS generic_runner_sources)
  if (NOT EXISTS "${source}")
    message (FATAL_ERROR "missing generic CUDA runner source: ${source}")
  endif ()
  file (READ "${source}" contents)
  if (contents MATCHES "${forbidden_direct_include_pattern}")
    message (
      FATAL_ERROR
      "generic CUDA runner has an application-specific direct include: ${source}"
      )
  endif ()
  if (contents MATCHES "(CLI|YAML)::")
    message (
      FATAL_ERROR
      "generic CUDA runner uses an application-layer API: ${source}"
      )
  endif ()
endforeach ()

message (STATUS "generic CUDA runner architecture boundary is clean")
