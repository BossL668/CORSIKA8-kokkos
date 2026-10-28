#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

# Verify the installed export interface, including a real executable link.
# Header identity complements, but does not replace, the isolated Conan lock:
# consumers must never mix ordinary and patched dependency binaries.
function (corsika_check_proposal_export CHECK_DIRECTORY OUTPUT_DIGEST EXPECTED_DIGEST)
  find_package (PROPOSAL 7.6.2 EXACT CONFIG REQUIRED)
  find_package (CubicInterpolation 0.1.5 EXACT CONFIG REQUIRED)

  set (EXPORT_HEADERS
    PROPOSAL/interpolation/InterpolationTableExport.h
    PROPOSAL/crosssection/CrossSection.h
    PROPOSAL/crosssection/CrossSectionDirect.h
    PROPOSAL/crosssection/CrossSectionMultiplier.h
    PROPOSAL/crosssection/CrossSectionDEDX/CrossSectionDEDXInterpolant.h
    PROPOSAL/crosssection/CrossSectionDNDX/CrossSectionDNDXInterpolant.h
    PROPOSAL/crosssection/parametrization/Bremsstrahlung.h
    PROPOSAL/crosssection/parametrization/PhotoPairProduction.h
    PROPOSAL/propagation_utility/Displacement.h
    PROPOSAL/propagation_utility/DisplacementBuilder.h
    PROPOSAL/propagation_utility/Interaction.h
    PROPOSAL/propagation_utility/InteractionBuilder.h
    PROPOSAL/propagation_utility/PropagationUtilityInterpolant.h
    PROPOSAL/methods.h
    CubicInterpolation/Axis.h
    CubicInterpolation/CubicSplines.h
    CubicInterpolation/BicubicSplines.h
    CubicInterpolation/Interpolant.h
  )
  set (HEADER_IDENTITIES "")
  foreach (EXPORT_HEADER IN LISTS EXPORT_HEADERS)
    # Avoid reusing a cache entry from another header or dependency graph.
    unset (EXPORT_INCLUDE CACHE)
    unset (EXPORT_INCLUDE)
    find_path (EXPORT_INCLUDE "${EXPORT_HEADER}"
      HINTS ${PROPOSAL_INCLUDE_DIRS} ${CubicInterpolation_INCLUDE_DIRS}
      NO_DEFAULT_PATH
    )
    if (NOT EXPORT_INCLUDE)
      message (FATAL_ERROR
        "Missing native export header ${EXPORT_HEADER}. Use the isolated patched "
        "PROPOSAL dependency graph (with_proposal_export=True)."
      )
    endif ()
    file (SHA256 "${EXPORT_INCLUDE}/${EXPORT_HEADER}" HEADER_DIGEST)
    string (APPEND HEADER_IDENTITIES "${EXPORT_HEADER}:${HEADER_DIGEST}\n")
  endforeach ()
  unset (EXPORT_INCLUDE CACHE)
  string (SHA256 HEADER_DIGEST "${HEADER_IDENTITIES}")
  if (NOT "${EXPECTED_DIGEST}" STREQUAL "" AND
      NOT "${HEADER_DIGEST}" STREQUAL "${EXPECTED_DIGEST}")
    message (FATAL_ERROR
      "PROPOSAL/CubicInterpolation export headers differ from those used to build "
      "this CORSIKA installation. Use its matching isolated dependency graph."
    )
  endif ()

  include (CheckCXXSourceCompiles)
  include (CMakePushCheckState)
  cmake_push_check_state (RESET)
  set (CMAKE_REQUIRED_LIBRARIES PROPOSAL::PROPOSAL)
  set (CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  if (NOT CMAKE_TRY_COMPILE_CONFIGURATION AND CMAKE_BUILD_TYPE)
    set (CMAKE_TRY_COMPILE_CONFIGURATION "${CMAKE_BUILD_TYPE}")
  endif ()
  file (READ "${CHECK_DIRECTORY}/CheckProposalExport.cpp" EXPORT_CHECK_SOURCE)
  # A cached success from a previous dependency graph must not bypass the link.
  unset (CORSIKA_HAS_PROPOSAL_EXPORT_API CACHE)
  check_cxx_source_compiles ("${EXPORT_CHECK_SOURCE}" CORSIKA_HAS_PROPOSAL_EXPORT_API)
  cmake_pop_check_state ()
  if (NOT CORSIKA_HAS_PROPOSAL_EXPORT_API)
    message (FATAL_ERROR
      "The native PROPOSAL export API did not compile and link. Do not mix patched "
      "headers with ordinary libraries; rebuild the complete isolated dependency graph."
    )
  endif ()
  set (${OUTPUT_DIGEST} "${HEADER_DIGEST}" PARENT_SCOPE)
endfunction ()
