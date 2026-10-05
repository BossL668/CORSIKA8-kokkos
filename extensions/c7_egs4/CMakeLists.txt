cmake_minimum_required(VERSION 3.18)
project(c7_egs4_cpp_prototype LANGUAGES CXX)
find_package(Kokkos CONFIG REQUIRED)
option(EGS4_CPP_CUDA "Link CUDA runtime/driver for a CUDA-enabled Kokkos package" OFF)
add_library(c7_egs4_tables Egs4Tables.cpp)
target_compile_features(c7_egs4_tables PUBLIC cxx_std_17)
target_link_libraries(c7_egs4_tables PUBLIC Kokkos::kokkos)
if(EGS4_CPP_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(c7_egs4_tables PUBLIC CUDA::cudart CUDA::cuda_driver)
endif()
target_include_directories(c7_egs4_tables PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(test_egs4_tables test_tables.cpp)
target_link_libraries(test_egs4_tables PRIVATE c7_egs4_tables)
# Frozen C++ coefficient fixture ONLY, not a runtime model or a Fortran library.
get_filename_component(_repo "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(EGS4_C8_SOURCE "${_repo}" CACHE PATH "C8 source tree")
set(EGS4_TABLE "${_repo}/resources/c7_egs4/EGSDAT6_.4" CACHE FILEPATH
  "EGSDAT6_.4 table (defaults to project resources)")
set(EGS4_C7_SOURCE "" CACHE FILEPATH "User-supplied corsika.F, reference tests only")
set(EGS4_C7_SCATTER_REFERENCE "" CACHE FILEPATH "Audited external C7 scattering oracle")
set(EGS4_STEP_REFERENCE_FILE "" CACHE FILEPATH "Optional external frozen C7StepTables.hpp fixture")
if(EGS4_STEP_REFERENCE_FILE)
  if(NOT EXISTS "${EGS4_STEP_REFERENCE_FILE}")
    message(FATAL_ERROR "EGS4_STEP_REFERENCE_FILE does not exist")
  endif()
  target_compile_definitions(test_egs4_tables PRIVATE EGS4_STEP_REFERENCE_FILE="${EGS4_STEP_REFERENCE_FILE}")
else()
  message(STATUS "Frozen C7 coefficient comparison disabled: external fixture not supplied")
endif()
if(NOT EXISTS "${EGS4_TABLE}")
  message(FATAL_ERROR "EGS4 table missing: ${EGS4_TABLE}; restore project resources or override EGS4_TABLE")
endif()
target_include_directories(test_egs4_tables PRIVATE
  "${EGS4_C8_SOURCE}")
enable_testing()
add_executable(test_egs4_memory_budget test_memory_budget.cpp)
target_compile_features(test_egs4_memory_budget PRIVATE cxx_std_17)
add_test(NAME native_cpp_c7_memory_budget COMMAND test_egs4_memory_budget)
add_executable(test_egs4_proposal_muon_handoff test_proposal_muon_handoff.cpp)
target_compile_features(test_egs4_proposal_muon_handoff PRIVATE cxx_std_17)
target_include_directories(test_egs4_proposal_muon_handoff PRIVATE
  "${EGS4_C8_SOURCE}")
add_test(NAME native_cpp_proposal_muon_handoff COMMAND test_egs4_proposal_muon_handoff)
add_executable(test_muon_ionization_precision test_muon_ionization_precision.cpp)
target_link_libraries(test_muon_ionization_precision PRIVATE c7_egs4_tables)
target_include_directories(test_muon_ionization_precision PRIVATE
  "${EGS4_C8_SOURCE}")
if(EGS4_CPP_CUDA)
  target_compile_options(test_muon_ionization_precision PRIVATE --extended-lambda)
endif()
add_test(NAME native_cpp_muon_ionization_precision COMMAND test_muon_ionization_precision)
set_tests_properties(native_cpp_muon_ionization_precision PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_test(NAME egs4_native_cpp_tables COMMAND test_egs4_tables
  "${EGS4_TABLE}")
set_tests_properties(egs4_native_cpp_tables PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_test(NAME reject_missing_egs4_table COMMAND test_egs4_tables
  "${CMAKE_CURRENT_BINARY_DIR}/nonexistent_EGSDAT_file")
set_tests_properties(reject_missing_egs4_table PROPERTIES WILL_FAIL TRUE
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# This executable is always native C++ and links only Kokkos/the table reader.
add_executable(test_egs4_interactions test_interactions.cpp)
target_link_libraries(test_egs4_interactions PRIVATE c7_egs4_tables)
add_executable(test_egs4_radiative test_radiative.cpp)
target_link_libraries(test_egs4_radiative PRIVATE c7_egs4_tables)
add_executable(test_egs4_competition test_competition.cpp)
target_link_libraries(test_egs4_competition PRIVATE c7_egs4_tables)
add_test(NAME native_cpp_c7_competition COMMAND test_egs4_competition)
set_tests_properties(native_cpp_c7_competition PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_scattering test_scattering.cpp)
target_link_libraries(test_egs4_scattering PRIVATE c7_egs4_tables)
add_executable(test_egs4_termination test_termination.cpp)
target_link_libraries(test_egs4_termination PRIVATE c7_egs4_tables)
add_executable(test_egs4_rare test_rare.cpp)
target_link_libraries(test_egs4_rare PRIVATE c7_egs4_tables)
add_executable(test_egs4_photonuclear test_photonuclear.cpp)
target_link_libraries(test_egs4_photonuclear PRIVATE c7_egs4_tables)
add_executable(test_egs4_nuclear_target test_nuclear_target.cpp)
target_link_libraries(test_egs4_nuclear_target PRIVATE c7_egs4_tables)
add_executable(test_egs4_meson test_meson.cpp)
target_link_libraries(test_egs4_meson PRIVATE c7_egs4_tables)
add_test(NAME native_cpp_c7_meson_contracts COMMAND test_egs4_meson --contracts)
set_tests_properties(native_cpp_c7_meson_contracts PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_two_body test_two_body.cpp)
target_link_libraries(test_egs4_two_body PRIVATE c7_egs4_tables)
add_test(NAME native_cpp_c7_two_body_contracts COMMAND test_egs4_two_body --contracts)
set_tests_properties(native_cpp_c7_two_body_contracts PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_resonance test_resonance.cpp)
target_link_libraries(test_egs4_resonance PRIVATE c7_egs4_tables)
add_test(NAME native_cpp_c7_resonance_contracts COMMAND test_egs4_resonance --contracts)
set_tests_properties(native_cpp_c7_resonance_contracts PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_test(NAME native_cpp_c7_photonuclear_contracts COMMAND test_egs4_photonuclear --contracts)
set_tests_properties(native_cpp_c7_photonuclear_contracts PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_test(NAME native_cpp_c7_rare_contracts COMMAND test_egs4_rare --contracts)
set_tests_properties(native_cpp_c7_rare_contracts PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_model test_model.cpp)
target_link_libraries(test_egs4_model PRIVATE c7_egs4_tables)
add_test(NAME native_cpp_c7_model_dispatch COMMAND test_egs4_model
  "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_model_dispatch PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_transport test_transport.cpp)
target_link_libraries(test_egs4_transport PRIVATE c7_egs4_tables)
add_test(NAME native_cpp_c7_transport_contracts COMMAND test_egs4_transport --contracts
  "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_transport_contracts PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_photon_transport test_photon_transport.cpp)
target_link_libraries(test_egs4_photon_transport PRIVATE c7_egs4_tables)
add_test(NAME native_cpp_c7_photon_transport_contracts COMMAND test_egs4_photon_transport --contracts
  "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_photon_transport_contracts PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_history test_history.cpp)
target_link_libraries(test_egs4_history PRIVATE c7_egs4_tables)
add_test(NAME native_cpp_c7_history COMMAND test_egs4_history
  "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_history PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_c8_adapter test_c8_adapter.cpp)
target_link_libraries(test_egs4_c8_adapter PRIVATE c7_egs4_tables)
target_include_directories(test_egs4_c8_adapter PRIVATE
  "${EGS4_C8_SOURCE}")
add_test(NAME native_cpp_c7_c8_adapter COMMAND test_egs4_c8_adapter
  "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_c8_adapter PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_wavefront test_wavefront.cpp)
target_link_libraries(test_egs4_wavefront PRIVATE c7_egs4_tables)
target_include_directories(test_egs4_wavefront PRIVATE
  "${EGS4_C8_SOURCE}")
add_test(NAME native_cpp_c7_wavefront COMMAND test_egs4_wavefront
  "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_wavefront PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_curved test_curved.cpp)
target_link_libraries(test_egs4_curved PRIVATE c7_egs4_tables)
add_executable(test_egs4_geometry test_geometry.cpp)
target_link_libraries(test_egs4_geometry PRIVATE c7_egs4_tables)
add_executable(test_egs4_observation test_observation.cpp)
target_link_libraries(test_egs4_observation PRIVATE c7_egs4_tables)
add_executable(test_egs4_curved_history test_curved_history.cpp)
target_link_libraries(test_egs4_curved_history PRIVATE c7_egs4_tables)
add_test(NAME native_cpp_c7_curved_history COMMAND test_egs4_curved_history
  "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_curved_history PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_detector_history test_detector_history.cpp)
target_link_libraries(test_egs4_detector_history PRIVATE c7_egs4_tables)
add_test(NAME native_cpp_c7_detector_history COMMAND test_egs4_detector_history
  "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_detector_history PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_c8_air test_c8_air.cpp)
target_link_libraries(test_egs4_c8_air PRIVATE c7_egs4_tables)
target_include_directories(test_egs4_c8_air PRIVATE
  "${EGS4_C8_SOURCE}")
add_test(NAME native_cpp_c7_c8_air COMMAND test_egs4_c8_air
  "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_c8_air PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_air_wavefront test_air_wavefront.cpp)
target_link_libraries(test_egs4_air_wavefront PRIVATE c7_egs4_tables)
target_include_directories(test_egs4_air_wavefront PRIVATE
  "${EGS4_C8_SOURCE}")
add_test(NAME native_cpp_c7_air_wavefront COMMAND test_egs4_air_wavefront
  "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_air_wavefront PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_thinning test_thinning.cpp)
target_link_libraries(test_egs4_thinning PRIVATE c7_egs4_tables)
target_include_directories(test_egs4_thinning PRIVATE "${EGS4_C8_SOURCE}")
add_test(NAME native_cpp_c7_thinning COMMAND test_egs4_thinning "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_thinning PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_c8_rare test_c8_rare.cpp)
target_link_libraries(test_egs4_c8_rare PRIVATE c7_egs4_tables)
target_include_directories(test_egs4_c8_rare PRIVATE
  "${EGS4_C8_SOURCE}")
add_test(NAME native_cpp_c7_c8_rare_transfer COMMAND test_egs4_c8_rare
  "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_c8_rare_transfer PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(test_egs4_c8_resonance test_c8_resonance.cpp)
target_link_libraries(test_egs4_c8_resonance PRIVATE c7_egs4_tables)
target_include_directories(test_egs4_c8_resonance PRIVATE
  "${EGS4_C8_SOURCE}")
add_test(NAME native_cpp_c7_c8_resonance_transfer COMMAND test_egs4_c8_resonance
  "${EGS4_TABLE}")
set_tests_properties(native_cpp_c7_c8_resonance_transfer PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
add_executable(bench_egs4_native bench_native.cpp)
target_link_libraries(bench_egs4_native PRIVATE c7_egs4_tables)
add_executable(bench_egs4_c7_comparison bench_c7_comparison.cpp)
target_link_libraries(bench_egs4_c7_comparison PRIVATE c7_egs4_tables)
if(NOT EGS4_CPP_CUDA)
  target_compile_options(bench_egs4_c7_comparison PRIVATE -ffp-contract=off)
endif()
target_include_directories(bench_egs4_native PRIVATE
  "${EGS4_C8_SOURCE}")
option(EGS4_CPP_C8_OUTPUT_TEST "Build isolated integration with real C8 writers and radio" OFF)
if(EGS4_CPP_C8_OUTPUT_TEST)
  set(EGS4_C8_INSTALL "" CACHE PATH "C8 generated headers and dependency configs")
  list(APPEND CMAKE_PREFIX_PATH "${EGS4_C8_INSTALL}/lib/cmake/dependencies")
  find_package(Eigen3 CONFIG REQUIRED)
  find_package(spdlog CONFIG REQUIRED)
  find_package(yaml-cpp CONFIG REQUIRED)
  find_package(Boost CONFIG REQUIRED COMPONENTS filesystem)
  find_package(Arrow CONFIG REQUIRED)
  add_library(c7_egs4_c8_session Egs4C8Session.cpp)
  target_link_libraries(c7_egs4_c8_session PUBLIC c7_egs4_tables Eigen3::Eigen spdlog::spdlog)
  target_include_directories(c7_egs4_c8_session PUBLIC "${EGS4_C8_SOURCE}" "${EGS4_C8_INSTALL}/include")
  if(EGS4_CPP_CUDA)
    target_compile_options(c7_egs4_c8_session PRIVATE --extended-lambda)
  endif()
  add_executable(test_egs4_c8_session test_c8_session.cpp)
  target_link_libraries(test_egs4_c8_session PRIVATE c7_egs4_c8_session)
  foreach(owner IN ITEMS owned external)
    add_test(NAME native_cpp_c7_session_${owner} COMMAND test_egs4_c8_session
      "${EGS4_TABLE}" "${owner}")
    set_tests_properties(native_cpp_c7_session_${owner} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  endforeach()
  add_executable(test_egs4_c8_output test_c8_output.cpp)
  if(EGS4_CPP_CUDA)
    # nvcc_wrapper's supported host-only mode: C8's quantity/output templates
    # remain ordinary C++, while Egs4C8Session.cpp compiles the CUDA kernels.
    set_source_files_properties(test_c8_output.cpp PROPERTIES COMPILE_OPTIONS "--host-only")
  endif()
  target_include_directories(test_egs4_c8_output PRIVATE "${EGS4_C8_SOURCE}" "${EGS4_C8_INSTALL}/include")
  # C8 geometry/writers/radio are header implementations. No full C8 aggregate
  # target: that would add unrelated PROPOSAL and Fortran hadronic libraries.
  target_link_libraries(test_egs4_c8_output PRIVATE c7_egs4_c8_session Eigen3::Eigen spdlog::spdlog
    yaml-cpp::yaml-cpp Boost::filesystem Parquet::parquet_static)
  add_test(NAME native_cpp_c7_real_c8_output COMMAND test_egs4_c8_output
    "${EGS4_TABLE}")
  set_tests_properties(native_cpp_c7_real_c8_output PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  set(EGS4_PYTHIA_ROOT "" CACHE PATH "Existing C8 Pythia8 C++ installation")
  find_library(EGS4_PYTHIA_LIBRARY NAMES pythia8 PATHS "${EGS4_PYTHIA_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  add_executable(test_egs4_c8_hybrid test_c8_hybrid.cpp)
  if(EGS4_CPP_CUDA)
    set_source_files_properties(test_c8_hybrid.cpp PROPERTIES COMPILE_OPTIONS "--host-only")
  endif()
  target_include_directories(test_egs4_c8_hybrid PRIVATE "${EGS4_C8_SOURCE}" "${EGS4_C8_INSTALL}/include" "${EGS4_PYTHIA_ROOT}/include")
  target_link_libraries(test_egs4_c8_hybrid PRIVATE c7_egs4_c8_session Eigen3::Eigen spdlog::spdlog
    yaml-cpp::yaml-cpp ${EGS4_PYTHIA_LIBRARY} ${CMAKE_DL_LIBS})
  add_test(NAME native_cpp_c7_real_c8_hybrid COMMAND test_egs4_c8_hybrid
    "${EGS4_TABLE}")
  set_tests_properties(native_cpp_c7_real_c8_hybrid PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  # Existing C8 hadronic generators remain CPU libraries. They are not C7's
  # Fortran electromagnetic code; only the isolated native EM kernels use Kokkos.
  find_library(EGS4_SOPHIA_LIBRARY NAMES Sophia PATHS "${EGS4_C8_INSTALL}/lib/corsika" NO_DEFAULT_PATH REQUIRED)
  find_library(EGS4_SIBYLL_LIBRARY NAMES SIBYLL23d PATHS "${EGS4_C8_INSTALL}/lib/corsika" NO_DEFAULT_PATH REQUIRED)
  add_executable(test_egs4_c8_host test_c8_host.cpp)
  if(EGS4_CPP_CUDA)
    set_source_files_properties(test_c8_host.cpp PROPERTIES COMPILE_OPTIONS "--host-only")
  endif()
  target_include_directories(test_egs4_c8_host PRIVATE "${EGS4_C8_SOURCE}" "${EGS4_C8_INSTALL}/include"
    "${EGS4_C8_INSTALL}/include/corsika_modules/common"
    "${EGS4_C8_INSTALL}/include/corsika_modules/sophia" "${EGS4_C8_INSTALL}/include/corsika_modules/sibyll23d")
  target_link_libraries(test_egs4_c8_host PRIVATE c7_egs4_c8_session Eigen3::Eigen spdlog::spdlog
    yaml-cpp::yaml-cpp ${EGS4_SOPHIA_LIBRARY} ${EGS4_SIBYLL_LIBRARY})
  add_test(NAME native_cpp_c7_real_c8_host COMMAND test_egs4_c8_host)
  set_tests_properties(native_cpp_c7_real_c8_host PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  option(EGS4_CPP_C8_SHOWER "Build isolated native EM + existing C8 hadron/muon application" OFF)
  if(EGS4_CPP_C8_SHOWER)
    find_package(PROPOSAL CONFIG REQUIRED)
    find_package(CLI11 CONFIG REQUIRED)
    add_executable(test_egs4_fe_reference test_fe_reference.cpp)
    target_link_libraries(test_egs4_fe_reference PRIVATE CLI11::CLI11)
    target_compile_features(test_egs4_fe_reference PRIVATE cxx_std_17)
    add_test(NAME native_cpp_fe_reference COMMAND test_egs4_fe_reference)
    find_library(EGS4_FLUKA_LIBRARY NAMES fluka PATHS "${EGS4_C8_INSTALL}/lib/corsika" NO_DEFAULT_PATH REQUIRED)
    find_library(EGS4_QGSJET_LIBRARY NAMES QGSJetII PATHS "${EGS4_C8_INSTALL}/lib/corsika" NO_DEFAULT_PATH REQUIRED)
    find_library(EGS4_QGSJETIII_LIBRARY NAMES QGSJetIII PATHS "${EGS4_C8_INSTALL}/lib/corsika" NO_DEFAULT_PATH REQUIRED)
    find_library(EGS4_TAUOLA_CXX_LIBRARY NAMES TauolaCxxInterface PATHS "${EGS4_C8_INSTALL}/lib/corsika" NO_DEFAULT_PATH REQUIRED)
    find_library(EGS4_TAUOLA_FORTRAN_LIBRARY NAMES TauolaFortran PATHS "${EGS4_C8_INSTALL}/lib/corsika" NO_DEFAULT_PATH REQUIRED)
    # Reuse the original C8 SIBYLL objects, with a separate SONAME and one
    # lifecycle hook. Never modify/reinstall the production hadronic library.
    find_library(EGS4_SIBYLL_STATIC_LIBRARY NAMES SIBYLL23d_static
      PATHS "${EGS4_C8_INSTALL}/lib/corsika"
      NO_DEFAULT_PATH REQUIRED)
    set(persistent_sibyll_sources PersistentSibyll.cpp
      "${EGS4_C8_SOURCE}/modules/sibyll/sibyll2.3d-public.cpp"
      "${EGS4_C8_SOURCE}/modules/sibyll/nuclib-public.cpp")
    add_library(EGS4PersistentSIBYLL23d SHARED ${persistent_sibyll_sources})
    target_include_directories(EGS4PersistentSIBYLL23d PRIVATE
      "${EGS4_C8_SOURCE}/modules/sibyll" "${EGS4_C8_SOURCE}/modules/common")
    target_link_libraries(EGS4PersistentSIBYLL23d PRIVATE ${EGS4_SIBYLL_STATIC_LIBRARY} gfortran)
    target_link_options(EGS4PersistentSIBYLL23d PRIVATE "LINKER:--exclude-libs,ALL")
    if(EGS4_CPP_CUDA)
      set_source_files_properties(${persistent_sibyll_sources} PROPERTIES COMPILE_OPTIONS "--host-only")
    endif()
    add_executable(c8_egs4_shower c8_egs4_shower.cpp)
    find_package(nlohmann_json CONFIG REQUIRED)
    find_package(CubicInterpolation CONFIG REQUIRED)
    find_package(OpenSSL CONFIG REQUIRED)
    # Build the UNMODIFIED C8 PROPOSAL/Kokkos sources inside this experiment.
    # Do not link an installed archive with potentially stale structure layouts.
    set(proposal_native_sources ProposalNativeTable.cpp ProposalNativeAux.cpp
      MoliereInterpolation.cpp ProposalNativeRequirements.cpp Sha256.cpp)
    list(TRANSFORM proposal_native_sources PREPEND "${EGS4_C8_SOURCE}/src/accelerator/em/tables/")
    add_library(egs4_proposal_native STATIC ${proposal_native_sources})
    target_include_directories(egs4_proposal_native PUBLIC "${EGS4_C8_SOURCE}" "${EGS4_C8_INSTALL}/include")
    target_link_libraries(egs4_proposal_native PUBLIC PROPOSAL::PROPOSAL yaml-cpp::yaml-cpp
      CubicInterpolation::CubicInterpolation nlohmann_json::nlohmann_json Eigen3::Eigen)
    target_compile_features(egs4_proposal_native PUBLIC cxx_std_17)
    set(muon_backend_sources KokkosEmBackend.cpp KokkosBackendSelection.cpp
      KokkosEmRunSession.cpp KokkosRuntime.cpp KokkosTuningCache.cpp)
    if(EGS4_CPP_CUDA)
      list(APPEND muon_backend_sources KokkosCudaBackendInstance.cpp)
      set(muon_backend_kind CUDA)
      set_source_files_properties(${proposal_native_sources} PROPERTIES COMPILE_OPTIONS "--host-only")
    else()
      list(APPEND muon_backend_sources KokkosOpenMPBackendInstance.cpp)
      set(muon_backend_kind OPENMP)
    endif()
    list(TRANSFORM muon_backend_sources PREPEND "${EGS4_C8_SOURCE}/src/accelerator/em/kokkos/")
    add_library(egs4_proposal_muon STATIC ${muon_backend_sources})
    target_link_libraries(egs4_proposal_muon PUBLIC egs4_proposal_native Kokkos::kokkos)
    target_compile_definitions(egs4_proposal_muon PRIVATE CORSIKA8_WITH_KOKKOS=1
      CORSIKA8_KOKKOS_BACKEND_${muon_backend_kind}=1 CORSIKA8_PROJECT_REVISION="egs4-muon-experiment")
    option(EGS4_CPP_STABLE_MUON_IONIZATION "Experiment-only equivalent transverse-momentum muon ionization angles" OFF)
    if(EGS4_CPP_STABLE_MUON_IONIZATION)
      include(stable_muon_overlay.cmake)
    endif()
    if(EGS4_CPP_CUDA)
      target_compile_options(egs4_proposal_muon PRIVATE --extended-lambda)
      target_link_libraries(egs4_proposal_muon PUBLIC CUDA::cudart CUDA::cuda_driver)
      foreach(host_source KokkosEmBackend.cpp KokkosBackendSelection.cpp KokkosEmRunSession.cpp KokkosTuningCache.cpp)
        set_source_files_properties("${EGS4_C8_SOURCE}/src/accelerator/em/kokkos/${host_source}"
          PROPERTIES COMPILE_OPTIONS "--host-only")
      endforeach()
    endif()
    add_executable(egs4_test_schedule egs4_schedule.cpp Egs4MultiGpuIO.cpp)
    # Some existing C8 dependency manifests omit OpenSSL's public include path.
    find_path(EGS4_OPENSSL_INCLUDE openssl/evp.h HINTS "${openssl_PACKAGE_FOLDER_RELEASE}/include" ${OpenSSL_INCLUDE_DIRS} REQUIRED)
    target_include_directories(egs4_test_schedule PRIVATE "${EGS4_OPENSSL_INCLUDE}")
    target_link_libraries(egs4_test_schedule PRIVATE CLI11::CLI11 nlohmann_json::nlohmann_json
      yaml-cpp::yaml-cpp Parquet::parquet_static OpenSSL::Crypto)
    target_compile_features(egs4_test_schedule PRIVATE cxx_std_17)
    add_executable(test_egs4_schedule_io test_schedule_io.cpp Egs4MultiGpuIO.cpp)
    target_include_directories(test_egs4_schedule_io PRIVATE "${EGS4_OPENSSL_INCLUDE}")
    target_link_libraries(test_egs4_schedule_io PRIVATE nlohmann_json::nlohmann_json yaml-cpp::yaml-cpp Parquet::parquet_static OpenSSL::Crypto)
    target_compile_features(test_egs4_schedule_io PRIVATE cxx_std_17)
    add_test(NAME native_cpp_c7_schedule_io COMMAND test_egs4_schedule_io)
    if(EGS4_CPP_CUDA)
      set_source_files_properties(c8_egs4_shower.cpp PROPERTIES COMPILE_OPTIONS "--host-only")
      set_source_files_properties(egs4_schedule.cpp Egs4MultiGpuIO.cpp test_schedule_io.cpp PROPERTIES COMPILE_OPTIONS "--host-only")
    endif()
    target_include_directories(c8_egs4_shower PRIVATE "${EGS4_C8_SOURCE}" "${EGS4_C8_INSTALL}/include"
      "${EGS4_PYTHIA_ROOT}/include" "${EGS4_C8_INSTALL}/include/corsika_modules/common"
      "${EGS4_C8_INSTALL}/include/corsika_modules"
      "${EGS4_C8_INSTALL}/include/corsika_modules/sophia" "${EGS4_C8_INSTALL}/include/corsika_modules/sibyll23d"
      "${EGS4_C8_INSTALL}/include/corsika_modules/fluka" "${EGS4_C8_INSTALL}/include/corsika_modules/qgsjetII"
      "${EGS4_C8_INSTALL}/include/corsika_modules/qgsjetIII")
    target_link_libraries(c8_egs4_shower PRIVATE c7_egs4_c8_session egs4_proposal_muon Eigen3::Eigen spdlog::spdlog
      yaml-cpp::yaml-cpp Boost::filesystem Parquet::parquet_static PROPOSAL::PROPOSAL CLI11::CLI11
      ${EGS4_SOPHIA_LIBRARY} EGS4PersistentSIBYLL23d ${EGS4_FLUKA_LIBRARY} ${EGS4_QGSJET_LIBRARY} ${EGS4_QGSJETIII_LIBRARY}
      ${EGS4_TAUOLA_CXX_LIBRARY} ${EGS4_TAUOLA_FORTRAN_LIBRARY} gfortran
      ${EGS4_PYTHIA_LIBRARY} ${CMAKE_DL_LIBS})
    set(EGS4_SHOWER_TEST_PYTHON "" CACHE FILEPATH "Optional Python with numpy/pyarrow for application output regression")
    include(persistent_output_overlay.cmake)
    set(EGS4_SHOWER_TEST_FLUPRO "" CACHE PATH "Existing licensed FLUKA data for the isolated application test")
    if(EGS4_SHOWER_TEST_PYTHON AND EGS4_SHOWER_TEST_FLUPRO)
      add_test(NAME native_cpp_persistent_events COMMAND ${EGS4_SHOWER_TEST_PYTHON}
        "${CMAKE_CURRENT_SOURCE_DIR}/test_persistent_events.py" $<TARGET_FILE:c8_egs4_shower>
        "${EGS4_TABLE}" --root "${CMAKE_CURRENT_BINARY_DIR}")
      set_tests_properties(native_cpp_persistent_events PROPERTIES TIMEOUT 1800
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;FLUPRO=${EGS4_SHOWER_TEST_FLUPRO}")
      add_test(NAME native_cpp_resident_output COMMAND ${EGS4_SHOWER_TEST_PYTHON}
        "${CMAKE_CURRENT_SOURCE_DIR}/test_resident_output.py" $<TARGET_FILE:c8_egs4_shower>
        "${EGS4_TABLE}" --root "${CMAKE_CURRENT_BINARY_DIR}")
      set_tests_properties(native_cpp_resident_output PROPERTIES TIMEOUT 1200
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;FLUPRO=${EGS4_SHOWER_TEST_FLUPRO}")
      add_test(NAME native_cpp_proposal_muon_application COMMAND ${EGS4_SHOWER_TEST_PYTHON}
        "${CMAKE_CURRENT_SOURCE_DIR}/test_muon_application.py" $<TARGET_FILE:c8_egs4_shower>
        "${EGS4_TABLE}" --root "${CMAKE_CURRENT_BINARY_DIR}")
      set_tests_properties(native_cpp_proposal_muon_application PROPERTIES TIMEOUT 900
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;FLUPRO=${EGS4_SHOWER_TEST_FLUPRO}")
      add_test(NAME native_cpp_c7_c8_mixed_shower COMMAND ${EGS4_SHOWER_TEST_PYTHON}
        "${CMAKE_CURRENT_SOURCE_DIR}/test_shower_application.py" $<TARGET_FILE:c8_egs4_shower>
        "${EGS4_TABLE}" --root "${CMAKE_CURRENT_BINARY_DIR}")
      set_tests_properties(native_cpp_c7_c8_mixed_shower PROPERTIES TIMEOUT 1200
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;FLUPRO=${EGS4_SHOWER_TEST_FLUPRO}")
      if(EGS4_CPP_CUDA)
        add_test(NAME native_cpp_c7_c8_mixed_shower_gpu_budget COMMAND ${EGS4_SHOWER_TEST_PYTHON}
          "${CMAKE_CURRENT_SOURCE_DIR}/test_shower_application.py" $<TARGET_FILE:c8_egs4_shower>
          "${EGS4_TABLE}" --root "${CMAKE_CURRENT_BINARY_DIR}"
          --gpu-memory-fraction 0.9)
        set_tests_properties(native_cpp_c7_c8_mixed_shower_gpu_budget PROPERTIES TIMEOUT 1200
          ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;FLUPRO=${EGS4_SHOWER_TEST_FLUPRO}")
      endif()
      if(NOT EGS4_CPP_CUDA)
        add_test(NAME native_cpp_c7_c8_cpu_schedule COMMAND ${EGS4_SHOWER_TEST_PYTHON}
          "${CMAKE_CURRENT_SOURCE_DIR}/test_schedule.py" $<TARGET_FILE:egs4_test_schedule> $<TARGET_FILE:c8_egs4_shower>
          "${EGS4_TABLE}" --root "${CMAKE_CURRENT_BINARY_DIR}")
        set_tests_properties(native_cpp_c7_c8_cpu_schedule PROPERTIES TIMEOUT 1200
          ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false;FLUPRO=${EGS4_SHOWER_TEST_FLUPRO}")
      endif()
    endif()
  endif()
endif()
option(EGS4_CPP_REFERENCE_TESTS "Build a SEPARATE original C7 reference fixture generator" OFF)
if(EGS4_CPP_REFERENCE_TESTS)
  if(NOT EXISTS "${EGS4_C7_SOURCE}" OR NOT EXISTS "${EGS4_C7_SCATTER_REFERENCE}")
    message(FATAL_ERROR "Reference tests require EGS4_C7_SOURCE and EGS4_C7_SCATTER_REFERENCE")
  endif()
  find_program(EGS4_TEST_GFORTRAN gfortran REQUIRED)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(ORACLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/external_reference")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_nuclear_target_reference"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_nuclear_target_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}/nuclear_target_oracle.f"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/nuclear_target_oracle.f" -o "${ORACLE_DIR}/c7_nuclear_target_reference"
    DEPENDS prepare_nuclear_target_oracle.py "${EGS4_C7_SOURCE}")
  add_custom_target(c7_external_nuclear_target_reference ALL DEPENDS "${ORACLE_DIR}/c7_nuclear_target_reference")
  add_test(NAME native_cpp_c7_nuclear_target_input COMMAND test_egs4_nuclear_target --write-input "${ORACLE_DIR}/nuclear_target_input.txt")
  add_test(NAME native_cpp_c7_nuclear_target_reference COMMAND "${ORACLE_DIR}/c7_nuclear_target_reference"
    "${ORACLE_DIR}/nuclear_target_input.txt" "${ORACLE_DIR}/nuclear_target_reference.txt")
  add_test(NAME native_cpp_c7_nuclear_target_compare COMMAND test_egs4_nuclear_target --compare "${ORACLE_DIR}/nuclear_target_reference.txt")
  set_tests_properties(native_cpp_c7_nuclear_target_input PROPERTIES FIXTURES_SETUP target_input)
  set_tests_properties(native_cpp_c7_nuclear_target_reference PROPERTIES FIXTURES_REQUIRED target_input FIXTURES_SETUP target_reference)
  set_tests_properties(native_cpp_c7_nuclear_target_compare PROPERTIES FIXTURES_REQUIRED target_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  add_custom_command(OUTPUT "${ORACLE_DIR}/meson_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_meson_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_meson_oracle.py "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_meson_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/meson_oracle.f" -o "${ORACLE_DIR}/c7_meson_reference"
    DEPENDS "${ORACLE_DIR}/meson_oracle.f")
  add_custom_target(c7_external_meson_reference ALL DEPENDS "${ORACLE_DIR}/c7_meson_reference")
  add_test(NAME generate_meson_cases COMMAND test_egs4_meson --make-cases "${ORACLE_DIR}/meson_inputs.txt")
  add_test(NAME generate_c7_meson_reference COMMAND "${ORACLE_DIR}/c7_meson_reference"
    "${ORACLE_DIR}/meson_inputs.txt" "${ORACLE_DIR}/meson_outputs.txt")
  add_test(NAME native_cpp_c7_meson_comparison COMMAND test_egs4_meson "${ORACLE_DIR}/meson_outputs.txt")
  set_tests_properties(generate_meson_cases PROPERTIES FIXTURES_SETUP meson_cases)
  set_tests_properties(generate_c7_meson_reference PROPERTIES FIXTURES_REQUIRED meson_cases FIXTURES_SETUP meson_reference TIMEOUT 60)
  set_tests_properties(native_cpp_c7_meson_comparison PROPERTIES FIXTURES_REQUIRED meson_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  add_custom_command(OUTPUT "${ORACLE_DIR}/two_body_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_two_body_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_two_body_oracle.py "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_two_body_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/two_body_oracle.f" -o "${ORACLE_DIR}/c7_two_body_reference"
    DEPENDS "${ORACLE_DIR}/two_body_oracle.f")
  add_custom_target(c7_external_two_body_reference ALL DEPENDS "${ORACLE_DIR}/c7_two_body_reference")
  add_test(NAME generate_two_body_cases COMMAND test_egs4_two_body --make-cases "${ORACLE_DIR}/two_body_inputs.txt")
  add_test(NAME generate_c7_two_body_reference COMMAND "${ORACLE_DIR}/c7_two_body_reference"
    "${ORACLE_DIR}/two_body_inputs.txt" "${ORACLE_DIR}/two_body_outputs.txt")
  add_test(NAME native_cpp_c7_two_body_comparison COMMAND test_egs4_two_body "${ORACLE_DIR}/two_body_outputs.txt")
  set_tests_properties(generate_two_body_cases PROPERTIES FIXTURES_SETUP two_body_cases)
  set_tests_properties(generate_c7_two_body_reference PROPERTIES FIXTURES_REQUIRED two_body_cases FIXTURES_SETUP two_body_reference TIMEOUT 60)
  set_tests_properties(native_cpp_c7_two_body_comparison PROPERTIES FIXTURES_REQUIRED two_body_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  add_custom_command(OUTPUT "${ORACLE_DIR}/resonance_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_resonance_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_resonance_oracle.py prepare_two_body_oracle.py "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_resonance_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/resonance_oracle.f" -o "${ORACLE_DIR}/c7_resonance_reference"
    DEPENDS "${ORACLE_DIR}/resonance_oracle.f")
  add_custom_target(c7_external_resonance_reference ALL DEPENDS "${ORACLE_DIR}/c7_resonance_reference")
  add_test(NAME generate_resonance_cases COMMAND test_egs4_resonance --make-cases "${ORACLE_DIR}/resonance_inputs.txt")
  add_test(NAME generate_c7_resonance_reference COMMAND "${ORACLE_DIR}/c7_resonance_reference"
    "${ORACLE_DIR}/resonance_inputs.txt" "${ORACLE_DIR}/resonance_outputs.txt")
  add_test(NAME native_cpp_c7_resonance_comparison COMMAND test_egs4_resonance "${ORACLE_DIR}/resonance_outputs.txt")
  set_tests_properties(generate_resonance_cases PROPERTIES FIXTURES_SETUP resonance_cases)
  set_tests_properties(generate_c7_resonance_reference PROPERTIES FIXTURES_REQUIRED resonance_cases FIXTURES_SETUP resonance_reference TIMEOUT 60)
  set_tests_properties(native_cpp_c7_resonance_comparison PROPERTIES FIXTURES_REQUIRED resonance_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  add_custom_command(OUTPUT "${ORACLE_DIR}/photonuclear_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_photonuclear_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_photonuclear_oracle.py "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_photonuclear_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/photonuclear_oracle.f" -o "${ORACLE_DIR}/c7_photonuclear_reference"
    DEPENDS "${ORACLE_DIR}/photonuclear_oracle.f")
  add_custom_target(c7_external_photonuclear_reference ALL DEPENDS "${ORACLE_DIR}/c7_photonuclear_reference")
  add_test(NAME generate_photonuclear_cases COMMAND test_egs4_photonuclear --make-cases "${ORACLE_DIR}/photonuclear_inputs.txt")
  add_test(NAME generate_c7_photonuclear_reference COMMAND "${ORACLE_DIR}/c7_photonuclear_reference"
    "${ORACLE_DIR}/photonuclear_inputs.txt" "${ORACLE_DIR}/photonuclear_outputs.txt")
  add_test(NAME native_cpp_c7_photonuclear_comparison COMMAND test_egs4_photonuclear "${ORACLE_DIR}/photonuclear_outputs.txt")
  set_tests_properties(generate_photonuclear_cases PROPERTIES FIXTURES_SETUP photonuclear_cases)
  set_tests_properties(generate_c7_photonuclear_reference PROPERTIES FIXTURES_REQUIRED photonuclear_cases FIXTURES_SETUP photonuclear_reference)
  set_tests_properties(native_cpp_c7_photonuclear_comparison PROPERTIES FIXTURES_REQUIRED photonuclear_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  add_custom_command(OUTPUT "${ORACLE_DIR}/rare_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_rare_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_rare_oracle.py "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_rare_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/rare_oracle.f" -o "${ORACLE_DIR}/c7_rare_reference"
    DEPENDS "${ORACLE_DIR}/rare_oracle.f")
  add_custom_target(c7_external_rare_reference ALL DEPENDS "${ORACLE_DIR}/c7_rare_reference")
  add_test(NAME generate_rare_cases COMMAND test_egs4_rare --make-cases "${ORACLE_DIR}/rare_inputs.txt")
  add_test(NAME generate_c7_rare_reference COMMAND "${ORACLE_DIR}/c7_rare_reference"
    "${ORACLE_DIR}/rare_inputs.txt" "${ORACLE_DIR}/rare_outputs.txt")
  add_test(NAME native_cpp_c7_rare_comparison COMMAND test_egs4_rare "${ORACLE_DIR}/rare_outputs.txt")
  set_tests_properties(generate_rare_cases PROPERTIES FIXTURES_SETUP rare_cases)
  set_tests_properties(generate_c7_rare_reference PROPERTIES FIXTURES_REQUIRED rare_cases FIXTURES_SETUP rare_reference TIMEOUT 60)
  set_tests_properties(native_cpp_c7_rare_comparison PROPERTIES FIXTURES_REQUIRED rare_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  add_custom_command(OUTPUT "${ORACLE_DIR}/observation_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_observation_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_observation_oracle.py
      "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_observation_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/observation_oracle.f" -o "${ORACLE_DIR}/c7_observation_reference"
    DEPENDS "${ORACLE_DIR}/observation_oracle.f")
  add_custom_target(c7_external_observation_reference ALL DEPENDS "${ORACLE_DIR}/c7_observation_reference")
  add_test(NAME generate_observation_cases COMMAND test_egs4_observation --make-cases "${ORACLE_DIR}/observation_inputs.txt")
  add_test(NAME generate_c7_observation_reference COMMAND "${ORACLE_DIR}/c7_observation_reference"
    "${ORACLE_DIR}/observation_inputs.txt" "${ORACLE_DIR}/observation_outputs.txt")
  add_test(NAME native_cpp_c7_observation_comparison COMMAND test_egs4_observation "${ORACLE_DIR}/observation_outputs.txt")
  set_tests_properties(generate_observation_cases PROPERTIES FIXTURES_SETUP observation_cases)
  set_tests_properties(generate_c7_observation_reference PROPERTIES FIXTURES_REQUIRED observation_cases FIXTURES_SETUP observation_reference)
  set_tests_properties(native_cpp_c7_observation_comparison PROPERTIES FIXTURES_REQUIRED observation_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  add_custom_command(OUTPUT "${ORACLE_DIR}/curved_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_curved_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_curved_oracle.py
      "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_curved_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/curved_oracle.f" -o "${ORACLE_DIR}/c7_curved_reference"
    DEPENDS "${ORACLE_DIR}/curved_oracle.f")
  add_custom_target(c7_external_curved_reference ALL DEPENDS "${ORACLE_DIR}/c7_curved_reference")
  add_test(NAME generate_curved_cases COMMAND test_egs4_curved --make-cases "${ORACLE_DIR}/curved_inputs.txt")
  add_test(NAME generate_c7_curved_reference COMMAND "${ORACLE_DIR}/c7_curved_reference"
    "${ORACLE_DIR}/curved_inputs.txt" "${ORACLE_DIR}/curved_outputs.txt")
  add_test(NAME native_cpp_c7_curved_comparison COMMAND test_egs4_curved "${ORACLE_DIR}/curved_outputs.txt")
  set_tests_properties(generate_curved_cases PROPERTIES FIXTURES_SETUP curved_cases)
  set_tests_properties(generate_c7_curved_reference PROPERTIES FIXTURES_REQUIRED curved_cases FIXTURES_SETUP curved_reference)
  set_tests_properties(native_cpp_c7_curved_comparison PROPERTIES FIXTURES_REQUIRED curved_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  add_custom_command(OUTPUT "${ORACLE_DIR}/geometry_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_geometry_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_geometry_oracle.py
      "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_geometry_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/geometry_oracle.f" -o "${ORACLE_DIR}/c7_geometry_reference"
    DEPENDS "${ORACLE_DIR}/geometry_oracle.f")
  add_custom_target(c7_external_geometry_reference ALL DEPENDS "${ORACLE_DIR}/c7_geometry_reference")
  add_test(NAME generate_geometry_cases COMMAND test_egs4_geometry --make-cases "${ORACLE_DIR}/geometry_inputs.txt")
  add_test(NAME generate_c7_geometry_reference COMMAND "${ORACLE_DIR}/c7_geometry_reference"
    "${ORACLE_DIR}/geometry_inputs.txt" "${ORACLE_DIR}/geometry_outputs.txt")
  add_test(NAME native_cpp_c7_geometry_comparison COMMAND test_egs4_geometry "${ORACLE_DIR}/geometry_outputs.txt")
  set_tests_properties(generate_geometry_cases PROPERTIES FIXTURES_SETUP geometry_cases)
  set_tests_properties(generate_c7_geometry_reference PROPERTIES FIXTURES_REQUIRED geometry_cases FIXTURES_SETUP geometry_reference)
  set_tests_properties(native_cpp_c7_geometry_comparison PROPERTIES FIXTURES_REQUIRED geometry_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  add_custom_command(OUTPUT "${ORACLE_DIR}/collision_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_collision_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_collision_oracle.py
      "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/collision_oracle.f" -o "${ORACLE_DIR}/c7_reference"
    DEPENDS "${ORACLE_DIR}/collision_oracle.f")
  add_custom_target(c7_external_reference ALL DEPENDS "${ORACLE_DIR}/c7_reference")
  add_test(NAME generate_collision_cases COMMAND test_egs4_interactions
    --make-cases "${ORACLE_DIR}/inputs.txt")
  add_test(NAME generate_c7_collision_reference COMMAND "${ORACLE_DIR}/c7_reference"
    "${ORACLE_DIR}/inputs.txt" "${ORACLE_DIR}/outputs.txt")
  add_test(NAME native_cpp_c7_collision_comparison COMMAND test_egs4_interactions
    "${ORACLE_DIR}/outputs.txt")
  set_tests_properties(generate_collision_cases PROPERTIES FIXTURES_SETUP collision_cases)
  set_tests_properties(generate_c7_collision_reference PROPERTIES
    FIXTURES_REQUIRED collision_cases FIXTURES_SETUP collision_reference)
  set_tests_properties(native_cpp_c7_collision_comparison PROPERTIES
    FIXTURES_REQUIRED collision_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

  add_custom_command(OUTPUT "${ORACLE_DIR}/radiative_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_radiative_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_radiative_oracle.py
      "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_radiative_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/radiative_oracle.f" -o "${ORACLE_DIR}/c7_radiative_reference"
    DEPENDS "${ORACLE_DIR}/radiative_oracle.f")
  add_custom_target(c7_external_radiative_reference ALL DEPENDS "${ORACLE_DIR}/c7_radiative_reference")
  add_test(NAME generate_radiative_cases COMMAND test_egs4_radiative
    --make-cases "${ORACLE_DIR}/radiative_inputs.txt")
  add_test(NAME generate_c7_radiative_reference COMMAND "${ORACLE_DIR}/c7_radiative_reference"
    "${ORACLE_DIR}/radiative_inputs.txt" "${ORACLE_DIR}/radiative_outputs.txt"
    "${EGS4_TABLE}")
  add_test(NAME native_cpp_c7_radiative_comparison COMMAND test_egs4_radiative
    "${ORACLE_DIR}/radiative_outputs.txt"
    "${EGS4_TABLE}")
  set_tests_properties(generate_radiative_cases PROPERTIES FIXTURES_SETUP radiative_cases)
  set_tests_properties(generate_c7_radiative_reference PROPERTIES
    FIXTURES_REQUIRED radiative_cases FIXTURES_SETUP radiative_reference)
  set_tests_properties(native_cpp_c7_radiative_comparison PROPERTIES
    FIXTURES_REQUIRED radiative_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

  set(B_SCATTER_REFERENCE "${EGS4_C7_SCATTER_REFERENCE}")
  file(SHA256 "${B_SCATTER_REFERENCE}" B_SCATTER_HASH)
  if(NOT B_SCATTER_HASH STREQUAL "6e5edee37f9ac56fff3e6b23b398c54dd1e30b6923520bba4870c0f69edf76d1")
    message(FATAL_ERROR "The earlier B scattering reference changed; re-audit before reuse")
  endif()
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_scattering_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${B_SCATTER_REFERENCE}" "${CMAKE_CURRENT_SOURCE_DIR}/scattering_reference_driver.f90"
      -o "${ORACLE_DIR}/c7_scattering_reference"
    DEPENDS "${B_SCATTER_REFERENCE}" scattering_reference_driver.f90)
  add_custom_target(c7_external_scattering_reference ALL DEPENDS "${ORACLE_DIR}/c7_scattering_reference")
  add_test(NAME generate_scattering_cases COMMAND test_egs4_scattering
    --make-cases "${ORACLE_DIR}/scattering_inputs.txt")
  add_test(NAME generate_c7_scattering_reference COMMAND "${ORACLE_DIR}/c7_scattering_reference"
    "${ORACLE_DIR}/scattering_inputs.txt" "${ORACLE_DIR}/scattering_outputs.txt")
  add_test(NAME native_cpp_c7_scattering_comparison COMMAND test_egs4_scattering
    "${ORACLE_DIR}/scattering_outputs.txt"
    "${EGS4_TABLE}")
  set_tests_properties(generate_scattering_cases PROPERTIES FIXTURES_SETUP scattering_cases)
  set_tests_properties(generate_c7_scattering_reference PROPERTIES
    FIXTURES_REQUIRED scattering_cases FIXTURES_SETUP scattering_reference)
  set_tests_properties(native_cpp_c7_scattering_comparison PROPERTIES
    FIXTURES_REQUIRED scattering_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

  add_custom_command(OUTPUT "${ORACLE_DIR}/termination_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_termination_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_termination_oracle.py
      "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_termination_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/termination_oracle.f" -o "${ORACLE_DIR}/c7_termination_reference"
    DEPENDS "${ORACLE_DIR}/termination_oracle.f")
  add_custom_target(c7_external_termination_reference ALL DEPENDS "${ORACLE_DIR}/c7_termination_reference")
  add_test(NAME generate_termination_cases COMMAND test_egs4_termination
    --make-cases "${ORACLE_DIR}/termination_inputs.txt")
  add_test(NAME generate_c7_termination_reference COMMAND "${ORACLE_DIR}/c7_termination_reference"
    "${ORACLE_DIR}/termination_inputs.txt" "${ORACLE_DIR}/termination_outputs.txt")
  add_test(NAME native_cpp_c7_termination_comparison COMMAND test_egs4_termination
    "${ORACLE_DIR}/termination_outputs.txt"
    "${EGS4_TABLE}")
  set_tests_properties(generate_termination_cases PROPERTIES FIXTURES_SETUP termination_cases)
  set_tests_properties(generate_c7_termination_reference PROPERTIES
    FIXTURES_REQUIRED termination_cases FIXTURES_SETUP termination_reference)
  set_tests_properties(native_cpp_c7_termination_comparison PROPERTIES
    FIXTURES_REQUIRED termination_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

  add_custom_command(OUTPUT "${ORACLE_DIR}/transport_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_transport_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_transport_oracle.py
      "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_transport_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "-J${ORACLE_DIR}" "${ORACLE_DIR}/transport_oracle.f" -o "${ORACLE_DIR}/c7_transport_reference"
    DEPENDS "${ORACLE_DIR}/transport_oracle.f")
  add_custom_target(c7_external_transport_reference ALL DEPENDS "${ORACLE_DIR}/c7_transport_reference")
  add_test(NAME generate_transport_cases COMMAND test_egs4_transport
    --make-cases "${ORACLE_DIR}/transport_inputs.txt")
  add_test(NAME generate_c7_transport_reference COMMAND "${ORACLE_DIR}/c7_transport_reference"
    "${ORACLE_DIR}/transport_inputs.txt" "${ORACLE_DIR}/transport_outputs.txt"
    "${EGS4_TABLE}")
  add_test(NAME native_cpp_c7_transport_comparison COMMAND test_egs4_transport
    "${ORACLE_DIR}/transport_outputs.txt"
    "${EGS4_TABLE}")
  set_tests_properties(generate_transport_cases PROPERTIES FIXTURES_SETUP transport_cases)
  set_tests_properties(generate_c7_transport_reference PROPERTIES
    FIXTURES_REQUIRED transport_cases FIXTURES_SETUP transport_reference)
  set_tests_properties(native_cpp_c7_transport_comparison PROPERTIES
    FIXTURES_REQUIRED transport_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

  add_custom_command(OUTPUT "${ORACLE_DIR}/photon_transport_oracle.f"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_photon_transport_oracle.py"
      "${EGS4_C7_SOURCE}" "${ORACLE_DIR}"
    DEPENDS prepare_photon_transport_oracle.py
      "${EGS4_C7_SOURCE}")
  add_custom_command(OUTPUT "${ORACLE_DIR}/c7_photon_transport_reference"
    COMMAND ${EGS4_TEST_GFORTRAN} -O2 -ffp-contract=off -ffixed-line-length-none
      "${ORACLE_DIR}/photon_transport_oracle.f" -o "${ORACLE_DIR}/c7_photon_transport_reference"
    DEPENDS "${ORACLE_DIR}/photon_transport_oracle.f")
  add_custom_target(c7_external_photon_transport_reference ALL DEPENDS "${ORACLE_DIR}/c7_photon_transport_reference")
  add_test(NAME generate_photon_transport_cases COMMAND test_egs4_photon_transport
    --make-cases "${ORACLE_DIR}/photon_transport_inputs.txt")
  add_test(NAME generate_c7_photon_transport_reference COMMAND "${ORACLE_DIR}/c7_photon_transport_reference"
    "${ORACLE_DIR}/photon_transport_inputs.txt" "${ORACLE_DIR}/photon_transport_outputs.txt"
    "${EGS4_TABLE}")
  add_test(NAME native_cpp_c7_photon_transport_comparison COMMAND test_egs4_photon_transport
    "${ORACLE_DIR}/photon_transport_outputs.txt"
    "${EGS4_TABLE}")
  set_tests_properties(generate_photon_transport_cases PROPERTIES FIXTURES_SETUP photon_transport_cases)
  set_tests_properties(generate_c7_photon_transport_reference PROPERTIES
    FIXTURES_REQUIRED photon_transport_cases FIXTURES_SETUP photon_transport_reference)
  set_tests_properties(native_cpp_c7_photon_transport_comparison PROPERTIES
    FIXTURES_REQUIRED photon_transport_reference ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endif()
