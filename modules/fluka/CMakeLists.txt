set(C8_FLUKALIB CACHE FILEPATH "path to libflukahp.a")

set(C8_FLUKALIB_ "")
if (EXISTS "$CACHE{C8_FLUKALIB}")
    message("libflukahp.a found at ${C8_FLUKALIB} via C8_FLUKALIB")
    set(C8_FLUKALIB_ "$CACHE{C8_FLUKALIB}")
elseif (EXISTS "$ENV{FLUPRO}/libflukahp.a")
    message("libflukahp.a found in directory $ENV{FLUPRO} via FLUPRO environment variable")
    set(C8_FLUKALIB_ "$ENV{FLUPRO}/libflukahp.a")
endif()

if ("${C8_FLUKALIB_}" STREQUAL "")
    message("C8_FLUKALIB not set/invalid and FLUPRO not set, building without FLUKA support")
    set(C8_FLUKA_ENABLED "" CACHE INTERNAL "" FORCE) # makeshift global var
else()
    message("FLUKA support is enabled.")
    set(C8_FLUKA_ENABLED "enabled" CACHE INTERNAL "" FORCE) # makeshift global var

    set (input_dir ${PROJECT_SOURCE_DIR}/src/modules/fluka)
    
    # we remove flrndm_.o from the original libflukahp.a and save the result as libflukahp-norndm.a
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libflukahp-norndm.a
        COMMAND ${input_dir}/strip_flukahp.sh ${C8_FLUKALIB_} ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_custom_target(generate_libfluka-norndm DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libflukahp-norndm.a)

    add_library(flukahp-norndm STATIC IMPORTED)
    add_dependencies(flukahp-norndm generate_libfluka-norndm)
    set_property(TARGET flukahp-norndm PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libflukahp-norndm.a)

    find_library(MATH_LIBRARY m)
    target_link_libraries(flukahp-norndm INTERFACE gfortran ${MATH_LIBRARY})
    target_include_directories (flukahp-norndm INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/corsika_modules/fluka>
    )

    try_compile(fluka_have_flkavr ${CMAKE_BINARY_DIR}
        SOURCES ${input_dir}/try_flkavr.cpp
        LINK_LIBRARIES ${C8_FLUKALIB_}
    )

    # build shared library containing FLUKA + our own implementation of flrndm_()
    # Note: This apparently works only when built as shared lib, not possible so easily as static lib

    add_library(fluka SHARED fluka.cpp)
    target_link_libraries(fluka PRIVATE flukahp-norndm)
    target_link_libraries(fluka PUBLIC modules_common)
    if (fluka_have_flkavr)
        message("FLUKA supports flkavr; version info available")
        target_compile_definitions(fluka PRIVATE FLUKA_HAS_FLKAVR)
    else()
        message("FLUKA does not support flkavr; cannot read version info, extracting via hack...")
        execute_process(
            COMMAND ${input_dir}/extract_version.sh ${C8_FLUKALIB_}
            RESULT_VARIABLE ret
            OUTPUT_VARIABLE extracted_fluka_version
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if (ret AND NOT ret EQUAL "0" OR extracted_fluka_version STREQUAL "")
          message("FLUKA version detection failed")
        else()
          message("extracted FLUKA version: \"${extracted_fluka_version}\"")
          target_compile_definitions(fluka PRIVATE FLUKA_EXTRACTED_VERSION=${extracted_fluka_version})
        endif()
    endif()

    target_include_directories (fluka PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:include/corsika_modules/fluka>
      )
    
    install (
      FILES
      FLUKA.hpp
      DESTINATION include/corsika_modules/fluka
      )
    
    install (
      TARGETS fluka
      EXPORT CORSIKA8PublicTargets
      LIBRARY DESTINATION lib/corsika
      INCLUDES DESTINATION include/corsika_modules/fluka
      )
    
    # add fluka to corsika8 build
    add_dependencies (CORSIKA8 fluka)
    target_link_libraries (CORSIKA8 INTERFACE fluka)
endif()
