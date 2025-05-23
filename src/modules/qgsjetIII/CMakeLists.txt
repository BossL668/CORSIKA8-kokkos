set (input_dir ${PROJECT_SOURCE_DIR}/src/modules/qgsjetIII)
set (output_dir ${PROJECT_BINARY_DIR}/corsika/modules/qgsjetIII)

file (MAKE_DIRECTORY ${output_dir})

add_custom_command (
  OUTPUT  ${output_dir}/Generated.inc
  COMMAND ${input_dir}/code_generator.py 
          ${PROJECT_BINARY_DIR}/corsika/framework/core/particle_db.pkl
          ${input_dir}/qgsjet-III-04-codes.dat
  DEPENDS GenParticlesHeaders
          ${input_dir}/code_generator.py
          ${input_dir}/qgsjet-III-04-codes.dat
          ${PROJECT_BINARY_DIR}/corsika/framework/core/particle_db.pkl
  WORKING_DIRECTORY ${output_dir}
  COMMENT "Generate conversion tables for particle codes QGSJetIII <-> CORSIKA"
  VERBATIM
  )

set_source_files_properties (
  ${output_dir}/Generated.inc
  PROPERTIES GENERATED TRUE
)

add_custom_target (SourceDirLinkQgsIII DEPENDS ${output_dir}/Generated.inc)
add_dependencies (CORSIKA8 SourceDirLinkQgsIII)

install (
  FILES ${output_dir}/Generated.inc
  DESTINATION include/corsika/modules/qgsjetIII
)
