# Reopening a writer in a new, independent event directory must rebuild its
# schema, not append columns from the previous library. Experiment only.
set(persistent_output_relative corsika/detail/output/ParquetStreamer.inl)
file(READ "${EGS4_C8_SOURCE}/${persistent_output_relative}" persistent_output_source)
set(persistent_output_anchor [=[  inline void ParquetStreamer::initStreamer(std::string const& filepath) {
]=])
string(FIND "${persistent_output_source}" "${persistent_output_anchor}" persistent_output_found)
if(persistent_output_found LESS 0)
  message(FATAL_ERROR "Parquet lifecycle source changed: re-audit isolated overlay")
endif()
string(REPLACE "${persistent_output_anchor}" "${persistent_output_anchor}
    if (isInit_) throw std::logic_error(\"Close previous event writer before reopening\");
    fields_.clear();
    schema_.reset();
" persistent_output_source "${persistent_output_source}")
set(persistent_output_root "${CMAKE_CURRENT_BINARY_DIR}/persistent_output_overlay")
file(MAKE_DIRECTORY "${persistent_output_root}/corsika/detail/output")
file(WRITE "${persistent_output_root}/${persistent_output_relative}" "${persistent_output_source}")
target_include_directories(c8_egs4_shower BEFORE PRIVATE "${persistent_output_root}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${EGS4_C8_SOURCE}/${persistent_output_relative}")
