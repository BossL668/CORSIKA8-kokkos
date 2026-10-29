/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#ifdef PRODUCTION_REFERENCE
#include "applications/detail/air_shower_multigpu/NativeCoordinator.hpp"
namespace tested = c8::multigpu;
using options_type = tested::Options;
#define OPTION(name) name
#define WORKER_SEED workerSeed
#define JOIN_FRONTIERS joinFrontiers
#define PARTITION_RADIO partitionRadioTracks
#define VALIDATE_OUTPUTS validateOutputs
#define MAIN_ENTRY mainEntry
#else
#include <corsika/kokkos/scheduling/AirShowerCoordinator.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <spdlog/sinks/ostream_sink.h>
namespace tested = corsika::kokkos::coordinator;
using options_type = tested::CoordinatorOptions;
#define OPTION(name) name##_
#define WORKER_SEED worker_seed
#define JOIN_FRONTIERS join_frontiers
#define PARTITION_RADIO partition_radio_tracks
#define VALIDATE_OUTPUTS validate_outputs
#define MAIN_ENTRY main_entry
#endif

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using json_type = nlohmann::json;

namespace {
  void check(arrow::Status const& status) {
    if (!status.ok()) throw std::runtime_error(status.ToString());
  }
  template <typename T>
  T take(arrow::Result<T> result) {
    check(result.status());
    return std::move(result).ValueOrDie();
  }
  std::shared_ptr<arrow::Array> array(std::vector<double> const& values) {
    arrow::DoubleBuilder builder;
    check(builder.AppendValues(values));
    return take(builder.Finish());
  }
  void write_table(fs::path const& path, std::string const& grid, double value,
                   double shift, bool records) {
    fs::create_directories(path.parent_path());
    auto const schema = arrow::schema({arrow::field("shower", arrow::float64()),
                                       arrow::field(grid, arrow::float64()),
                                       arrow::field("value", arrow::float64())});
    auto const table =
        records ? arrow::Table::Make(schema, {array({0}), array({shift}), array({value})})
                : arrow::Table::Make(schema, {array({0, 0}), array({shift, 1 + shift}),
                                              array({value, 2 * value})});
    auto stream = take(arrow::io::FileOutputStream::Open(path.string()));
    check(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), stream, 64));
    check(stream->Close());
    std::ofstream(path.parent_path() / "config.yaml") << "fixture: fixed-grid\n";
  }
  void write_outputs(json_type const& input) {
    fs::path root = input.at("root").get<std::string>();
    double value = input.at("value"), shift = input.value("shift", 0.);
    for (auto const* rel :
         {"profile/profile.parquet", "production_profile/profile.parquet",
          "energyloss/dEdX.parquet", "CoREAS/observers.parquet",
          "ZHS/observers.parquet"}) {
      bool radio = std::string(rel).find("observers") != std::string::npos;
      write_table(root / rel, radio ? "Time" : "X", value, shift, false);
    }
    for (auto const* rel :
         {"particles/particles.parquet", "interactions/interactions.parquet"})
      write_table(root / rel, "E", value, shift, true);
    fs::create_directories(root / "primary");
    std::ofstream(root / "primary/config.yaml") << "fixture: protocol-only\n";
    std::ofstream(root / "primary/summary.yaml") << "primary: fixed\n";
    std::ofstream(root / "interactions/summary.yaml") << "first_interaction: fixed\n";
    if (input.value("firn", false)) {
      fs::remove_all(root / "CoREAS");
      fs::remove_all(root / "ZHS");
      fs::create_directory(root / "firn_radio");
      for (auto const* name : {"radio_waveforms.parquet", "radio_spectra.parquet"}) {
        std::string const grid = std::string(name).find("spectra") != std::string::npos
                                     ? "frequency_hz"
                                     : "time_s";
        auto schema = arrow::schema({arrow::field("shower", arrow::float64()),
                                     arrow::field("observer", arrow::float64()),
                                     arrow::field(grid, arrow::float64()),
                                     arrow::field("field", arrow::float64())});
        double const observerShift = input.value("observer_shift", 0.);
        auto data = arrow::Table::Make(
            schema, {array({0, 0, 0, 0}), array({observerShift, observerShift, 1, 1}),
                     array({0, 1, 0, 1}), array({value, 2 * value, -value, -2 * value})});
        auto stream = take(
            arrow::io::FileOutputStream::Open((root / "firn_radio" / name).string()));
        check(
            parquet::arrow::WriteTable(*data, arrow::default_memory_pool(), stream, 64));
        check(stream->Close());
      }
      std::ofstream(root / "firn_radio/config.yaml")
          << "profile_fingerprint: fixed\nexecution: " << input.value("execution", "CPU")
          << "\ninclude_surface_reflection: " << input.value("reflection", true) << '\n';
    }
  }
  json_type read_table(fs::path const& path) {
    auto input = take(arrow::io::ReadableFile::Open(path.string()));
    std::unique_ptr<parquet::arrow::FileReader> reader;
    check(parquet::arrow::OpenFile(input, arrow::default_memory_pool(), &reader));
    std::shared_ptr<arrow::Table> table;
    check(reader->ReadTable(&table));
    check(table->ValidateFull());
    json_type output = json_type::object();
    for (int column = 0; column < table->num_columns(); ++column) {
      auto& values = output[table->field(column)->name()];
      values = json_type::array();
      for (auto const& chunk : table->column(column)->chunks())
        for (auto value : *std::static_pointer_cast<arrow::DoubleArray>(chunk))
          values.push_back(*value);
    }
    return output;
  }
  std::vector<fs::path> paths(json_type const& input) {
    std::vector<fs::path> result;
    for (auto const& item : input) result.emplace_back(item.get<std::string>());
    return result;
  }
  json_type dispatch(json_type const& input) {
    auto const operation = input.at("operation").get<std::string>();
    if (operation == "defaults") {
      options_type options;
      return {{"backend", options.OPTION(emBackend)},
              {"egs4_stepfc", options.OPTION(egs4Stepfc)},
              {"seed", options.OPTION(seed)},
              {"events", options.OPTION(events)},
              {"prefix_workers", options.OPTION(prefixWorkers)},
              {"split_energy", options.OPTION(prefixSplitEnergy)},
              {"energy", options.OPTION(energy)},
              {"memory", options.OPTION(memoryFraction)},
              {"frontier_energy", options.OPTION(frontierEnergy)},
              {"timeout", options.OPTION(timeout)},
              {"worker", options.OPTION(worker).string()},
              {"output", options.OPTION(output).string()},
              {"devices", options.OPTION(devices)},
              {"physics", options.OPTION(physics)}};
    }
    if (operation == "seed")
      return tested::WORKER_SEED(input.at("seed").get<std::uint64_t>(),
                                 input.at("index").get<unsigned>());
    if (operation == "requested") {
      auto storage = input.at("args").get<std::vector<std::string>>();
      std::vector<char*> args;
      for (auto& text : storage) args.push_back(text.data());
      return tested::requested(args.size(), args.data());
    }
    if (operation == "digest") return tested::digest(input.at("path").get<std::string>());
    if (operation == "partition")
      return tested::partition(input.at("source").get<std::string>(),
                               paths(input.at("destinations")));
    if (operation == "join")
      return tested::JOIN_FRONTIERS(paths(input.at("sources")),
                                    input.at("destination").get<std::string>());
    if (operation == "radio")
      return tested::PARTITION_RADIO(paths(input.at("sources")),
                                     input.at("workers").get<std::vector<unsigned>>());
    if (operation == "save") {
      tested::save(input.at("path").get<std::string>(), input.at("value"));
      return true;
    }
    if (operation == "validate") {
      tested::VALIDATE_OUTPUTS(input.at("root").get<std::string>());
      return true;
    }
#ifndef PRODUCTION_REFERENCE
    if (operation == "validate_firn_replay") {
      tested::validate_radio_replay(input.at("root").get<std::string>(),
                                    input.at("expected").get<std::uint64_t>());
      return true;
    }
#endif
    if (operation == "merge")
      return tested::merge(paths(input.at("parts")),
                           input.at("destination").get<std::string>());
    if (operation == "fixture") {
      write_outputs(input);
      return true;
    }
    if (operation == "read") return read_table(input.at("path").get<std::string>());
    throw std::runtime_error("Unknown fixture operation");
  }
} // namespace

int main(int argc, char** argv) {
#ifndef PRODUCTION_REFERENCE
  // Keep the frozen CLI payload comparison independent of logger decoration.
  auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(std::cerr);
  sink->set_pattern("%v");
  corsika::get_logger("corsika", true)->sinks() = {sink};
#endif
  if (argc > 1 && std::string(argv[1]) == "entry")
    return tested::MAIN_ENTRY(argc - 1, argv + 1);
  try {
    json_type input;
    std::cin >> input;
    auto output = dispatch(input);
    std::cout << json_type{{"ok", true}, {"value", output}}.dump() << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cout << json_type{{"ok", false}, {"error", error.what()}}.dump() << '\n';
    return 2;
  }
}
