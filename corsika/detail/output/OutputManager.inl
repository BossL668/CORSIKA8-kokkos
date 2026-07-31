/*
 * (c) Copyright 2021 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <algorithm>
#include <fstream>
#include <functional>

#include <iomanip>
#include <ctime>
#include <sstream>

#include <boost/filesystem.hpp>

#include <fmt/core.h>
#include <fmt/chrono.h>

namespace corsika {

  inline OutputManager::OutputManager(std::string const& dir_path, const long& vseed = 0,
                                      std::string const& vargs = "",
                                      bool useCompression = false)
      : root_(dir_path)
      , cmnd_line_args_(vargs)
      , useCompression_(useCompression)
      , count_(0)
      , seed_(vseed) {

    // check if this directory already exists
    if (boost::filesystem::exists(root_)) {
      CORSIKA_LOGGER_ERROR(logger_,
                           "Output directory '{}' already exists! Do not overwrite!.",
                           root_.string());
      throw std::runtime_error("Output directory already exists.");
    }

    // construct the directory for this library
    boost::filesystem::create_directories(root_);

    CORSIKA_LOGGER_INFO(logger_, fmt::format("Output library: \"{}\"", root_.string()));
    writeYAML(getConfig(), root_ / ("config.yaml"));
  }

  template <typename TOutput>
  inline void OutputManager::add(std::string const& name, TOutput& output) {

    if (state_ != OutputState::NoInit) {
      // if "add" is called after the ouptput has started, this is an ERROR.
      CORSIKA_LOGGER_ERROR(
          logger_, "Cannot add more outputs to OutputManager after output was started.");
      throw std::runtime_error(
          "Cannot add more outputs to OutputManager after output was started.");
    }

    // check if that name is already in the map
    if (outputs_.count(name) > 0) {
      CORSIKA_LOGGER_ERROR(
          logger_, "'{}' is already registered. All outputs must have unique names.",
          name);
      throw std::runtime_error("Output already exists. Do not overwrite!");
    }

    // if we get here, the name is not already in the map
    // so we create the output and register it into the map
    outputs_.insert(std::make_pair(name, std::ref(output)));

    // create the directory for this process.
    boost::filesystem::create_directory(root_ / name);
  }

  inline OutputManager::~OutputManager() {

    if (state_ == OutputState::ShowerInProgress) {
      // if this the destructor is called before the shower has been explicitly
      // ended, print a warning and end the shower before continuing.
      CORSIKA_LOGGER_WARN(logger_,
                          "OutputManager was destroyed before endOfShower() called."
                          " The last shower in this libray may be incomplete.");
      endOfShower();
    }

    // write the top level summary file (summary.yaml)
    writeSummary();

    // if we are being destructed but EndOfLibrary() has not been called,
    // make sure that we gracefully close all the outputs. This is a supported
    // method of operation so we don't issue a warning here
    if (state_ == OutputState::LibraryReady) { endOfLibrary(); }

    if (useCompression_) {
      auto const parent = root_.parent_path();
      auto const final_part = root_.filename();

      // Ensure we use "./" if the parent path is empty
      std::string parent_path = parent.empty() ? "./" : parent.string() + "/";

      std::string const cmd = "tar cf " + parent_path + final_part.string() + ".tar -C " +
                              parent_path + " " + final_part.string();
      CORSIKA_LOG_INFO("Compressing output directory using: {}", cmd.c_str());
      int const returnCode = std::system(cmd.c_str());

      if (returnCode) {
        CORSIKA_LOG_ERROR("Compression returned with error code {}", returnCode);
      } else {
        // remove the original directory
        boost::filesystem::remove_all(root_);
      }
    }
  }

  inline int OutputManager::getEventId() const { return count_; }

  inline bool OutputManager::showerInProgress() const noexcept {
    return state_ == OutputState::ShowerInProgress;
  }

  inline YAML::Node OutputManager::getConfig() const {

    YAML::Node config;

    // some basic info
    config["path"] = root_.string();  // the simulation name
    config["creator"] = "CORSIKA8";   // a tag to identify C8 libraries
    config["version"] = "8.0.1";      // the current version
    config["args"] = cmnd_line_args_; // the command line parameters
    return config;
  }

  inline YAML::Node OutputManager::getSummary() const {

    YAML::Node summary;

    // the total number of showers contained in the library
    summary["showers"] = count_;

    summary["seed"] = seed_;

    // this next section handles writing some time and duration information

    // create a quick lambda function to convert a time-instance to a string
    auto timeToString = [&](auto const time) -> std::string {
      // ISO 8601 time format
      auto format{"%FT%T%z"};

      // convert the clock to a time_t
      auto time_tc{std::chrono::system_clock::to_time_t(time)};

      // create the string and push the time onto it
      std::ostringstream oss;
      oss << std::put_time(std::localtime(&time_tc), format);

      return oss.str();
    };

    auto end_time{std::chrono::system_clock::now()};

    // calculate the number of days
    auto const start_t = std::chrono::system_clock::to_time_t(start_time);
    auto const end_t = std::chrono::system_clock::to_time_t(end_time);
    int const durationDays = std::difftime(end_t, start_t) / (60 * 60 * 24);

    // add the time and duration info
    summary["start time"] = timeToString(start_time);
    summary["end time"] = timeToString(end_time);
    summary["runtime"] = (durationDays ? fmt::format("+{}d ", durationDays) : "") +
                         fmt::format("{:%H:%M:%S}", end_time - start_time);
    summary["runtime_raw"] =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)
            .count();

    std::vector<std::string> output_dirs;
    for (auto const& outs : outputs_) { output_dirs.push_back(outs.first); }
    summary["output_dirs"] = output_dirs;

    return summary;
  }

  inline void OutputManager::writeSummary() const {

    // write the node to a file
    writeYAML(getSummary(), root_ / ("summary.yaml"));
  }

  inline void OutputManager::startOfLibrary() {

    // this is only valid when we haven't started a library
    // or have already finished a library
    if (!(state_ == OutputState::NoInit || state_ == OutputState::LibraryFinished)) {

      throw std::runtime_error("startOfLibrary() called in invalid state.");
    }

    // we now forward this signal to all of our outputs
    for (auto& [name, output] : outputs_) {

      // and start the library
      output.get().startOfLibrary(root_ / name);

      // get the config from this output
      auto config = output.get().getConfig();

      // add the name keyword
      config["name"] = name;

      // write the output configuration to config.yaml in the output directory
      writeYAML(config, root_ / name / ("config.yaml"));
    }

    // we have now started running
    state_ = OutputState::LibraryReady;
    count_ = 0; // event counter
  }

  inline void OutputManager::startOfShower() {

    // if this is called and we still in the "no init" state, then
    // this is the first shower in the library so make sure we start it
    if (state_ == OutputState::NoInit) { startOfLibrary(); }

    // now start the event for all the outputs
    for (auto& [name, output] : outputs_) { output.get().startOfShower(count_); }

    // and transition to the in progress state
    state_ = OutputState::ShowerInProgress;
  }

  inline void OutputManager::endOfShower() {

    for (auto& [name, output] : outputs_) { output.get().endOfShower(count_); }

    // switch back to the initialized state
    state_ = OutputState::LibraryReady;

    // increment our shower count
    ++count_;
  }

  inline void OutputManager::endOfLibrary() {

    // we can only call endOfLibrary when we have already started
    if (state_ == OutputState::NoInit) {
      throw std::runtime_error("endOfLibrary() called in invalid state.");
    }

    // write the summary for each output and forward the endOfLibrary call()
    for (auto& [name, output] : outputs_) {
      // save eventual YAML summary
      YAML::Node const summary = output.get().getSummary();
      if (!summary.IsNull()) {
        writeYAML(output.get().getSummary(), root_ / name / ("summary.yaml"));
      }

      // and forward the end of library call
      output.get().endOfLibrary();
    }

    // and the library has finished
    state_ = OutputState::LibraryFinished;
  }
} // namespace corsika
