/*
 * (c) Copyright 2021 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <chrono>
#include <string>
#include <boost/filesystem.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/output/BaseOutput.hpp>
#include <corsika/output/YAMLStreamer.hpp>

namespace corsika {

  /*!
   * Manages CORSIKA 8 output streams.
   */
  class OutputManager : public YAMLStreamer {

    /**
     * Indicates the current state of this manager.
     */
    enum class OutputState {
      NoInit,
      LibraryReady,
      ShowerInProgress,
      LibraryFinished,
    };

  public:
    /**
     * Construct an OutputManager instance with a name in a given directory.
     *
     * @param dir_path    The name of this output collection.
     * @param vseed       The seed for the simulation (written to summary file)
     * @param input_args  The command line arguments at runtime (written to summary file)
     */
    OutputManager(std::string const& dir_path, const long& vseed,
                  std::string const& input_args, bool useCompression);

    /**
     * Handle graceful closure of the outputs upon destruction.
     */
    ~OutputManager();

    template <typename TOutput>
    void add(std::string const& name, TOutput& output);

    /**
     * Produces the summary YAML.
     *
     * @return YAML::Node
     */
    YAML::Node getSummary() const;

    /**
     * Produces the config YAML.
     *
     * @return YAML::Node
     */
    YAML::Node getConfig() const;

  private:
    /**
     * Write the top-level config of this simulation.
     */
    void writeConfig() const;

    /**
     * Write the top-level summary of this library.
     */
    void writeSummary() const;

  public:
    /**
     * Called at the start of each library.
     */
    void startOfLibrary();

    /**
     * Called at the start of each event/shower.
     * This iteratively calls startOfEvent on each registered output.
     */
    void startOfShower();

    /**
     * Called at the end of each event/shower.
     * This iteratively calls endOfEvent on each registered output.
     */
    void endOfShower();

    /**
     * Called at the end of each library.
     * This iteratively calls endOfLibrary on each registered output.
     */
    void endOfLibrary();

    /**
     * Return current event number.
     */
    int getEventId() const;

    /**
     * True while startOfShower() has not yet been paired with
     * endOfShower(). This is primarily used by strict accelerator failure
     * handling to close an explicitly incomplete event safely.
     */
    bool showerInProgress() const noexcept;

  private:
    boost::filesystem::path root_;           ///< The unique output directory.
    OutputState state_{OutputState::NoInit}; ///< The current state of this manager.
    std::string const cmnd_line_args_; ///< The command line arguments used in this run
    bool useCompression_;              ///< compress the files to tarball
    int count_{0};                     ///< The current ID of this shower.
    long seed_{0};
    std::chrono::time_point<std::chrono::system_clock> const start_time{
        std::chrono::system_clock::now()}; ///< The time the manager is created.
    inline static auto logger_{get_logger("output")}; ///< A custom logger.
    /**
     * The outputs that have been registered here.
     */
    std::map<std::string, std::reference_wrapper<BaseOutput>> outputs_;

  }; // class OutputManager

} // namespace corsika

#include <corsika/detail/output/OutputManager.inl>
