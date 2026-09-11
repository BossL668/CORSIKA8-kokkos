/*
 * (c) Copyright 2021 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/framework/core/Logging.hpp>
#include <corsika/output/Configurable.hpp>
#include <corsika/output/YAMLStreamer.hpp>
#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>

namespace corsika {

  /**
   * This is the base class for all outputs so that they
   * can be stored in homogeneous containers.
   */
  class BaseOutput : public Configurable {

  protected:
    BaseOutput() = default;
    virtual ~BaseOutput() = default;

  public:
    /**
     * Called at the start of each run.
     */
    virtual void startOfLibrary(boost::filesystem::path const& directory) = 0;

    /**
     * Called at the start of each event/shower.
     *
     * @param showerId Shower counter.
     */
    virtual void startOfShower(unsigned int const /*showerId*/) {}

    /**
     * Called at the end of each event/shower.
     *
     * @param showerId Shower counter.
     */
    virtual void endOfShower(unsigned int const showerId) = 0; // LCOV_EXCL_LINE

    /**
     * Called at the end of each run.
     */
    virtual void endOfLibrary() = 0; // LCOV_EXCL_LINE

    /**
     * Flag to indicate readiness.
     */
    bool isInit() const { return is_init_; }

    /**
     * The output logger.
     */
    static auto getLogger() { return logger_; }

    /**
     * Provide YAML Summary for this BaseOutput.
     */
    virtual YAML::Node getSummary() const { return YAML::Node(); }

    /** Serialize summaries without requiring a whole-library YAML graph. */
    virtual void writeSummary(boost::filesystem::path const& path) const {
      auto const summary = getSummary();
      if (!summary.IsNull()) YAMLStreamer{}.writeYAML(summary, path);
    }

  protected:
    /**
     * Set init flag.
     */
    void setInit(bool const v) { is_init_ = v; }

  private:
    bool is_init_{false};                             ///< flag to indicate readiness
    inline static auto logger_{get_logger("output")}; ///< A custom logger.
  };

} // namespace corsika
