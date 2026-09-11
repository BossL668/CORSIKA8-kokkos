/*
 * (c) Copyright 2021 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/output/ShowerSummary.hpp>

#include <corsika/output/BaseOutput.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/media/ShowerAxis.hpp>
#include <corsika/modules/writers/WriterOff.hpp>
#include <corsika/modules/writers/LongitudinalProfileWriterParquet.hpp>

#include <vector>
#include <array>

/**
 * @file ProductionWriter.hpp
 */

namespace corsika {

  /**
   * Local helper namespace to store number and names of particle number profile columns.
   */
  namespace production_profile {

    /**
     * Definition of longitudinal profile columns.
     */
    enum class ProjectileIndex {
      Pion,
      Kaon,
      Heavy,
      Hadron,
      Photon,
      ElectronPositron,
      Muon,
      All,
      Entries
    };

    /**
     * Number of columns (static).
     */
    size_t constexpr NColumns = static_cast<size_t>(ProjectileIndex::Entries);

    /**
     * Names of columns in output.
     */
    static std::array<char const*, NColumns> constexpr ProjectileIndexNames{
        {"pion", "kaon", "heavy", "hadron", "photon", "electron-positron", "muon",
         "all"}};

    /**
     * Data type to store column data.
     */
    typedef std::array<double, NColumns> ProfileData;
  } // namespace production_profile

  // clang-format-off
  /**
   * The ProductionWriter can be used to pool the particle counts of several
   * longitudinal profile processes into one output file/stream.
   *
   * Typically several processes/modules can lead to particle counts along the shower axis
   * in the shower. The LongitudianalWriter can be used in combination with the SubWriter
   * class to collect all of them into a single output stream:
   *
   * \code {.cpp}
   * # showerAxis must be a ShowerAxis object
   * # the X binning can be specified.
   * ProductionWriter profile{showerAxis, 10_g / square(1_cm), 200};
   * # add to OutputManager:
   * output.add("profile", profile);
   * # add SubWriters, e.g. LongitudinalProfile, CONEX:
   * LongitudinalProfile<SubWriter<decltype(profile)>> long{profile};
   * CONEXhybrid<SubWriter<decltype(profile)>> conex{..., profile};
   * ...
   * \endcode
   *
   * The default output option is parquet format.
   *
   * @tparam TOutput
   */
  // clang-format-on

  template <typename TOutput =
                LongitudinalProfileWriterParquet<production_profile::NColumns>>
  class ProductionWriter : public TOutput {

  public:
    /**
     * Construct a new writer.
     */
    ProductionWriter(ShowerAxis const& axis,
                     GrammageType dX = 10_g / square(1_cm)); // profile binning

    ProductionWriter(ShowerAxis const& axis, size_t nbins,
                     GrammageType dX = 10_g / square(1_cm));

    void startOfLibrary(boost::filesystem::path const& directory) final override;

    void startOfShower(unsigned int const showerId) final override;

    void endOfShower(unsigned int const showerId) final override;

    void endOfLibrary() final override;

    /**
     * Add continuous profile.
     */
    void write(Point const& p0, Code const pid, double const weight);

    /**
     * Add a production vertex whose ShowerAxis grammage is already known.
     *
     * This is the projected-coordinate counterpart of write().  It is used
     * by accelerated transports which evaluate the ShowerAxis projection on
     * the device and therefore must not reconstruct an approximate position
     * on the host.
     */
    void writeProjected(GrammageType grammage, Code pid, double weight);

    /**
     * Add a production vertex directly to an output bin.
     *
     * The projectile classification is deliberately shared with write() so
     * CPU and accelerator paths populate the same parent and "all" columns.
     */
    void addBin(size_t bin, Code pid, double weight);

    size_t getNBins() const noexcept { return nBins_; }

    /**
     * Return a summary.
     */
    YAML::Node getSummary() const override;

    void writeSummary(boost::filesystem::path const& path) const override {
      summary_.writeSummary(path);
    }

    /**
     * Return the configuration of this output.
     */
    YAML::Node getConfig() const override;

    production_profile::ProfileData const& getProfile(
        production_profile::ProjectileIndex index) const {
      return profile_.at(static_cast<int>(index));
    }

    production_profile::ProfileData const& getBin(size_t bin) const {
      return profile_.at(bin);
    }

  private:
    ShowerAxis const& showerAxis_; ///< conversion between geometry and grammage
    GrammageType const dX_;        ///< binning of profile.
    size_t const nBins_;           ///< number of profile bins.
    std::vector<production_profile::ProfileData> profile_; // longitudinal profile
    ShowerSummary summary_;
  };

} // namespace corsika

#include <corsika/detail/modules/writers/ProductionWriter.inl>
