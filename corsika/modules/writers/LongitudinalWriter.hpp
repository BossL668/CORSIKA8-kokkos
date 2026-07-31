/*
 * (c) Copyright 2021 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/output/BaseOutput.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/media/ShowerAxis.hpp>
#include <corsika/modules/writers/WriterOff.hpp>
#include <corsika/modules/writers/LongitudinalProfileWriterParquet.hpp>

#include <vector>
#include <array>

/**
 * @file LongitudinalWriter.hpp
 */

namespace corsika {

  /**
   * Local helper namespace to store number and names of particle number profile columns.
   */
  namespace number_profile {

    /**
     * Definition of longitudinal profile columns.
     */
    enum class ProfileIndex {
      Charged,
      Hadron,
      Photon,
      Electron,
      Positron,
      MuPlus,
      MuMinus,
      Entries
    };

    /**
     * Number of columns (static).
     */
    size_t constexpr NColumns = static_cast<size_t>(ProfileIndex::Entries);

    /**
     * Names of columns in output.
     */
    static std::array<char const*, NColumns> constexpr ProfileIndexNames{
        {"charged", "hadron", "photon", "electron", "positron", "muplus", "muminus"}};

    /**
     * Data type to store column data.
     */
    typedef std::array<double, NColumns> ProfileData;
  } // namespace number_profile

  // clang-format-off
  /**
   * The LongitudinalWriter can be used to pool the particle counts of several
   * longitudinal profile processes into one output file/stream.
   *
   * Typically several processes/modules can lead to particle counts along the shower axis
   * in the shower. The LongitudianalWriter can be used in combination with the SubWriter
   * class to collect all of them into a single output stream:
   *
   * \code {.cpp}
   * # showerAxis must be a ShowerAxis object
   * # the X binning can be specified.
   * LongitudinalWriter profile{showerAxis, 10_g / square(1_cm), 200};
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

  template <typename TOutput = LongitudinalProfileWriterParquet<number_profile::NColumns>>
  class LongitudinalWriter : public TOutput {

  public:
    /**
     * Construct a new writer.
     */
    LongitudinalWriter(ShowerAxis const& axis,
                       GrammageType dX = 10_g / square(1_cm)); // profile binning

    LongitudinalWriter(ShowerAxis const& axis, size_t nbins,
                       GrammageType dX = 10_g / square(1_cm));

    void startOfLibrary(boost::filesystem::path const& directory) final override;

    void startOfShower(unsigned int const showerId) final override;

    void endOfShower(unsigned int const showerId) final override;

    void endOfLibrary() final override;

    /**
     * Add continuous profile.
     */
    void write(Point const& p0, Point const& p1, Code const pid, double const weight);

    /**
     * Add a track whose endpoints have already been projected onto the
     * configured ShowerAxis.
     */
    void writeProjected(GrammageType const Xstart, GrammageType const Xend,
                        Code const pid, double const weight);

    /**
     * Add binned profile.
     */
    void write(GrammageType const Xstart, GrammageType const Xend, Code const pid,
               double const weight);

    /**
     * Merge an already accumulated contribution into one output bin.
     * This is used by the resident CUDA profile accumulator at end of shower.
     */
    void addBin(size_t bin, Code pid, double weight);

    size_t getNBins() const noexcept { return nBins_; }

    /**
     * Return a summary.
     */
    YAML::Node getSummary() const override;

    /**
     * Return the configuration of this output.
     */
    YAML::Node getConfig() const override;

    number_profile::ProfileData const& getProfile(
        number_profile::ProfileIndex index) const {
      return profile_.at(static_cast<int>(index));
    }

  private:
    ShowerAxis const& showerAxis_; ///< conversion between geometry and grammage
    GrammageType const dX_;        ///< binning of profile.
    size_t const nBins_;           ///< number of profile bins.
    std::vector<number_profile::ProfileData> profile_; // longitudinal profile
  };

} // namespace corsika

#include <corsika/detail/modules/writers/LongitudinalWriter.inl>
