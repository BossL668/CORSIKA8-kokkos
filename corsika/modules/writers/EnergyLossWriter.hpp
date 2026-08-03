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
#include <corsika/modules/writers/EnergyLossWriterParquet.hpp>

#include <vector>
#include <array>

/**
 * @file EnergyLossWriter.hpp
 */

namespace corsika {

  // clang-format-off
  /**
   * The energy loss writer can be used to pool several energy loss processes into one
   * output file/stream.
   *
   * Typically many processes/modules can lead to energy losses in the shower. The
   * EnergyLossWriter can be used in combination with the SubWriter class to collect all
   * of them into a single output stream:
   *
   * \code {.cpp}
   * # showerAxis must be a ShowerAxis object
   * # the X binning can be specified.
   * EnergyLossWriter dEdX{showerAxis, 10_g / square(1_cm), 200};
   * # add to OutputManager:
   * output.add("energyloss", dEdX);
   * # add SubWriters, e.g. Bethe-Bloch:
   * BetheBlochPDG<SubWriter<decltype(dEdX)>> eLoss{dEdX};
   * ...
   * \endcode
   *
   * The EnergyLossWriter processes data on single-particle-level. The final output
   * writer, e.g. EnergyLossWriterParquet, processes data on profile-level (bins in X).
   * The default output option is parquet format.
   *
   * @tparam TOutput
   */
  // clang-format-on

  /**
   * Local helper namespace to store number and names of dEdX profile columns.
   */
  namespace dEdX_output {

    /**
     * Definition of longitudinal profile columns.
     */
    enum class ProfileIndex { Total, Entries };

    /**
     * Number of columns (static).
     */
    size_t constexpr NColumns = static_cast<int>(ProfileIndex::Entries);

    /**
     * Names of columns in output.
     */
    static std::array<char const*, NColumns> constexpr ProfileIndexNames{{"total"}};

    /**
     * Data type to store column data.
     */
    typedef std::array<HEPEnergyType, NColumns> Profile;

  } // namespace dEdX_output

  /**
   * The EnergyLossWriter can be used to pool the dEdX energy loss of several
   * processes/modules into one output file/stream.
   *
   * Typically several processes/modules can lead to energy losses along the shower axis
   * in the shower. The EnergyLossWriter can be used in combination with the SubWriter
   * class to collect all of them into a single output stream:
   *
   * \code {.cpp}
   * # showerAxis must be a ShowerAxis object
   * # the X binning can be specified.
   * EnergyLossWriter dEdX{showerAxis, 10_g / square(1_cm), 200};
   * # add to OutputManager:
   * output.add("energyloss", dEdX);
   * # add SubWriters, e.g. BetheBlochPDG, CONEX:
   * BetheBlochPDG<SubWriter<decltype(dEdX)>> long{dEdX};
   * CONEXhybrid<SubWriter<decltype(dEdX)>> conex{..., dEdX};
   * ...
   * \endcode
   *
   * The default output option is parquet format.
   *
   * @tparam TOutput
   */

  template <typename TOutput = EnergyLossWriterParquet<dEdX_output::NColumns>>
  class EnergyLossWriter : public TOutput {

  public:
    /**
     * Construct a new writer.
     */

    // Number of bins defined explicitly
    EnergyLossWriter(ShowerAxis const& axis,
                     GrammageType dX = 10_g / square(1_cm), // profile binning
                     GrammageType dX_threshold = 0.0001_g /
                                                 square(1_cm)); // ignore too short tracks

    // Number of bins defined explicitly
    EnergyLossWriter(ShowerAxis const& axis,
                     unsigned int const nBins,              // number of bins
                     GrammageType dX = 10_g / square(1_cm), // profile binning
                     GrammageType dX_threshold = 0.0001_g /
                                                 square(1_cm)); // ignore too short tracks

    void startOfLibrary(boost::filesystem::path const& directory) final override;

    void startOfShower(unsigned int const showerId) final override;

    void endOfShower(unsigned int const showerId) final override;

    void endOfLibrary() final override;

    /**
     * Add continuous energy loss.
     */
    void write(Point const& p0, Point const& p1, Code const PID, HEPEnergyType const dE);

    /**
     * Add continuous loss for endpoints already projected onto ShowerAxis.
     */
    void writeProjected(GrammageType Xstart, GrammageType Xend, Code const PID,
                        HEPEnergyType const dE);

    /**
     * Add localized energy loss.
     */
    void write(Point const& point, Code const PID, HEPEnergyType const dE);

    /**
     * Add binned energy loss.
     */
    void write(GrammageType const Xstart, GrammageType const Xend, Code const PID,
               HEPEnergyType const dE);

    /**
     * Merge one already accumulated longitudinal energy-loss bin.
     */
    void addBin(size_t bin, HEPEnergyType dE);

    /**
     * Merge one already accumulated electromagnetic energy-loss bin.
     *
     * The CUDA resident EM cascade uses this entry point because its profile
     * bins no longer carry an individual particle PID when they are copied
     * back to the host.
     */
    void addElectromagneticBin(size_t bin, HEPEnergyType dE);

    auto GetNBins() const { return nBins_; }

    /**
     * Get total observed energy loss.
     *
     * @return HEPEnergyType The total energy.
     */
    HEPEnergyType getEnergyLost() const;

    /**
     * Return a summary.
     */
    YAML::Node getSummary() const override;

    /**
     * Return the configuration of this output.
     */
    YAML::Node getConfig() const override;

  private:
    ShowerAxis const& showerAxis_; ///< conversion between geometry and grammage
    GrammageType dX_;              ///< binning of profile.
    size_t nBins_;                 ///< number of profile bins.
    GrammageType dX_threshold_;    ///< too short tracks are discarded.
    std::vector<dEdX_output::Profile> profile_; // longitudinal profile
    HEPEnergyType electromagneticEnergyLost_{HEPEnergyType::zero()};
    YAML::Node summary_;
  }; // namespace corsika

} // namespace corsika

#include <corsika/detail/modules/writers/EnergyLossWriter.inl>
