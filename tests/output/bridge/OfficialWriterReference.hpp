/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#ifndef CORSIKA_OUTPUT_BRIDGE_NATIVE_ONLY
#include <corsika/kokkos/output/CorsikaOutputSink.hpp>
#include <corsika/kokkos/output/EmRunOutput.hpp>
#endif
#include <corsika/modules/writers/LongitudinalWriter.hpp>
#include <corsika/modules/writers/EnergyLossWriter.hpp>
#include <corsika/modules/writers/ProductionWriter.hpp>
#include <corsika/modules/ObservationPlane.hpp>
#include <corsika/modules/writers/InteractionWriter.hpp>

namespace corsika::official_output_reference {

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
     * Add localized energy loss.
     */
    void write(Point const& point, Code const PID, HEPEnergyType const dE);

    /**
     * Add binned energy loss.
     */
    void write(GrammageType const Xstart, GrammageType const Xend, Code const PID,
               HEPEnergyType const dE);

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
    YAML::Node summary_;
  }; // namespace corsika

  template <typename TOutput>
  inline EnergyLossWriter<TOutput>::EnergyLossWriter(ShowerAxis const& axis,
                                                     GrammageType dX,
                                                     GrammageType dX_threshold)
      : EnergyLossWriter<TOutput>{axis,
                                  static_cast<unsigned int>(axis.getMaximumX() / dX) + 1,
                                  dX, dX_threshold} {}

  template <typename TOutput>
  inline EnergyLossWriter<TOutput>::EnergyLossWriter(ShowerAxis const& axis,
                                                     unsigned int const nBins,
                                                     GrammageType dX,
                                                     GrammageType dX_threshold)
      : TOutput(dEdX_output::ProfileIndexNames)
      , showerAxis_(axis)
      , dX_(dX)
      , nBins_(nBins)
      , dX_threshold_(dX_threshold) {}

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::startOfLibrary(
      boost::filesystem::path const& directory) {
    TOutput::startOfLibrary(directory);
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::startOfShower(unsigned int const showerId) {
    TOutput::startOfShower(showerId);
    // reset profile
    profile_.clear();
    profile_.resize(nBins_);
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::endOfLibrary() {
    TOutput::endOfLibrary();
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::endOfShower(unsigned int const showerId) {

    // determined Xmax and dEdXmax from quadratic interpolation
    double maximum = 0;
    size_t iMaximum = 0;
    for (size_t i = 0; i < profile_.size() - 3; ++i) {
      double value =
          (profile_[i + 0].at(static_cast<int>(dEdX_output::ProfileIndex::Total)) +
           profile_[i + 1].at(static_cast<int>(dEdX_output::ProfileIndex::Total)) +
           profile_[i + 2].at(static_cast<int>(dEdX_output::ProfileIndex::Total))) /
          1_GeV;
      if (value > maximum) {
        maximum = value;
        iMaximum = i;
      }
    }

    double const dX = dX_ / 1_g * square(1_cm);

    auto [Xmax, dEdXmax] = FindXmax::interpolateProfile(
        dX * (0.5 + iMaximum), dX * (1.5 + iMaximum), dX * (2.5 + iMaximum),
        profile_[iMaximum + 0].at(static_cast<int>(dEdX_output::ProfileIndex::Total)) /
            1_GeV,
        profile_[iMaximum + 1].at(static_cast<int>(dEdX_output::ProfileIndex::Total)) /
            1_GeV,
        profile_[iMaximum + 2].at(static_cast<int>(dEdX_output::ProfileIndex::Total)) /
            1_GeV);

    summary_["shower_" + std::to_string(showerId)]["sum_dEdX"] = getEnergyLost() / 1_GeV;
    summary_["shower_" + std::to_string(showerId)]["Xmax"] = Xmax;
    summary_["shower_" + std::to_string(showerId)]["dEdXmax"] = dEdXmax;

    int iRow{0};
    for (dEdX_output::Profile const& row : profile_) {
      // here: write to underlying writer (e.g. parquet)
      TOutput::write(showerId, iRow * dX_, row);
      iRow++;
    }

    TOutput::endOfShower(showerId);
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::write(Point const& p0, Point const& p1,
                                               Code const PID, HEPEnergyType const dE) {

    GrammageType grammageStart = showerAxis_.getProjectedX(p0);
    GrammageType grammageEnd = showerAxis_.getProjectedX(p1);

    if (grammageStart > grammageEnd) { // particle going upstream
      std::swap(grammageStart, grammageEnd);
    }

    GrammageType const deltaX = grammageEnd - grammageStart;

    CORSIKA_LOGGER_TRACE(
        TOutput::getLogger(),
        "dE={} GeV, grammageStart={} g/cm2, End={}g /cm2, deltaX={} g/cm2", dE / 1_GeV,
        grammageStart / 1_g * square(1_cm), grammageEnd / 1_g * square(1_cm),
        deltaX / 1_g * square(1_cm));

    if (deltaX < dX_threshold_) {
      CORSIKA_LOGGER_TRACE(TOutput::getLogger(), "Point-like dE");
      this->write(p0, PID, dE);
      return;
    }

    // only register the range that is covered by the profile
    int const maxBin = int(profile_.size() - 1);
    int binStart = grammageStart / dX_;
    if (binStart < 0) binStart = 0;
    if (binStart > maxBin) binStart = maxBin;
    int binEnd = grammageEnd / dX_;
    if (binEnd < 0) binEnd = 0;
    if (binEnd > maxBin) binEnd = maxBin;

    CORSIKA_LOGGER_TRACE(TOutput::getLogger(), "maxBin={}, binStart={}, binEnd={}",
                         maxBin, binStart, binEnd);

    auto energyCount = HEPEnergyType::zero();

    auto const factor = dE / deltaX; // [ energy / grammage ]
    auto fill = [&](int const bin, GrammageType const weight) {
      auto const increment = factor * weight;
      CORSIKA_LOGGER_TRACE(TOutput::getLogger(),
                           "filling bin={} with weight {} : dE={} GeV ", bin, weight,
                           increment / 1_GeV);
      profile_[bin][static_cast<int>(dEdX_output::ProfileIndex::Total)] += increment;
      energyCount += increment;
    };

    // fill longitudinal profile
    if (binStart == binEnd) {
      fill(binStart, deltaX);
    } else {
      fill(binStart, ((1 + binStart) * dX_ - grammageStart));
      fill(binEnd, (grammageEnd - binEnd * dX_));
      for (int bin = binStart + 1; bin < binEnd; ++bin) { fill(bin, dX_); }
    }

    CORSIKA_LOGGER_TRACE(TOutput::getLogger(), "total energy added to histogram: {} GeV ",
                         energyCount / 1_GeV);
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::write(Point const& point, Code const,
                                               HEPEnergyType const dE) {
    GrammageType grammage = showerAxis_.getProjectedX(point);
    int const maxBin = int(profile_.size() - 1);
    int bin = grammage / dX_;
    if (bin < 0) bin = 0;
    if (bin > maxBin) bin = maxBin;

    CORSIKA_LOGGER_TRACE(TOutput::getLogger(), "add local energy loss bin={} dE={} GeV ",
                         bin, dE / 1_GeV);

    profile_[bin][static_cast<int>(dEdX_output::ProfileIndex::Total)] += dE;
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::write(GrammageType const Xstart,
                                               GrammageType const Xend, Code const,
                                               HEPEnergyType const dE) {
    double const bstart = Xstart / dX_;
    double const bend = Xend / dX_;

    if (abs(bstart - floor(bstart + 0.5)) > 1e-2 ||
        abs(bend - floor(bend + 0.5)) > 1e-2 || abs(bend - bstart - 1) > 1e-2) {
      CORSIKA_LOGGER_ERROR(
          TOutput::getLogger(),
          "CascadeEquation (CONEX) and Corsika8 dX grammage binning are not the same! "
          "Xstart={} Xend={} dX={} g/cm2",
          Xstart / 1_g * square(1_cm), Xend / 1_g * square(1_cm),
          dX_ / 1_g * square(1_cm));
      throw std::runtime_error(
          "CONEX and Corsika8 dX grammage binning are not the same!");
    }

    size_t const bin = size_t((bend + bstart) / 2);
    CORSIKA_LOGGER_TRACE(TOutput::getLogger(),
                         "add binned energy loss {} {} bin={} dE={} GeV ", bstart, bend,
                         bin, dE / 1_GeV);
    if (bin >= profile_.size()) {
      CORSIKA_LOGGER_WARN(TOutput::getLogger(),
                          "Grammage bin {} outside of profile {}. skipping.", bin,
                          profile_.size());
      return;
    }
    profile_[bin][static_cast<int>(dEdX_output::ProfileIndex::Total)] += dE;
  }

  template <typename TOutput>
  inline HEPEnergyType EnergyLossWriter<TOutput>::getEnergyLost() const {
    HEPEnergyType tot = HEPEnergyType::zero();
    for (dEdX_output::Profile const& row : profile_)
      tot += row.at(static_cast<int>(dEdX_output::ProfileIndex::Total));
    return tot;
  }

  template <typename TOutput>
  inline YAML::Node EnergyLossWriter<TOutput>::getConfig() const {

    YAML::Node node;

    node["type"] = "EnergyLoss";
    node["units"]["energy"] = "GeV";
    node["units"]["grammage"] = "g/cm^2";
    node["bin-size"] = dX_ / (1_g / square(1_cm));
    node["nbins"] = nBins_;
    node["grammage_threshold"] = dX_threshold_ / (1_g / square(1_cm));

    return node;
  }

  template <typename TOutput>
  inline YAML::Node EnergyLossWriter<TOutput>::getSummary() const {

    return summary_;
  }

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
     * Add binned profile.
     */
    void write(GrammageType const Xstart, GrammageType const Xend, Code const pid,
               double const weight);

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

  template <typename TOutput>
  inline LongitudinalWriter<TOutput>::LongitudinalWriter(ShowerAxis const& axis,
                                                         GrammageType dX)
      : LongitudinalWriter<TOutput>{
            axis, static_cast<unsigned int>(axis.getMaximumX() / dX) + 1, dX} {}

  template <typename TOutput>
  inline LongitudinalWriter<TOutput>::LongitudinalWriter(ShowerAxis const& axis,
                                                         size_t nbins, GrammageType dX)
      : TOutput(number_profile::ProfileIndexNames)
      , showerAxis_(axis)
      , dX_(dX)
      , nBins_(nbins)
      , profile_{nbins} {}

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::startOfLibrary(
      boost::filesystem::path const& directory) {
    TOutput::startOfLibrary(directory);
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::startOfShower(unsigned int const showerId) {
    profile_.clear();
    for (size_t i = 0; i < nBins_; ++i) { profile_.emplace_back(); }
    TOutput::startOfShower(showerId);
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::endOfLibrary() {
    TOutput::endOfLibrary();
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::endOfShower(unsigned int const showerId) {

    int iRow{0};
    for (number_profile::ProfileData const& row : profile_) {
      // here: write to underlying writer (e.g. parquet)
      TOutput::write(showerId, iRow * dX_, row);
      iRow++;
    }
    TOutput::endOfShower(showerId);
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::write(Point const& p0, Point const& p1,
                                                 Code const pid, double const weight) {
    GrammageType const grammageStart = showerAxis_.getProjectedX(p0);
    GrammageType const grammageEnd = showerAxis_.getProjectedX(p1);

    // Avoid over counting in first bin when backscattered particle goes beyond the
    // injection point.
    if (grammageStart == grammageEnd) { return; }

    // Note: particle may go also "upward", thus, grammageEnd<grammageStart
    size_t const binStart = std::ceil(grammageStart / dX_);
    size_t const binEnd = std::floor(grammageEnd / dX_);

    CORSIKA_LOGGER_TRACE(TOutput::getLogger(),
                         "grammageStart={} End={} binStart={}, end={}",
                         grammageStart / 1_g * square(1_cm),
                         grammageEnd / 1_g * square(1_cm), binStart, binEnd);

    for (size_t bin = binStart; bin <= std::min(binEnd, profile_.size() - 1); ++bin) {
      if (pid == Code::Photon) {
        profile_.at(bin)[static_cast<int>(number_profile::ProfileIndex::Photon)] +=
            weight;
      } else if (pid == Code::Positron) {
        profile_.at(bin)[static_cast<int>(number_profile::ProfileIndex::Positron)] +=
            weight;
      } else if (pid == Code::Electron) {
        profile_.at(bin)[static_cast<int>(number_profile::ProfileIndex::Electron)] +=
            weight;
      } else if (pid == Code::MuPlus) {
        profile_.at(bin)[static_cast<int>(number_profile::ProfileIndex::MuPlus)] +=
            weight;
      } else if (pid == Code::MuMinus) {
        profile_.at(bin)[static_cast<int>(number_profile::ProfileIndex::MuMinus)] +=
            weight;
      } else if (is_hadron(pid)) {
        profile_.at(bin)[static_cast<int>(number_profile::ProfileIndex::Hadron)] +=
            weight;
      }
      if (is_charged(pid)) {
        profile_[bin][static_cast<int>(number_profile::ProfileIndex::Charged)] += weight;
      }
    }
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::write(GrammageType const Xstart,
                                                 GrammageType const Xend, Code const pid,
                                                 double const weight) {
    double const bstart = Xstart / dX_;
    double const bend = Xend / dX_;

    if (abs(bstart - floor(bstart + 0.5)) > 1e-2 ||
        abs(bend - floor(bend + 0.5)) > 1e-2 || abs(bend - bstart - 1) > 1e-2) {
      CORSIKA_LOGGER_ERROR(TOutput::getLogger(),
                           "CONEX and Corsika8 dX grammage binning are not the same! "
                           "Xstart={} Xend={} dX={}",
                           Xstart / 1_g * square(1_cm), Xend / 1_g * square(1_cm),
                           dX_ / 1_g * square(1_cm));
      throw std::runtime_error(
          "CONEX and Corsika8 dX grammage binning are not the same!");
    }

    size_t const bin = size_t((bend + bstart) / 2);
    if (bin >= profile_.size()) {
      CORSIKA_LOGGER_WARN(TOutput::getLogger(),
                          "Grammage bin {} outside of profile {}. skipping.", bin,
                          profile_.size());
      return;
    }

    if (pid == Code::Photon) {
      profile_.at(bin)[static_cast<int>(number_profile::ProfileIndex::Photon)] += weight;
    } else if (pid == Code::Positron) {
      profile_.at(bin)[static_cast<int>(number_profile::ProfileIndex::Positron)] +=
          weight;
    } else if (pid == Code::Electron) {
      profile_.at(bin)[static_cast<int>(number_profile::ProfileIndex::Electron)] +=
          weight;
    } else if (pid == Code::MuPlus) {
      profile_.at(bin)[static_cast<int>(number_profile::ProfileIndex::MuPlus)] += weight;
    } else if (pid == Code::MuMinus) {
      profile_.at(bin)[static_cast<int>(number_profile::ProfileIndex::MuMinus)] += weight;
    } else if (is_hadron(pid)) {
      profile_.at(bin)[static_cast<int>(number_profile::ProfileIndex::Hadron)] += weight;
    }
    if (is_charged(pid)) {
      profile_[bin][static_cast<int>(number_profile::ProfileIndex::Charged)] += weight;
    }
  }

  template <typename TOutput>
  inline YAML::Node LongitudinalWriter<TOutput>::getConfig() const {

    YAML::Node node;

    node["type"] = "LongitudinalProfile";
    node["units"]["grammage"] = "g/cm^2";
    node["bin-size"] = dX_ / (1_g / square(1_cm));
    node["nbins"] = nBins_;

    return node;
  }

  template <typename TOutput>
  inline YAML::Node LongitudinalWriter<TOutput>::getSummary() const {
    YAML::Node summary;
    return summary;
  }

} // namespace corsika::official_output_reference
