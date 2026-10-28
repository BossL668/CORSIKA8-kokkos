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

namespace corsika::production_output_reference {
#ifndef CORSIKA_OUTPUT_BRIDGE_NATIVE_ONLY
  using namespace ::corsika::kokkos;
  namespace radio = ::corsika::kokkos::radio;
  namespace scalar_constants = ::corsika::kokkos::scalar_constants;
#endif

  /** Lightweight, output-format-independent secondary at one interaction. */
  struct InteractionSecondarySnapshot {
    Code pid_;
    HEPEnergyType totalEnergy_;
    DirectionVector direction_;
  };

  /**
   * Output-format-independent first-interaction state shared by scalar Stack
   * views and accelerator output adapters.
   */
  struct FirstInteractionSnapshot {
    Code parentPid_;
    HEPEnergyType parentKineticEnergy_;
    Point position_;
    DirectionVector direction_;
    TimeType time_;
    std::vector<InteractionSecondarySnapshot> secondaries_;
  };

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
    LongitudinalWriter(ShowerAxis const& axis, GrammageType dX = 10_g / square(1_cm),
                       ProfileCrossingMode mode = ProfileCrossingMode::Both);

    LongitudinalWriter(ShowerAxis const& axis, size_t nbins,
                       GrammageType dX = 10_g / square(1_cm),
                       ProfileCrossingMode mode = ProfileCrossingMode::Both);

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
     * The immutable constructor policy chooses direction and endpoint ownership.
     * The low-level default remains Both for existing callers (e.g. terrain);
     * the air application explicitly defaults to OriginalC8. Counts are never
     * signed flux or unique particle identities. Transport is unaffected.
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
    ProfileCrossingMode getCrossingMode() const noexcept { return crossingMode_; }

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
    ShowerAxis const& showerAxis_;           ///< conversion between geometry and grammage
    GrammageType const dX_;                  ///< binning of profile.
    size_t const nBins_;                     ///< number of profile bins.
    ProfileCrossingMode const crossingMode_; ///< immutable observer convention
    std::vector<number_profile::ProfileData> profile_; // longitudinal profile
  };

  template <typename TOutput>
  inline LongitudinalWriter<TOutput>::LongitudinalWriter(ShowerAxis const& axis,
                                                         GrammageType dX,
                                                         ProfileCrossingMode mode)
      : LongitudinalWriter<TOutput>{
            axis, static_cast<unsigned int>(axis.getMaximumX() / dX) + 1, dX, mode} {}

  template <typename TOutput>
  inline LongitudinalWriter<TOutput>::LongitudinalWriter(ShowerAxis const& axis,
                                                         size_t nbins, GrammageType dX,
                                                         ProfileCrossingMode mode)
      : TOutput(number_profile::ProfileIndexNames)
      , showerAxis_(axis)
      , dX_(dX)
      , nBins_(nbins)
      , crossingMode_(mode)
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
    writeProjected(grammageStart, grammageEnd, pid, weight);
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::writeProjected(
      GrammageType const grammageStart, GrammageType const grammageEnd, Code const pid,
      double const weight) {

    auto const crossings = ::corsika::longitudinal_profile_bins(
        grammageStart / dX_, grammageEnd / dX_, profile_.size(), crossingMode_);

    CORSIKA_LOGGER_TRACE(
        TOutput::getLogger(), "grammageStart={} End={} binStart={}, end(exclusive)={}",
        grammageStart / 1_g * square(1_cm), grammageEnd / 1_g * square(1_cm),
        crossings.begin_, crossings.end_);

    for (size_t bin = crossings.begin_; bin < crossings.end_; ++bin) {
      addBin(bin, pid, weight);
    }
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::addBin(size_t bin, Code pid, double weight) {
    auto& row = profile_.at(bin);
    if (pid == Code::Photon) {
      row[static_cast<int>(number_profile::ProfileIndex::Photon)] += weight;
    } else if (pid == Code::Positron) {
      row[static_cast<int>(number_profile::ProfileIndex::Positron)] += weight;
    } else if (pid == Code::Electron) {
      row[static_cast<int>(number_profile::ProfileIndex::Electron)] += weight;
    } else if (pid == Code::MuPlus) {
      row[static_cast<int>(number_profile::ProfileIndex::MuPlus)] += weight;
    } else if (pid == Code::MuMinus) {
      row[static_cast<int>(number_profile::ProfileIndex::MuMinus)] += weight;
    } else if (is_hadron(pid)) {
      row[static_cast<int>(number_profile::ProfileIndex::Hadron)] += weight;
    }
    if (is_charged(pid)) {
      row[static_cast<int>(number_profile::ProfileIndex::Charged)] += weight;
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

    addBin(bin, pid, weight);
  }

  template <typename TOutput>
  inline YAML::Node LongitudinalWriter<TOutput>::getConfig() const {

    YAML::Node node;

    node["type"] = "LongitudinalProfile";
    node["units"]["grammage"] = "g/cm^2";
    node["bin-size"] = dX_ / (1_g / square(1_cm));
    node["nbins"] = nBins_;
    node["counting"] = "bidirectional-plane-crossings-v1";
    node["crossing-interval"] = "(min(Xstart,Xend),max(Xstart,Xend)]";
    node["profile-crossings"] = "both";
    if (crossingMode_ == ProfileCrossingMode::Forward) {
      node["profile-crossings"] = "forward";
      node["counting"] = "forward-plane-crossings-v1";
      node["crossing-interval"] = "(Xstart,Xend], Xend>Xstart";
    } else if (crossingMode_ == ProfileCrossingMode::OriginalC8) {
      node["profile-crossings"] = "original-c8";
      node["counting"] = "original-c8-forward-closed-v1";
      node["crossing-interval"] = "[Xstart,Xend], Xend>Xstart";
    }
    node["affects-transport"] = false;

    return node;
  }

  template <typename TOutput>
  inline YAML::Node LongitudinalWriter<TOutput>::getSummary() const {
    YAML::Node summary;
    return summary;
  }

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

    void writeSummary(boost::filesystem::path const& path) const override {
      summary_.writeSummary(path);
    }

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
    ShowerSummary summary_;
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
    summary_.open(directory);
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::startOfShower(unsigned int const showerId) {
    TOutput::startOfShower(showerId);
    // reset profile
    profile_.clear();
    profile_.resize(nBins_);
    electromagneticEnergyLost_ = HEPEnergyType::zero();
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

    summary_.event(showerId)["sum_dEdX"] = getEnergyLost() / 1_GeV;
    summary_.event(showerId)["sum_dEdX_em"] = electromagneticEnergyLost_ / 1_GeV;
    summary_.event(showerId)["Xmax"] = Xmax;
    summary_.event(showerId)["dEdXmax"] = dEdXmax;

    int iRow{0};
    for (dEdX_output::Profile const& row : profile_) {
      // here: write to underlying writer (e.g. parquet)
      TOutput::write(showerId, iRow * dX_, row);
      iRow++;
    }

    TOutput::endOfShower(showerId);
    summary_.flush();
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::write(Point const& p0, Point const& p1,
                                               Code const PID, HEPEnergyType const dE) {

    GrammageType grammageStart = showerAxis_.getProjectedX(p0);
    GrammageType grammageEnd = showerAxis_.getProjectedX(p1);
    writeProjected(grammageStart, grammageEnd, PID, dE);
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::writeProjected(GrammageType grammageStart,
                                                        GrammageType grammageEnd,
                                                        Code const PID,
                                                        HEPEnergyType const dE) {

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
      int const maxBin = int(profile_.size() - 1);
      int bin = grammageStart / dX_;
      if (bin < 0) bin = 0;
      if (bin > maxBin) bin = maxBin;
      profile_[bin][static_cast<int>(dEdX_output::ProfileIndex::Total)] += dE;
      if (is_em(PID)) { electromagneticEnergyLost_ += dE; }
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
    if (is_em(PID)) { electromagneticEnergyLost_ += energyCount; }
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::addBin(size_t bin, HEPEnergyType dE) {
    profile_.at(bin)[static_cast<int>(dEdX_output::ProfileIndex::Total)] += dE;
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::addElectromagneticBin(size_t bin,
                                                               HEPEnergyType dE) {
    addBin(bin, dE);
    electromagneticEnergyLost_ += dE;
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::write(Point const& point, Code const PID,
                                               HEPEnergyType const dE) {
    GrammageType grammage = showerAxis_.getProjectedX(point);
    int const maxBin = int(profile_.size() - 1);
    int bin = grammage / dX_;
    if (bin < 0) bin = 0;
    if (bin > maxBin) bin = maxBin;

    CORSIKA_LOGGER_TRACE(TOutput::getLogger(), "add local energy loss bin={} dE={} GeV ",
                         bin, dE / 1_GeV);

    profile_[bin][static_cast<int>(dEdX_output::ProfileIndex::Total)] += dE;
    if (is_em(PID)) { electromagneticEnergyLost_ += dE; }
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::write(GrammageType const Xstart,
                                               GrammageType const Xend, Code const PID,
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
    if (is_em(PID)) { electromagneticEnergyLost_ += dE; }
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

    return summary_.snapshot();
  }

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

  template <typename TOutput>
  inline ProductionWriter<TOutput>::ProductionWriter(ShowerAxis const& axis,
                                                     GrammageType dX)
      : ProductionWriter<TOutput>{
            axis, static_cast<unsigned int>(axis.getMaximumX() / dX) + 1, dX} {}

  template <typename TOutput>
  inline ProductionWriter<TOutput>::ProductionWriter(ShowerAxis const& axis, size_t nbins,
                                                     GrammageType dX)
      : TOutput(production_profile::ProjectileIndexNames)
      , showerAxis_(axis)
      , dX_(dX)
      , nBins_(nbins)
      , profile_{nbins} {}

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::startOfLibrary(
      boost::filesystem::path const& directory) {
    TOutput::startOfLibrary(directory);
    summary_.open(directory);
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::startOfShower(unsigned int const showerId) {
    profile_.clear();
    for (size_t i = 0; i < nBins_; ++i) { profile_.emplace_back(); }
    TOutput::startOfShower(showerId);
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::endOfLibrary() {
    TOutput::endOfLibrary();
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::endOfShower(unsigned int const showerId) {

    // determined XMumax and dNdXmax from quadratic interpolation
    double maximum = 0;
    size_t iMaximum = 0;
    for (size_t i = 0; i < profile_.size() - 3; ++i) {
      double value = (profile_[i + 0].at(
                          static_cast<int>(production_profile::ProjectileIndex::All)) +
                      profile_[i + 1].at(
                          static_cast<int>(production_profile::ProjectileIndex::All)) +
                      profile_[i + 2].at(
                          static_cast<int>(production_profile::ProjectileIndex::All)));
      if (value > maximum) {
        maximum = value;
        iMaximum = i;
      }
    }
    double const dX = dX_ / 1_g * square(1_cm);

    auto [Xmumax, dNdXmumax] = FindXmax::interpolateProfile(
        dX * (0.5 + iMaximum), dX * (1.5 + iMaximum), dX * (2.5 + iMaximum),
        profile_[iMaximum + 0].at(
            static_cast<int>(production_profile::ProjectileIndex::All)),
        profile_[iMaximum + 1].at(
            static_cast<int>(production_profile::ProjectileIndex::All)),
        profile_[iMaximum + 2].at(
            static_cast<int>(production_profile::ProjectileIndex::All)));

    summary_.event(showerId)["XmuMax"] = Xmumax;
    summary_.event(showerId)["dNdXmuMax"] = dNdXmumax;

    // write profile to file
    int iRow{0};
    for (production_profile::ProfileData const& row : profile_) {
      // here: write to underlying writer (e.g. parquet)
      TOutput::write(showerId, iRow * dX_, row);
      iRow++;
    }
    TOutput::endOfShower(showerId);
    summary_.flush();
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::write(Point const& p0, Code const projectile_pid,
                                               double const weight) {

    GrammageType const grammage = showerAxis_.getProjectedX(p0);

    writeProjected(grammage, projectile_pid, weight);
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::writeProjected(GrammageType const grammage,
                                                        Code const projectile_pid,
                                                        double const weight) {

    size_t const bin1 = std::floor(grammage / dX_);
    size_t const bin = std::min(bin1, profile_.size() - 1);

    CORSIKA_LOGGER_TRACE(TOutput::getLogger(), "grammage={} bin={}",
                         grammage / 1_g * square(1_cm), bin);
    addBin(bin, projectile_pid, weight);
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::addBin(size_t const bin,
                                                Code const projectile_pid,
                                                double const weight) {
    if (bin >= profile_.size()) {
      CORSIKA_LOGGER_WARN(TOutput::getLogger(),
                          "Production grammage bin {} outside of profile {}. skipping.",
                          bin, profile_.size());
      return;
    }
    if (projectile_pid == Code::Photon) {
      profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::Photon)] +=
          weight;
    } else if (projectile_pid == Code::Positron || projectile_pid == Code::Electron) {
      profile_.at(
          bin)[static_cast<int>(production_profile::ProjectileIndex::ElectronPositron)] +=
          weight;
    } else if (is_muon(projectile_pid)) {
      profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::Muon)] +=
          weight;
    } else if (is_hadron(projectile_pid)) {
      profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::Hadron)] +=
          weight;
      if (projectile_pid == Code::PiPlus || projectile_pid == Code::PiMinus) {
        profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::Pion)] +=
            weight;
      } else if (is_kaon(projectile_pid)) {
        profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::Kaon)] +=
            weight;
      } else {
        profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::Heavy)] +=
            weight;
      }
    }
    profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::All)] +=
        weight;
  }

  template <typename TOutput>
  inline YAML::Node ProductionWriter<TOutput>::getConfig() const {

    YAML::Node node;

    node["type"] = "ProductionProfile";
    node["units"]["grammage"] = "g/cm^2";
    node["bin-size"] = dX_ / (1_g / square(1_cm));
    node["nbins"] = nBins_;

    return node;
  }

  template <typename TOutput>
  inline YAML::Node ProductionWriter<TOutput>::getSummary() const {
    return summary_.snapshot();
  }

  struct ObservationPlaneStatistics {
    std::uint64_t particles_{};
    double weightedTotalEnergyGeV_{};
  };

  /**
   * @ingroup Modules
   * @{
   *
   * The ObservationPlane writes PDG codes, kinetic energies, locations and momentum unit
   * vectors of particles with respect to the central point of the plane into its output
   * file. By default, the particles are considered "absorbed" afterwards. You can also
   * set the ObservationPlane as non-absorbing.
   *
   * The default output format is parquet.
   *
   * **Note/Limitation:** as discussed in
   * https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/-/issues/397
   * you cannot put two ObservationPlanes exactly on top of each
   * other. Even if one of them is "permeable". You have to put a
   * small gap in between the two plane in such a scenario, or develop
   * another more specialized output class.
   */
  template <typename TTracking, typename TOutput = ParticleWriterParquet>
  class ObservationPlane : public ContinuousProcess<ObservationPlane<TTracking, TOutput>>,
                           public TOutput {

    using TOutput::write;

  public:
    /**
     * Construct a new Observation Plane object.
     *
     * @tparam TArgs
     * @param plane The plane.
     * @param x_dir The x-direction/axis.
     * @param absorbing Flag to make the plane absorbing.
     * @param outputArgs
     */
    template <typename... TArgs>
    ObservationPlane(Plane const& plane, DirectionVector const& x_dir,
                     bool const absorbing = true, TArgs&&... outputArgs);

    ~ObservationPlane() {}

    template <typename TParticle>
    ProcessReturn doContinuous(Step<TParticle>&, bool const stepLimit);

    /**
     * Write one particle already known to have reached this observation
     * plane.
     *
     * This is the non-tracking counterpart of doContinuous().  It is used by
     * transport backends (for example the CUDA EM wavefront backend) that
     * solve the surface intersection themselves and therefore do not own a
     * CORSIKA Step object at the observation boundary.
     */
    void writeParticle(Code const pid, HEPEnergyType const kineticEnergy,
                       Point const& position, DirectionVector const& direction,
                       TimeType const time, double const weight);

    template <typename TParticle, typename TTrajectory>
    LengthType getMaxStepLength(TParticle const&, TTrajectory const& vTrajectory);

    Plane getPlane() const { return plane_; }
    DirectionVector getXAxis() const { return xAxis_; }
    DirectionVector getYAxis() const { return yAxis_; }

    ObservationPlaneStatistics const& statistics() const noexcept { return statistics_; }

    YAML::Node getConfig() const;

  private:
    Plane const plane_;
    DirectionVector const xAxis_;
    DirectionVector const yAxis_;
    bool const deleteOnHit_;
    ObservationPlaneStatistics statistics_{};
  };
  //! @}

  template <typename TTracking, typename TOutput>
  template <typename... TArgs>
  ObservationPlane<TTracking, TOutput>::ObservationPlane(Plane const& obsPlane,
                                                         DirectionVector const& x_axis,
                                                         bool const deleteOnHit,
                                                         TArgs&&... args)
      : TOutput(std::forward<TArgs>(args)...)
      , plane_(obsPlane)
      , xAxis_(x_axis.normalized())
      , yAxis_(obsPlane.getNormal().cross(xAxis_))
      , deleteOnHit_(deleteOnHit) {}

  template <typename TTracking, typename TOutput>
  template <typename TParticle>
  inline ProcessReturn ObservationPlane<TTracking, TOutput>::doContinuous(
      Step<TParticle>& step, bool const stepLimit) {
    /*
       The current step did not yet reach the ObservationPlane, do nothing now and wait:
     */
    if (!stepLimit) {
      // @todo this is actually needed to fix small instabilities of the leap-frog
      // tracking: Note, this is NOT a general solution and should be clearly revised with
      // a more robust tracking. #ifdef DEBUG
      if (deleteOnHit_) {
        // since this is basically a bug, it cannot be tested LCOV_EXCL_START
        LengthType const check =
            (step.getPositionPost() - plane_.getCenter()).dot(plane_.getNormal());
        if (check < 0_m) {
          CORSIKA_LOG_WARN("PARTICLE AVOIDED OBSERVATIONPLANE {}", check);
          CORSIKA_LOG_WARN("Temporary fix: write and remove particle.");
        } else
          return ProcessReturn::Ok;
        // LCOV_EXCL_STOP
      } else
        // #endif
        return ProcessReturn::Ok;
    }

    writeParticle(step.getParticlePre().getPID(), step.getEkinPost(),
                  step.getPositionPost(), step.getDirectionPost(), step.getTimePost(),
                  step.getParticlePre().getWeight());

    CORSIKA_LOG_TRACE("Particle detected absorbed={}", deleteOnHit_);

    if (deleteOnHit_) {
      return ProcessReturn::ParticleAbsorbed;
    } else {
      // due to numerical precision of tracker, can fall into a Zeno's paradox
      // ensure that the particle is on the far side to avoid continuously hitting this
      // plane
      double epsilon = 1.0;
      while ((step.getPositionPost() - plane_.getCenter()).dot(plane_.getNormal()) >
             0_m) {
        auto const travelDist =
            -(step.getPositionPost() - plane_.getCenter()).dot(plane_.getNormal()) /
            step.getDirectionPost().dot(plane_.getNormal()) * step.getDirectionPost();
        // should be numerically close to plane after tracking to this extra length
        // should also be numerically close to zero
        step.add_displacement(travelDist * (1.0 + epsilon * 1e-4));
        epsilon *= 2.0;
      }

      return ProcessReturn::Ok;
    }
  } // namespace corsika

  template <typename TTracking, typename TOutput>
  inline void ObservationPlane<TTracking, TOutput>::writeParticle(
      Code const pid, HEPEnergyType const kineticEnergy, Point const& position,
      DirectionVector const& direction, TimeType const time, double const weight) {
    ++statistics_.particles_;
    statistics_.weightedTotalEnergyGeV_ +=
        weight * (kineticEnergy + get_mass(pid)) / 1_GeV;
    Vector const displacement = position - plane_.getCenter();
    this->write(pid, kineticEnergy, displacement.dot(xAxis_), displacement.dot(yAxis_),
                0_m, direction.dot(xAxis_), direction.dot(yAxis_),
                direction.dot(plane_.getNormal()), time, weight);
  }

  template <typename TTracking, typename TOutput>
  template <typename TParticle, typename TTrajectory>
  inline LengthType ObservationPlane<TTracking, TOutput>::getMaxStepLength(
      TParticle const& particle, TTrajectory const& trajectory) {

    CORSIKA_LOG_TRACE("getMaxStepLength, particle={}, pos={}, dir={}, plane={}",
                      particle.asString(), particle.getPosition(),
                      particle.getDirection(), plane_.asString());

    auto const intersection = TTracking::intersect(particle, plane_);

    TimeType const timeOfIntersection = intersection.getEntry();
    CORSIKA_LOG_TRACE("timeOfIntersection={}", timeOfIntersection);
    if (timeOfIntersection <= TimeType::zero()) {
      return std::numeric_limits<double>::infinity() * 1_m;
    }
    if (timeOfIntersection > trajectory.getDuration()) {
      return std::numeric_limits<double>::infinity() * 1_m;
    }
    double const fractionOfIntersection = timeOfIntersection / trajectory.getDuration();
    CORSIKA_LOG_TRACE("ObservationPlane: getMaxStepLength dist={} m, pos={}",
                      trajectory.getLength(fractionOfIntersection) / 1_m,
                      trajectory.getPosition(fractionOfIntersection));
    return trajectory.getLength(fractionOfIntersection);
  }

  template <typename TTracking, typename TOutput>
  inline YAML::Node ObservationPlane<TTracking, TOutput>::getConfig() const {
    using namespace units::si;

    // construct the top-level node
    YAML::Node node;

    // basic info
    node["type"] = "ObservationPlane";
    node["units"]["length"] = "m"; // add default units for values
    node["units"]["energy"] = "GeV";
    node["units"]["time"] = "s";

    // the center of the plane
    auto const center{plane_.getCenter()};

    // save each component in its native coordinate system
    auto const center_coords{center.getCoordinates(center.getCoordinateSystem())};
    node["plane"]["center"].push_back(center_coords.getX() / 1_m);
    node["plane"]["center"].push_back(center_coords.getY() / 1_m);
    node["plane"]["center"].push_back(center_coords.getZ() / 1_m);

    // the normal vector of the plane
    auto const normal{plane_.getNormal().getComponents()};
    node["plane"]["normal"].push_back(normal.getX().magnitude());
    node["plane"]["normal"].push_back(normal.getY().magnitude());
    node["plane"]["normal"].push_back(normal.getZ().magnitude());

    // the x-axis vector
    auto const xAxis_coords{xAxis_.getComponents(xAxis_.getCoordinateSystem())};
    node["x-axis"].push_back(xAxis_coords.getX().magnitude());
    node["x-axis"].push_back(xAxis_coords.getY().magnitude());
    node["x-axis"].push_back(xAxis_coords.getZ().magnitude());

    // the y-axis vector
    auto const yAxis_coords{yAxis_.getComponents(yAxis_.getCoordinateSystem())};
    node["y-axis"].push_back(yAxis_coords.getX().magnitude());
    node["y-axis"].push_back(yAxis_coords.getY().magnitude());
    node["y-axis"].push_back(yAxis_coords.getZ().magnitude());

    node["delete_on_hit"] = deleteOnHit_;

    return node;
  }

  template <typename TTracking, typename TOutput>
  class InteractionWriter
      : public SecondariesProcess<InteractionWriter<TTracking, TOutput>>,
        public BaseOutput {

  public:
    /**
     *
     * @param axis - axis used for calculating slant depth of interaction
     * @param obsPlane - used to define the location of the particles
     */
    InteractionWriter(ShowerAxis const& axis,
                      ObservationPlane<TTracking, TOutput> const& obsPlane);

    /**
     *
     * @tparam TStackView
     */
    template <typename TStackView>
    void doSecondaries(TStackView&);

    /**
     * Record a pre-thinning first-interaction snapshot produced outside the
     * scalar Stack path. Returns false when an earlier CPU/GPU interaction
     * has already claimed the shower's single output record.
     */
    bool recordFirstInteraction(FirstInteractionSnapshot const&);

    void startOfLibrary(boost::filesystem::path const&) override;
    void endOfLibrary() override;

    void startOfShower(unsigned int const) override;
    void endOfShower(unsigned int const) override;

    YAML::Node getConfig() const override;
    YAML::Node getSummary() const override;

    void writeSummary(boost::filesystem::path const& path) const override {
      summary_.writeSummary(path);
    }

    auto getInteractionCounter() { return interactionCounter_; }

  private:
    void writeFirstInteraction(FirstInteractionSnapshot const&);

    ObservationPlane<TTracking, TOutput> const obsPlane_;
    ShowerAxis const& showerAxis_;

    unsigned int interactionCounter_;
    unsigned int showerId_;

    ParquetStreamer output_;
    ShowerSummary summary_;

  }; // namespace corsika

  template <typename TTracking, typename TOutput>
  inline InteractionWriter<TTracking, TOutput>::InteractionWriter(
      ShowerAxis const& axis, ObservationPlane<TTracking, TOutput> const& obsPlane)
      : obsPlane_(obsPlane)
      , showerAxis_(axis)
      , interactionCounter_(0)
      , showerId_(0) {}

  template <typename TTracking, typename TOutput>
  template <typename TStackView>
  inline void InteractionWriter<TTracking, TOutput>::doSecondaries(TStackView& vS) {
    if (interactionCounter_ != 0) {
      ++interactionCounter_;
      return;
    }

    auto const primary = vS.getProjectile();
    FirstInteractionSnapshot snapshot{primary.getPID(),      primary.getKineticEnergy(),
                                      primary.getPosition(), primary.getDirection(),
                                      primary.getTime(),     {}};
    for (auto particle = vS.begin(); particle != vS.end(); ++particle) {
      snapshot.secondaries_.push_back(InteractionSecondarySnapshot{
          particle.getPID(), particle.getKineticEnergy() + get_mass(particle.getPID()),
          particle.getDirection()});
    }
    recordFirstInteraction(snapshot);
  }

  template <typename TTracking, typename TOutput>
  inline bool InteractionWriter<TTracking, TOutput>::recordFirstInteraction(
      FirstInteractionSnapshot const& snapshot) {
    if (interactionCounter_++) { return false; }
    writeFirstInteraction(snapshot);
    return true;
  }

  template <typename TTracking, typename TOutput>
  inline void InteractionWriter<TTracking, TOutput>::writeFirstInteraction(
      FirstInteractionSnapshot const& snapshot) {
    auto const dX = showerAxis_.getProjectedX(snapshot.position_);
    CORSIKA_LOG_INFO("First interaction at dX {}", dX);
    CORSIKA_LOG_INFO("Primary: {}, E_kin {}", snapshot.parentPid_,
                     snapshot.parentKineticEnergy_);

    Vector const displacement = snapshot.position_ - obsPlane_.getPlane().getCenter();
    auto const x = displacement.dot(obsPlane_.getXAxis());
    auto const y = displacement.dot(obsPlane_.getYAxis());
    auto const z = displacement.dot(obsPlane_.getPlane().getNormal());
    auto const nx = snapshot.direction_.dot(obsPlane_.getXAxis());
    auto const ny = snapshot.direction_.dot(obsPlane_.getYAxis());
    auto const nz = snapshot.direction_.dot(obsPlane_.getPlane().getNormal());
    auto const parent_total_energy =
        snapshot.parentKineticEnergy_ + get_mass(snapshot.parentPid_);
    auto const parent_momentum =
        snapshot.direction_ *
        calculate_momentum(parent_total_energy, get_mass(snapshot.parentPid_));
    auto const px = parent_momentum.dot(obsPlane_.getXAxis());
    auto const py = parent_momentum.dot(obsPlane_.getYAxis());
    auto const pz = parent_momentum.dot(obsPlane_.getPlane().getNormal());

    summary_.event(showerId_)["pdg"] = static_cast<int>(get_PDG(snapshot.parentPid_));
    summary_.event(showerId_)["name"] =
        static_cast<std::string>(get_name(snapshot.parentPid_));
    summary_.event(showerId_)["total_energy"] = parent_total_energy / 1_GeV;
    summary_.event(showerId_)["kinetic_energy"] = snapshot.parentKineticEnergy_ / 1_GeV;
    summary_.event(showerId_)["x"] = x / 1_m;
    summary_.event(showerId_)["y"] = y / 1_m;
    summary_.event(showerId_)["z"] = z / 1_m;
    summary_.event(showerId_)["nx"] = static_cast<double>(nx);
    summary_.event(showerId_)["ny"] = static_cast<double>(ny);
    summary_.event(showerId_)["nz"] = static_cast<double>(nz);
    summary_.event(showerId_)["px"] = static_cast<double>(px / 1_GeV);
    summary_.event(showerId_)["py"] = static_cast<double>(py / 1_GeV);
    summary_.event(showerId_)["pz"] = static_cast<double>(pz / 1_GeV);
    summary_.event(showerId_)["time"] = snapshot.time_ / 1_s;
    summary_.event(showerId_)["slant_depth"] = dX / (1_g / 1_cm / 1_cm);

    for (auto const& secondary : snapshot.secondaries_) {
      if (secondary.totalEnergy_ < get_mass(secondary.pid_)) {
        throw std::runtime_error(
            "first-interaction secondary total energy is below rest mass");
      }
      auto const momentum =
          secondary.direction_ *
          calculate_momentum(secondary.totalEnergy_, get_mass(secondary.pid_));
      *(output_.getWriter())
          << showerId_ << static_cast<int>(get_PDG(secondary.pid_))
          << static_cast<float>(momentum.dot(obsPlane_.getXAxis()) / 1_GeV)
          << static_cast<float>(momentum.dot(obsPlane_.getYAxis()) / 1_GeV)
          << static_cast<float>(momentum.dot(obsPlane_.getPlane().getNormal()) / 1_GeV)
          << parquet::EndRow;
      CORSIKA_LOG_INFO(" 2ndary: {}, E_tot {}", secondary.pid_, secondary.totalEnergy_);
    }
    summary_.event(showerId_)["n_secondaries"] = snapshot.secondaries_.size();
  }

  template <typename TTracking, typename TOutput>
  inline void InteractionWriter<TTracking, TOutput>::startOfLibrary(
      boost::filesystem::path const& directory) {
    summary_.open(directory);
    output_.initStreamer((directory / ("interactions.parquet")).string());

    // enable compression with the default level
    output_.enableCompression();

    output_.addField("pdg", parquet::Repetition::REQUIRED, parquet::Type::INT32,
                     parquet::ConvertedType::INT_32);
    output_.addField("px", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("py", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("pz", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);

    output_.buildStreamer();

    showerId_ = 0;
    interactionCounter_ = 0;
  }

  template <typename TTracking, typename TOutput>
  inline void InteractionWriter<TTracking, TOutput>::startOfShower(
      unsigned int const showerId) {
    showerId_ = showerId;
    interactionCounter_ = 0;
  }

  template <typename TTracking, typename TOutput>
  inline void InteractionWriter<TTracking, TOutput>::endOfShower(unsigned int const) {
    output_.flushStreamer();
    summary_.flush();
  }

  template <typename TTracking, typename TOutput>
  inline void InteractionWriter<TTracking, TOutput>::endOfLibrary() {
    output_.closeStreamer();
  }

  template <typename TTracking, typename TOutput>
  inline YAML::Node InteractionWriter<TTracking, TOutput>::getConfig() const {
    YAML::Node node;
    node["type"] = "Interactions";
    node["units"]["energy"] = "GeV";
    node["units"]["length"] = "m";
    node["units"]["time"] = "ns";
    node["units"]["grammage"] = "g/cm^2";
    return node;
  }

  template <typename TTracking, typename TOutput>
  inline YAML::Node InteractionWriter<TTracking, TOutput>::getSummary() const {
    return summary_.snapshot();
  }

#ifndef CORSIKA_OUTPUT_BRIDGE_NATIVE_ONLY

  /**
   * Marker used by validation routers that retain records but do not stream
   * them into production CORSIKA writers.
   */
  struct NullCorsikaOutputSink {};

  struct CorsikaOutputSinkStatistics {
    std::uint64_t steps_{};
    std::uint64_t depositedSteps_{};
    std::uint64_t radioTracks_{};
    std::uint64_t observations_{};
    std::uint64_t firstInteractions_{};
    double weightedDepositedEnergyGeV_{};
    double weightedMuonParentProductions_{};
  };

  namespace output_detail {

    inline Code code_from_device_pid(std::int32_t pid) {
      return convert_from_PDG(static_cast<PDGCode>(pid));
    }

    inline Point point_from_array(CoordinateSystemPtr const& coordinate_system,
                                  double const values_m[3]) {
      return Point{coordinate_system, values_m[0] * meter, values_m[1] * meter,
                   values_m[2] * meter};
    }

    inline DirectionVector direction_from_array(
        CoordinateSystemPtr const& coordinate_system, double const values[3]) {
      DirectionVector direction{coordinate_system, {values[0], values[1], values[2]}};
      auto const norm = direction.getNorm();
      if (!std::isfinite(norm) || !(norm > 0.)) {
        throw std::runtime_error("GPU EM output contains an invalid particle direction");
      }
      return direction.normalized();
    }

    /**
     * Minimal particle facade required by Step and the current CoREAS/ZHS
     * implementations.  It intentionally does not emulate a stack handle.
     */
    class RadioTrackParticle {
    public:
      RadioTrackParticle(Code pid, HEPEnergyType kinetic_energy,
                         DirectionVector direction, Point position, TimeType time,
                         double weight)
          : pid_(pid)
          , kineticEnergy_(kinetic_energy)
          , direction_(std::move(direction))
          , position_(std::move(position))
          , time_(time)
          , weight_(weight) {}

      Code getPID() const { return pid_; }
      HEPEnergyType getKineticEnergy() const { return kineticEnergy_; }
      DirectionVector const& getDirection() const { return direction_; }
      Point const& getPosition() const { return position_; }
      TimeType getTime() const { return time_; }
      double getWeight() const { return weight_; }

    private:
      Code pid_;
      HEPEnergyType kineticEnergy_;
      DirectionVector direction_;
      Point position_;
      TimeType time_;
      double weight_;
    };

  } // namespace output_detail

  /**
   * Stream records produced by PhysicalCudaEmRouter into the same writers and
   * CPU radio algorithms used by the scalar cascade.
   *
   * The device schema stores total energy in GeV.  CORSIKA stack particles
   * and the observation writer use kinetic energy, so the rest mass is
   * subtracted exactly at this boundary.  Energy deposited by a weighted
   * particle is also weighted here, matching ContinuousProcess and
   * ParticleCut on the scalar path.
   */
  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  class CorsikaOutputSink {
  public:
    CorsikaOutputSink(CoordinateSystemPtr coordinate_system,
                      TEnergyLossWriter& energy_loss, TLongitudinalWriter& longitudinal,
                      TProductionWriter& production, TObservationPlane& observation,
                      TInteractionWriter& interaction, TCoreas& coreas, TZhs& zhs,
                      bool radio_enabled, bool gpu_radio_enabled = false)
        : coordinateSystem_(std::move(coordinate_system))
        , energyLoss_(energy_loss)
        , longitudinal_(longitudinal)
        , production_(production)
        , observation_(observation)
        , interaction_(interaction)
        , coreas_(coreas)
        , zhs_(zhs)
        , radioEnabled_(radio_enabled)
        , gpuRadioEnabled_(gpu_radio_enabled) {
      if (!coordinateSystem_) {
        throw std::invalid_argument(
            "CUDA EM CORSIKA output sink requires a coordinate system");
      }
    }

    void onStep(StepRecord const& record) {
      validateStep(record);
      auto const pid = output_detail::code_from_device_pid(record.pid_);
      auto const start =
          output_detail::point_from_array(coordinateSystem_, record.startPositionM_);
      auto const end =
          output_detail::point_from_array(coordinateSystem_, record.endPositionM_);

      longitudinal_.write(start, end, pid, record.weight_);
      auto const continuous_deposit =
          record.depositedEnergyGeV_ - record.cutDepositedEnergyGeV_;
      if (continuous_deposit > 0.) {
        auto const weighted_deposit = continuous_deposit * record.weight_;
        energyLoss_.write(start, end, pid, weighted_deposit * 1_GeV);
        statistics_.weightedDepositedEnergyGeV_ += weighted_deposit;
        ++statistics_.depositedSteps_;
      }
      if (record.cutDepositedEnergyGeV_ > 0. || record.nativeMuonPairCut_ != 0U) {
        auto const weighted_deposit = record.cutDepositedEnergyGeV_ * record.weight_;
        energyLoss_.write(end, pid, weighted_deposit * 1_GeV);
        statistics_.weightedDepositedEnergyGeV_ += weighted_deposit;
        ++statistics_.depositedSteps_;
      }
      ++statistics_.steps_;
    }

    void onProjectedStep(ProjectedStepRecord const& record) {
      validateProjectedStep(record);
      auto const pid = output_detail::code_from_device_pid(record.pid_);
      auto const start = record.startGrammageGPerCm2_ * 1_g / square(1_cm);
      auto const end = record.endGrammageGPerCm2_ * 1_g / square(1_cm);

      longitudinal_.writeProjected(start, end, pid, record.weight_);
      auto const continuous_deposit =
          record.depositedEnergyGeV_ - record.cutDepositedEnergyGeV_;
      if (continuous_deposit > 0.) {
        auto const weighted_deposit = continuous_deposit * record.weight_;
        energyLoss_.writeProjected(start, end, pid, weighted_deposit * 1_GeV);
        statistics_.weightedDepositedEnergyGeV_ += weighted_deposit;
        ++statistics_.depositedSteps_;
      }
      if (record.cutDepositedEnergyGeV_ > 0. || record.nativeMuonPairCut_ != 0U) {
        auto const weighted_deposit = record.cutDepositedEnergyGeV_ * record.weight_;
        // Use the point-writer bin convention, independent of the continuous
        // dX threshold (which may be zero). No original CPU writer is changed.
        auto const bin_width = energyLoss_.getConfig()["bin-size"].template as<double>();
        auto const bins = energyLoss_.GetNBins();
        if (!std::isfinite(bin_width) || !(bin_width > 0.) || bins == 0)
          throw std::runtime_error("CUDA projected point deposit has invalid binning");
        auto const candidate =
            static_cast<std::size_t>(record.endGrammageGPerCm2_ / bin_width);
        auto const bin = candidate < bins ? candidate : bins - 1;
        if (is_em(pid))
          energyLoss_.addElectromagneticBin(bin, weighted_deposit * 1_GeV);
        else
          energyLoss_.addBin(bin, weighted_deposit * 1_GeV);
        statistics_.weightedDepositedEnergyGeV_ += weighted_deposit;
        ++statistics_.depositedSteps_;
      }
      ++statistics_.steps_;
    }

    void onGpuProfile(ProfileResult const& result) {
      auto const bins = result.photons_.size();
      if (result.electrons_.size() != bins || result.positrons_.size() != bins ||
          result.muonsMinus_.size() != bins || result.muonsPlus_.size() != bins ||
          result.muonParentProductions_.size() != bins ||
          result.energyLossGeV_.size() != bins ||
          result.muonEnergyLossGeV_.size() != bins || bins != longitudinal_.getNBins() ||
          bins != production_.getNBins() || bins != energyLoss_.GetNBins()) {
        throw std::runtime_error(
            "CUDA resident profile binning does not match CORSIKA writers");
      }
      if (result.fixedPointOverflows_ != 0 || result.invalidRecords_ != 0 ||
          !std::isfinite(result.weightedDepositedEnergyGeV_) ||
          !std::isfinite(result.weightedNativeMuonPairCutEnergyGeV_) ||
          result.weightedNativeMuonPairCutEnergyGeV_ > 0. ||
          result.weightedDepositedEnergyGeV_ <
              result.weightedNativeMuonPairCutEnergyGeV_ - 1.e-12 ||
          !std::isfinite(result.weightedMediumRestMassInputGeV_) ||
          result.weightedMediumRestMassInputGeV_ < 0. ||
          !std::isfinite(result.weightedCutRestMassEnergyGeV_) ||
          result.weightedCutRestMassEnergyGeV_ < 0. ||
          !std::isfinite(result.weightedObservedTotalEnergyGeV_) ||
          result.weightedObservedTotalEnergyGeV_ < 0. ||
          !std::isfinite(result.weightedEscapedTotalEnergyGeV_) ||
          result.weightedEscapedTotalEnergyGeV_ < 0. ||
          !std::isfinite(result.weightedUnwrittenPhotoelectricBindingEnergyGeV_) ||
          result.weightedUnwrittenPhotoelectricBindingEnergyGeV_ < 0. ||
          !std::isfinite(result.weightedObservationCutOverlapEnergyGeV_) ||
          result.weightedObservationCutOverlapEnergyGeV_ < 0. ||
          !std::isfinite(result.weightedMassConventionCorrectionGeV_)) {
        throw std::runtime_error(
            "CUDA resident profile contains invalid accumulated output");
      }
      for (std::size_t bin = 0; bin < bins; ++bin) {
        auto const photon = result.photons_[bin];
        auto const electron = result.electrons_[bin];
        auto const positron = result.positrons_[bin];
        auto const muon_minus = result.muonsMinus_[bin];
        auto const muon_plus = result.muonsPlus_[bin];
        auto const muon_parent_productions = result.muonParentProductions_[bin];
        auto const deposited = result.energyLossGeV_[bin];
        auto const muon_deposited = result.muonEnergyLossGeV_[bin];
        if (!std::isfinite(photon) || photon < 0. || !std::isfinite(electron) ||
            electron < 0. || !std::isfinite(positron) || positron < 0. ||
            !std::isfinite(muon_minus) || muon_minus < 0. || !std::isfinite(muon_plus) ||
            muon_plus < 0. || !std::isfinite(muon_parent_productions) ||
            muon_parent_productions < 0. || !std::isfinite(deposited) ||
            deposited < result.weightedNativeMuonPairCutEnergyGeV_ - 1.e-12 ||
            !std::isfinite(muon_deposited) ||
            muon_deposited < result.weightedNativeMuonPairCutEnergyGeV_ - 1.e-12 ||
            muon_deposited > deposited) {
          throw std::runtime_error(
              "CUDA resident profile contains a non-finite or negative bin");
        }
        if (photon != 0.) { longitudinal_.addBin(bin, Code::Photon, photon); }
        if (electron != 0.) { longitudinal_.addBin(bin, Code::Electron, electron); }
        if (positron != 0.) { longitudinal_.addBin(bin, Code::Positron, positron); }
        if (muon_minus != 0.) { longitudinal_.addBin(bin, Code::MuMinus, muon_minus); }
        if (muon_plus != 0.) { longitudinal_.addBin(bin, Code::MuPlus, muon_plus); }
        if (muon_parent_productions != 0.) {
          // ProductionWriter classifies both muon signs into the same parent
          // column and always mirrors the contribution into "all".
          production_.addBin(bin, Code::MuMinus, muon_parent_productions);
          statistics_.weightedMuonParentProductions_ += muon_parent_productions;
        }
        if (deposited != 0. || muon_deposited != 0.) {
          auto const electromagnetic_deposited = deposited - muon_deposited;
          if (electromagnetic_deposited != 0.) {
            energyLoss_.addElectromagneticBin(bin, electromagnetic_deposited * 1_GeV);
          }
          if (muon_deposited != 0.) { energyLoss_.addBin(bin, muon_deposited * 1_GeV); }
        }
      }
      statistics_.steps_ += result.steps_;
      statistics_.depositedSteps_ += result.depositedSteps_;
      statistics_.weightedDepositedEnergyGeV_ += result.weightedDepositedEnergyGeV_;
    }

    void onRadioTrack(RadioTrackRecord const& record) {
      if (!radioEnabled_) { return; }
      if (gpuRadioEnabled_) { return; }
      validateStep(record.step_);
      auto const pid = output_detail::code_from_device_pid(record.step_.pid_);
      if (pid != Code::Electron && pid != Code::Positron) { return; }

      auto const start = output_detail::point_from_array(coordinateSystem_,
                                                         record.step_.startPositionM_);
      auto const end =
          output_detail::point_from_array(coordinateSystem_, record.step_.endPositionM_);
      auto const start_direction =
          output_detail::direction_from_array(coordinateSystem_, record.startDirection_);
      auto const end_direction =
          output_detail::direction_from_array(coordinateSystem_, record.endDirection_);
      auto const duration = (record.step_.endTimeS_ - record.step_.startTimeS_) * second;
      auto const displacement = end - start;
      if (!(duration > TimeType::zero()) ||
          displacement.getSquaredNorm() == static_pow<2>(0_m)) {
        return;
      }

      auto const mass = get_mass(pid);
      auto const kinetic_energy = record.step_.startEnergyGeV_ * 1_GeV - mass;
      if (kinetic_energy < HEPEnergyType::zero()) {
        throw std::runtime_error("CUDA EM radio record has total energy below rest mass");
      }
      output_detail::RadioTrackParticle particle{pid,
                                                 kinetic_energy,
                                                 start_direction,
                                                 start,
                                                 record.step_.startTimeS_ * second,
                                                 record.step_.weight_};

      auto const average_speed = displacement.getNorm() / duration;
      Line const line{start, displacement / duration};
      StraightTrajectory const trajectory{line, duration, duration,
                                          start_direction * average_speed,
                                          end_direction * average_speed};
      Step<output_detail::RadioTrackParticle> step{particle, trajectory};
      step.add_dEkin((record.step_.endEnergyGeV_ - record.step_.startEnergyGeV_) * 1_GeV);

      auto const coreas_return = coreas_.doContinuous(step, false);
      auto const zhs_return = zhs_.doContinuous(step, false);
      if (coreas_return != ProcessReturn::Ok || zhs_return != ProcessReturn::Ok) {
        throw std::runtime_error("CPU radio process rejected a CUDA EM track segment");
      }
      ++statistics_.radioTracks_;
    }

    void onGpuRadioWaveforms(radio::RadioWaveforms const& waveforms,
                             std::uint64_t track_count) {
      if (!radioEnabled_ || !gpuRadioEnabled_) {
        throw std::logic_error(
            "received CUDA radio waveforms while GPU radio is disabled");
      }
      auto& coreas_observers = coreas_.observerCollection().getObservers();
      auto& zhs_observers = zhs_.observerCollection().getObservers();
      if (waveforms.coreas_.size() != coreas_observers.size() ||
          waveforms.zhs_.size() != zhs_observers.size()) {
        throw std::runtime_error("CUDA radio waveform observer count mismatch");
      }
      for (std::size_t index = 0; index < waveforms.coreas_.size(); ++index) {
        auto const& waveform = waveforms.coreas_[index];
        coreas_observers[index].addWaveform(waveform.x_, waveform.y_, waveform.z_);
      }
      for (std::size_t index = 0; index < waveforms.zhs_.size(); ++index) {
        auto const& waveform = waveforms.zhs_[index];
        zhs_observers[index].addWaveform(waveform.x_, waveform.y_, waveform.z_);
      }
      statistics_.radioTracks_ += track_count;
    }

    void onObservation(ObservationRecord const& record) {
      if (record.status_ != ObservationStatus::ReachedObservationSurface) { return; }
      auto const pid = output_detail::code_from_device_pid(record.particle_.pid_);
      auto const total_energy = record.particle_.energyGeV_ * 1_GeV;
      auto const kinetic_energy = total_energy - get_mass(pid);
      if (kinetic_energy < HEPEnergyType::zero()) {
        throw std::runtime_error("CUDA EM observation has total energy below rest mass");
      }
      auto const position =
          output_detail::point_from_array(coordinateSystem_, record.particle_.positionM_);
      auto const direction = output_detail::direction_from_array(
          coordinateSystem_, record.particle_.direction_);
      observation_.writeParticle(pid, kinetic_energy, position, direction,
                                 record.particle_.timeS_ * second,
                                 record.particle_.weight_);
      ++statistics_.observations_;
    }

    void onFirstInteraction(FirstInteractionRecord const& record) {
      if (record.parentAtVertex_.generation_ != 0 || record.secondaryCount_ == 0 ||
          record.secondaryCount_ > 3) {
        throw std::runtime_error(
            "CUDA EM first-interaction record is structurally invalid");
      }
      auto const parent_pid =
          output_detail::code_from_device_pid(record.parentAtVertex_.pid_);
      auto const parent_total_energy = record.parentAtVertex_.energyGeV_ * 1_GeV;
      auto const parent_kinetic_energy = parent_total_energy - get_mass(parent_pid);
      if (parent_kinetic_energy < HEPEnergyType::zero()) {
        throw std::runtime_error(
            "CUDA first-interaction parent energy is below rest mass");
      }
      FirstInteractionSnapshot snapshot{
          parent_pid,
          parent_kinetic_energy,
          output_detail::point_from_array(coordinateSystem_,
                                          record.parentAtVertex_.positionM_),
          output_detail::direction_from_array(coordinateSystem_,
                                              record.parentAtVertex_.direction_),
          record.parentAtVertex_.timeS_ * second,
          {}};
      snapshot.secondaries_.reserve(record.secondaryCount_);
      for (std::uint32_t index = 0; index < record.secondaryCount_; ++index) {
        auto const& child = record.secondaries_[index];
        auto const pid = output_detail::code_from_device_pid(child.pid_);
        auto const total_energy = child.energyGeV_ * 1_GeV;
        if (total_energy < get_mass(pid)) {
          throw std::runtime_error(
              "CUDA first-interaction child energy is below rest mass");
        }
        snapshot.secondaries_.push_back(InteractionSecondarySnapshot{
            pid, total_energy,
            output_detail::direction_from_array(coordinateSystem_, child.direction_)});
      }
      if (interaction_.recordFirstInteraction(snapshot)) {
        ++statistics_.firstInteractions_;
      }
    }

    CorsikaOutputSinkStatistics const& statistics() const noexcept { return statistics_; }

  private:
    static void validateStep(StepRecord const& record) {
      auto const native_cut = is_native_muon_pair_cut_deposit(
                                  record.pid_, record.nativeMuonPairCut_,
                                  record.endEnergyGeV_, record.cutDepositedEnergyGeV_) &&
                              record.startTimeS_ == record.endTimeS_ &&
                              record.startEnergyGeV_ == record.endEnergyGeV_ &&
                              record.depositedEnergyGeV_ == record.cutDepositedEnergyGeV_;
      if (!std::isfinite(record.startTimeS_) || !std::isfinite(record.endTimeS_) ||
          record.endTimeS_ < record.startTimeS_ ||
          !std::isfinite(record.startEnergyGeV_) ||
          !std::isfinite(record.endEnergyGeV_) || record.startEnergyGeV_ < 0. ||
          record.endEnergyGeV_ < 0. || !std::isfinite(record.depositedEnergyGeV_) ||
          (record.depositedEnergyGeV_ < 0. && !native_cut) ||
          !std::isfinite(record.cutDepositedEnergyGeV_) ||
          (record.cutDepositedEnergyGeV_ < 0. && !native_cut) ||
          (record.nativeMuonPairCut_ != 0U && !native_cut) ||
          record.cutDepositedEnergyGeV_ > record.depositedEnergyGeV_ ||
          !std::isfinite(record.weight_) || record.weight_ < 0.) {
        throw std::runtime_error("CUDA EM output contains an invalid step record");
      }
      for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(record.startPositionM_[axis]) ||
            !std::isfinite(record.endPositionM_[axis]) ||
            (native_cut && record.startPositionM_[axis] != record.endPositionM_[axis])) {
          throw std::runtime_error("CUDA EM output contains a non-finite position");
        }
      }
    }

    static void validateProjectedStep(ProjectedStepRecord const& record) {
      auto const native_cut =
          is_native_muon_pair_cut_deposit(record.pid_, record.nativeMuonPairCut_,
                                          record.endEnergyGeV_,
                                          record.cutDepositedEnergyGeV_) &&
          record.startGrammageGPerCm2_ == record.endGrammageGPerCm2_ &&
          record.transportLimit_ ==
              static_cast<std::int32_t>(LeptonTransportLimit::ParticleCut) &&
          record.depositedEnergyGeV_ == record.cutDepositedEnergyGeV_;
      if (!std::isfinite(record.startGrammageGPerCm2_) ||
          !std::isfinite(record.endGrammageGPerCm2_) ||
          record.startGrammageGPerCm2_ < 0. || record.endGrammageGPerCm2_ < 0. ||
          !std::isfinite(record.endEnergyGeV_) || record.endEnergyGeV_ < 0. ||
          !std::isfinite(record.depositedEnergyGeV_) ||
          (record.depositedEnergyGeV_ < 0. && !native_cut) ||
          !std::isfinite(record.cutDepositedEnergyGeV_) ||
          (record.cutDepositedEnergyGeV_ < 0. && !native_cut) ||
          (record.nativeMuonPairCut_ != 0U && !native_cut) ||
          record.cutDepositedEnergyGeV_ > record.depositedEnergyGeV_ ||
          !std::isfinite(record.weight_) || record.weight_ < 0.) {
        throw std::runtime_error(
            "CUDA EM output contains an invalid projected step record");
      }
    }

    CoordinateSystemPtr coordinateSystem_;
    TEnergyLossWriter& energyLoss_;
    TLongitudinalWriter& longitudinal_;
    TProductionWriter& production_;
    TObservationPlane& observation_;
    TInteractionWriter& interaction_;
    TCoreas& coreas_;
    TZhs& zhs_;
    bool radioEnabled_{};
    bool gpuRadioEnabled_{};
    CorsikaOutputSinkStatistics statistics_{};
  };

  /**
   * Small OutputManager participant carrying CUDA-EM provenance and per-shower
   * completion state.
   *
   * Physics writers own the large tabular products. This output intentionally
   * writes only YAML so a failed CUDA shower is still self-describing and can
   * never be mistaken for a complete production event.
   */
  class EmRunOutput final : public BaseOutput {
  public:
    explicit EmRunOutput(YAML::Node configuration)
        : configuration_(std::move(configuration)) {
      configuration_["scalar_transport_constants_version"] = scalar_constants::Version;
      configuration_["magnetic_rigidity_GeV_per_T_m"] =
          scalar_constants::MagneticRigidityGeVPerTeslaMeter;
    }

    void startOfLibrary(boost::filesystem::path const& directory) final {
      summary_.open(directory);
      setInit(true);
    }

    void startOfShower(unsigned int const shower_id) final {
      auto shower = summary_.event(shower_id);
      shower["complete"] = false;
      shower["status"] = "in_progress";
      activeShower_ = shower_id;
      showerRecorded_ = false;
    }

    void recordComplete(unsigned int shower_id, YAML::Node metadata) {
      auto shower = summary_.event(shower_id);
      shower["complete"] = true;
      shower["status"] = "complete";
      shower["statistics"] = std::move(metadata);
      activeShower_ = shower_id;
      showerRecorded_ = true;
      summary_.flush();
    }

    void recordIncomplete(unsigned int shower_id, std::string const& reason) {
      auto shower = summary_.event(shower_id);
      shower["complete"] = false;
      shower["status"] = "incomplete";
      shower["failure_reason"] = reason;
      activeShower_ = shower_id;
      showerRecorded_ = true;
      summary_.flush();
    }

    void recordIncomplete(unsigned int shower_id, std::string const& reason,
                          YAML::Node metadata) {
      auto shower = summary_.event(shower_id);
      shower["complete"] = false;
      shower["status"] = "incomplete";
      shower["failure_reason"] = reason;
      shower["statistics"] = std::move(metadata);
      activeShower_ = shower_id;
      showerRecorded_ = true;
      summary_.flush();
    }

    void endOfShower(unsigned int const shower_id) final {
      // HybridCascade closes OutputManager before the application has access
      // to its final backend/router counters. Keep the in-progress marker
      // until recordComplete() is called immediately after run().
      activeShower_ = shower_id;
    }

    void endOfLibrary() final {
      if (activeShower_ != InvalidShower && !showerRecorded_) {
        recordIncomplete(activeShower_,
                         "CUDA EM library ended without a completion record");
      }
      setInit(false);
    }

    YAML::Node getConfig() const final { return configuration_; }

    YAML::Node getSummary() const final { return summary_.snapshot(); }

    void writeSummary(boost::filesystem::path const& path) const final {
      if (activeShower_ != InvalidShower && !showerRecorded_) {
        auto shower = summary_.event(activeShower_);
        shower["complete"] = false;
        shower["status"] = "incomplete";
        shower["failure_reason"] = "CUDA EM library ended without a completion record";
      }
      summary_.writeSummary(path);
    }

  private:
    static constexpr unsigned int InvalidShower = static_cast<unsigned int>(-1);

    YAML::Node configuration_{};
    mutable ShowerSummary summary_;
    unsigned int activeShower_{InvalidShower};
    bool showerRecorded_{};
  };

#endif

} // namespace corsika::production_output_reference
