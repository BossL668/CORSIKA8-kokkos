/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/proposal/NativeTableExporter.hpp>
#include <corsika/modules/proposal/NativeCalculatorView.hpp>

#include <PROPOSAL/Constants.h>
#include <PROPOSAL/crosssection/CrossSection.h>
#include <PROPOSAL/interpolation/InterpolationTableExport.h>
#include <PROPOSAL/medium/Components.h>
#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/propagation_utility/Displacement.h>
#include <PROPOSAL/propagation_utility/Interaction.h>
#include <PROPOSAL/version.h>
#include <CubicInterpolation/version.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace corsika::kokkos {

  using namespace corsika::units::si;

  namespace {

    template <typename TValue>
    TValue const* borrowed_pointer(
        std::optional<std::reference_wrapper<TValue const>> const& value) {
      return value ? &value->get() : nullptr;
    }

    NativeAxisDescriptor export_axis(
        cubic_splines::AxisDescriptor<double> const& source) {
      NativeAxisType type{};
      switch (source.type) {
        case cubic_splines::AxisType::Linear:
          type = NativeAxisType::Linear;
          break;
        case cubic_splines::AxisType::Exponential:
          type = NativeAxisType::Exponential;
          break;
        case cubic_splines::AxisType::ExponentialMinusOne:
          type = NativeAxisType::ExponentialMinusOne;
          break;
        default:
          throw std::invalid_argument("unsupported CubicInterpolation coordinate axis");
      }
      if (source.nodes > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("native interpolation axis is too large");
      return {type, static_cast<std::uint32_t>(source.nodes), source.low, source.high,
              source.stepsize};
    }

    using composition_identity_type =
        std::vector<std::pair<std::uint64_t, std::uint64_t>>;

    std::uint64_t double_bits(double value) {
      std::uint64_t result{};
      static_assert(sizeof(result) == sizeof(value));
      std::memcpy(&result, &value, sizeof(result));
      return result;
    }

    composition_identity_type composition_identity(PROPOSAL::Medium const& medium) {
      composition_identity_type result;
      result.reserve(medium.GetComponents().size() + 1);
      result.emplace_back(0u, double_bits(medium.GetSumNucleons()));
      for (auto const& component : medium.GetComponents()) {
        result.emplace_back(static_cast<std::uint64_t>(component.GetHash()),
                            double_bits(component.GetAtomInMolecule()));
      }
      std::sort(result.begin() + 1, result.end());
      return result;
    }

    bool supported_native_projectile(Code projectile) {
      switch (projectile) {
        case Code::Photon:
        case Code::Electron:
        case Code::Positron:
        case Code::MuMinus:
        case Code::MuPlus:
          return true;
        default:
          return false;
      }
    }

    void require_single_composition(
        std::vector<proposal::NativeInteractionCalculatorView> const& interactions,
        std::vector<proposal::NativeContinuousCalculatorView> const& continuous) {
      composition_identity_type expected;
      std::array<std::uint64_t, 7> expected_ionization_parameters{};
      bool have_expected = false;
      auto inspect = [&](PROPOSAL::Medium const* medium) {
        if (!medium)
          throw std::invalid_argument("null medium in PROPOSAL native calculator view");
        auto const identity = composition_identity(*medium);
        auto const ionization_parameters = std::array<std::uint64_t, 7>{
            double_bits(medium->GetI()),  double_bits(medium->GetC()),
            double_bits(medium->GetA()),  double_bits(medium->GetM()),
            double_bits(medium->GetX0()), double_bits(medium->GetX1()),
            double_bits(medium->GetD0())};
        if (!have_expected) {
          expected = identity;
          expected_ionization_parameters = ionization_parameters;
          have_expected = true;
        } else if (identity != expected ||
                   ionization_parameters != expected_ionization_parameters) {
          throw std::invalid_argument(
              "proposal-native supports one medium physics definition; "
              "layers may differ only in mass density, not composition, "
              "mean excitation energy, or density-effect parameters");
        }
      };
      for (auto const& view : interactions)
        if (supported_native_projectile(view.getProjectile()))
          inspect(borrowed_pointer(view.getMedium()));
      for (auto const& view : continuous)
        if (supported_native_projectile(view.getProjectile()))
          inspect(borrowed_pointer(view.getMedium()));
      if (!have_expected)
        throw std::invalid_argument(
            "proposal-native export contains no supported projectile calculators");
    }

    NativeLossTransform export_loss_transform(PROPOSAL::RelativeLossTransform transform) {
      switch (transform) {
        case PROPOSAL::RelativeLossTransform::RelativeLog:
          return NativeLossTransform::RelativeLog;
        case PROPOSAL::RelativeLossTransform::RelativeLogPowerOnePointFive:
          return NativeLossTransform::RelativeLogPowerOnePointFive;
        case PROPOSAL::RelativeLossTransform::OneMinusLog:
          return NativeLossTransform::OneMinusLog;
      }
      throw std::invalid_argument("unsupported PROPOSAL relative-loss transform");
    }

    NativeKinematicModel classify_kinematic_model(std::int32_t process_id,
                                                  std::string const& parametrization,
                                                  std::int32_t pdg_id) {
      auto const electron = pdg_id == 11 || pdg_id == -11;
      auto const muon = pdg_id == 13 || pdg_id == -13;
      if (process_id == 1000000013 && pdg_id == 22 && parametrization == "kochmotz")
        return NativeKinematicModel::PhotonPairFullLoss;
      if (process_id == 1000000010 && pdg_id == 22 && parametrization == "KleinNishina")
        return NativeKinematicModel::Compton;
      if (process_id == 1000000016 && pdg_id == 22 && parametrization == "Sauter")
        return NativeKinematicModel::OnlyStochastic;
      if (process_id == 1000000012 && pdg_id == -11 && parametrization == "Heitler")
        return NativeKinematicModel::OnlyStochastic;
      if (process_id == 1000000014 && pdg_id == 22 &&
          parametrization == "HeckC7Shadowing")
        return NativeKinematicModel::OnlyStochastic;
      if (process_id == 1000000002 && electron && parametrization == "ElectronScreening")
        return NativeKinematicModel::BremsElectronScreening;
      if (process_id == 1000000004 && electron &&
          parametrization == "ForElectronPositron")
        return NativeKinematicModel::ElectronPair;
      if (process_id == 1000000003 && pdg_id == 11 &&
          parametrization == "BergerSeltzerMoller")
        return NativeKinematicModel::IonizationMoller;
      if (process_id == 1000000003 && pdg_id == -11 &&
          parametrization == "BergerSeltzerBhabha")
        return NativeKinematicModel::IonizationBhabha;
      if (process_id == 1000000003 && muon && parametrization == "BetheBlochRossi")
        return NativeKinematicModel::IonizationBetheBlochRossi;

      if (process_id == 1000000002 && muon &&
          parametrization == "KelnerKokoulinPetrukhin")
        return NativeKinematicModel::BremsKelnerKokoulinPetrukhin;
      if (process_id == 1000000004 && muon &&
          parametrization == "KelnerKokoulinPetrukhin")
        return NativeKinematicModel::ElectronPair;
      if (process_id == 1000000005 && (electron || muon) &&
          parametrization == "AbramowiczLevinLevyMaor97")
        return NativeKinematicModel::PhotonuclearAllm97;
      if (process_id == 1000000015 && pdg_id == 22 &&
          parametrization == "BurkhardtKelnerKokoulin") {
        // This process' 0.5 +/- sqrt(...) interval is the *internal* x used
        // by the CPU secondary generator.  Parametrization.hpp specializes
        // is_only_stochastic<PhotoMuPairBurkhardtKelnerKokoulin> to true, so
        // CrossSection::CalculateStochasticLoss returns the external loss
        // fraction v=1 for every residual rate.  Keep that scalar contract;
        // the separate device/CPU final-state query samples x independently.
        return NativeKinematicModel::OnlyStochastic;
      }

      std::ostringstream message;
      message << "unsupported proposal-native process/PID/parametrization: "
              << "process=" << process_id << ", pdg=" << pdg_id
              << ", parametrization=" << parametrization;
      throw std::invalid_argument(message.str());
    }

    PROPOSAL::Component component_for_hash(PROPOSAL::Medium const& medium,
                                           std::uint64_t hash) {
      for (auto const& component : medium.GetComponents()) {
        if (static_cast<std::uint64_t>(component.GetHash()) == hash) return component;
      }
      if (static_cast<std::uint64_t>(medium.GetHash()) == hash)
        return PROPOSAL::Component{};
      throw std::invalid_argument(
          "PROPOSAL native table references a component outside its medium");
    }

    NativeBicubicDescriptor append_bicubic(
        ProposalNativeTableSet& result, PROPOSAL::DndxInterpolationExport const& source,
        bool precompute_polynomial) {
      auto const rows = source.spline.dimensions[0];
      auto const columns = source.spline.dimensions[1];
      if (rows < 2 || columns < 2 || rows > std::numeric_limits<std::uint32_t>::max() ||
          columns > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("invalid native bicubic dimensions");
      auto const count = static_cast<std::uint64_t>(rows) * columns;
      if (source.spline.values.size() != count ||
          source.spline.derivative_axis0.size() != count ||
          source.spline.derivative_axis1.size() != count ||
          source.spline.mixed_derivative.size() != count)
        throw std::invalid_argument("incomplete native bicubic coefficient arrays");
      if (result.bicubicValues_.size() >
          std::numeric_limits<std::uint64_t>::max() - count)
        throw std::overflow_error("native bicubic coefficient offset overflow");
      auto const offset = static_cast<std::uint64_t>(result.bicubicValues_.size());
      result.bicubicValues_.insert(result.bicubicValues_.end(),
                                   source.spline.values.begin(),
                                   source.spline.values.end());
      result.bicubicDerivativeEnergy_.insert(result.bicubicDerivativeEnergy_.end(),
                                             source.spline.derivative_axis0.begin(),
                                             source.spline.derivative_axis0.end());
      result.bicubicDerivativeLoss_.insert(result.bicubicDerivativeLoss_.end(),
                                           source.spline.derivative_axis1.begin(),
                                           source.spline.derivative_axis1.end());
      result.bicubicMixedDerivative_.insert(result.bicubicMixedDerivative_.end(),
                                            source.spline.mixed_derivative.begin(),
                                            source.spline.mixed_derivative.end());
      auto polynomial_offset = NoNativePolynomialCoefficients;
      if (precompute_polynomial) {
        polynomial_offset =
            static_cast<std::uint64_t>(result.bicubicPolynomialCoefficients_.size());
        auto const cells = static_cast<std::uint64_t>(rows - 1) *
                           static_cast<std::uint64_t>(columns - 1);
        if (cells > (std::numeric_limits<std::size_t>::max() -
                     result.bicubicPolynomialCoefficients_.size()) /
                        16u)
          throw std::overflow_error("native bicubic polynomial coefficient overflow");
        result.bicubicPolynomialCoefficients_.reserve(
            result.bicubicPolynomialCoefficients_.size() + 16u * cells);
        Eigen::Matrix4d m;
        m << 1., 0., -3., 2., 0., 0., 3., -2., 0., 1., -2., 1., 0., 0., -1., 1.;
        auto const element = [&](std::vector<double> const& values, std::size_t row,
                                 std::size_t column) {
          return values[row + rows * column];
        };
        // Match BicubicSplines::evaluate(): form C=M^T*(T*M) with Eigen on
        // the host.  This is a lossless layout conversion of the exported
        // spline, not a second physics sampling table.
        auto const row_count = static_cast<std::size_t>(rows);
        auto const column_count = static_cast<std::size_t>(columns);
        for (std::size_t loss = 0; loss + 1 < column_count; ++loss) {
          for (std::size_t energy = 0; energy + 1 < row_count; ++energy) {
            Eigen::Matrix4d temp;
            for (std::size_t row = 0; row < 4; ++row) {
              for (std::size_t column = 0; column < 4; ++column) {
                auto const source_row = energy + (row & 1u);
                auto const source_column = loss + (column & 1u);
                auto const* values = &source.spline.values;
                if (row >= 2 && column < 2)
                  values = &source.spline.derivative_axis0;
                else if (row < 2 && column >= 2)
                  values = &source.spline.derivative_axis1;
                else if (row >= 2 && column >= 2)
                  values = &source.spline.mixed_derivative;
                temp(row, column) = element(*values, source_row, source_column);
              }
            }
            Eigen::Matrix4d const coefficients = m.transpose() * (temp * m);
            for (std::size_t row = 0; row < 4; ++row)
              for (std::size_t column = 0; column < 4; ++column)
                result.bicubicPolynomialCoefficients_.push_back(
                    coefficients(row, column));
          }
        }
      }
      return {export_axis(source.axes[0]),
              export_axis(source.axes[1]),
              static_cast<std::uint32_t>(rows),
              static_cast<std::uint32_t>(columns),
              offset,
              polynomial_offset};
    }

    NativeCubicDescriptor append_cubic(
        ProposalNativeTableSet& result, cubic_splines::AxisDescriptor<double> const& axis,
        bool has_transform, cubic_splines::AxisDescriptor<double> const& value_transform,
        cubic_splines::CubicSplines<double>::ExportData const& spline) {
      if (spline.values.size() < 2 ||
          spline.values.size() != spline.node_derivatives.size() ||
          spline.values.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("invalid native cubic coefficient arrays");
      auto const offset = static_cast<std::uint64_t>(result.cubicValues_.size());
      result.cubicValues_.insert(result.cubicValues_.end(), spline.values.begin(),
                                 spline.values.end());
      result.cubicNodeDerivatives_.insert(result.cubicNodeDerivatives_.end(),
                                          spline.node_derivatives.begin(),
                                          spline.node_derivatives.end());
      return {export_axis(axis),
              has_transform ? export_axis(value_transform) : NativeAxisDescriptor{},
              offset, static_cast<std::uint32_t>(spline.values.size()),
              has_transform ? 1u : 0u};
    }

  } // namespace

  NativeKinematicModel classify_proposal_native_kinematic_model(
      std::int32_t process_id, std::string const& parametrization, std::int32_t pdg_id) {
    return classify_kinematic_model(process_id, parametrization, pdg_id);
  }
  ProposalNativeTableSet export_proposal_native_tables(
      std::vector<proposal::NativeInteractionCalculatorView> const& interactions,
      std::vector<proposal::NativeContinuousCalculatorView> const& continuous) {
    ProposalNativeTableSet result;
    result.proposalVersion_ = getPROPOSALVersion();
    result.cubicInterpolationVersion_ = getCubicInterpolationVersion();
    result.constants_ = {PROPOSAL::ME, PROPOSAL::MPI,   PROPOSAL::ALPHA, PROPOSAL::RE,
                         PROPOSAL::PI, PROPOSAL::SQRTE, PROPOSAL::MMU};
    require_single_composition(interactions, continuous);
    for (auto const& view : interactions) {
      if (!supported_native_projectile(view.getProjectile())) continue;
      auto const cut_MeV = static_cast<double>(view.getStochasticEnergyCut() / 1_MeV);
      if (!(cut_MeV > 0.)) continue;
      if (result.stochasticCutMeV_ == 0.)
        result.stochasticCutMeV_ = cut_MeV;
      else if (std::abs(result.stochasticCutMeV_ - cut_MeV) >
               1.e-12 * result.stochasticCutMeV_)
        throw std::invalid_argument(
            "proposal-native calculators use inconsistent stochastic cuts");
    }

    // CORSIKA's five atmosphere nodes construct separate PROPOSAL Medium
    // objects whose hashes include their reference density. Cross sections
    // are rates per grammage and therefore depend on composition, not that
    // reference density. Export one deterministic representative calculator
    // per projectile and leave the local density to GPU tracking/LPM.
    std::map<Code, proposal::NativeInteractionCalculatorView const*> unique_interactions;
    for (auto const& view : interactions) {
      if (!supported_native_projectile(view.getProjectile())) continue;
      if (!borrowed_pointer(view.getMedium()) || !borrowed_pointer(view.getInteraction()))
        throw std::invalid_argument("null PROPOSAL native interaction view");
      auto const found = unique_interactions.find(view.getProjectile());
      if (found == unique_interactions.end() ||
          view.getMediumHash() < found->second->getMediumHash())
        unique_interactions[view.getProjectile()] = &view;
    }

    for (auto const& entry : unique_interactions) {
      auto const& view = *entry.second;
      auto const pdg_id = static_cast<std::int32_t>(get_PDG(view.getProjectile()));
      auto const medium_hash = static_cast<std::uint64_t>(view.getMediumHash());
      result.interactionIdentities_.push_back(
          {pdg_id, 0u, medium_hash,
           static_cast<std::uint64_t>(
               borrowed_pointer(view.getInteraction())->GetHash())});
      std::uint32_t selection_ordinal = 0;
      std::uint32_t dedx_accumulation_ordinal = 0;
      for (auto const& cross_section :
           borrowed_pointer(view.getInteraction())->GetCrossSections()) {
        if (!cross_section)
          throw std::logic_error("null PROPOSAL cross section during native export");
        auto const process_id =
            static_cast<std::int32_t>(cross_section->GetInteractionType());
        auto const parametrization = cross_section->GetParametrizationName();
        auto const process_kinematic_model =
            classify_proposal_native_kinematic_model(process_id, parametrization, pdg_id);
        auto const first_process_column = result.dndxColumns_.size();
        auto dndx = cross_section->ExportDndxInterpolation();
        for (auto const& source : dndx) {
          ++result.proposalCacheTableCount_;
          if (source.cache_hit) ++result.proposalCacheHitCount_;
          auto const component =
              component_for_hash(*borrowed_pointer(view.getMedium()),
                                 static_cast<std::uint64_t>(source.component_hash));
          if (source.parametrization != parametrization)
            throw std::invalid_argument(
                "PROPOSAL native dN/dX export changed its parametrization identity");
          auto const kinematic_model = classify_proposal_native_kinematic_model(
              process_id, source.parametrization, pdg_id);
          if (kinematic_model != process_kinematic_model)
            throw std::logic_error(
                "PROPOSAL native kinematic classification is inconsistent");
          result.dndxColumns_.push_back(
              {pdg_id, process_id, medium_hash,
               static_cast<std::uint64_t>(source.component_hash),
               static_cast<std::uint64_t>(source.cross_section_hash),
               static_cast<std::uint64_t>(source.table_hash),
               export_loss_transform(source.loss_transform), kinematic_model,
               NativeRateModel::BicubicSpline,
               // Pre-expand every native Hermite cell using the same Eigen
               // matrix products as CubicInterpolation 0.1.5.  Earlier
               // revisions did this only for the very small muon rates, but
               // the generic basis form can also perturb a tiny residual
               // quantile for photon/electron selection.  This remains a
               // deterministic layout conversion, not physics resampling.
               append_bicubic(result, source, true), source.component_weight,
               view.getParticleMassMeV(),
               source.component_hash == view.getMediumHash() ? 0.
                                                             : component.GetNucCharge(),
               source.component_hash == view.getMediumHash() ? 0.
                                                             : component.GetAtomicNum(),
               source.component_hash == view.getMediumHash()
                   ? 0.
                   : component.GetAverageNucleonWeight(),
               0., source.lower_energy_limit_MeV, source.absolute_cut_MeV,
               source.relative_cut});
        }

        // ParametrizationDirect processes have no interpolation object to
        // export.  The production CORSIKA configuration uses three closed
        // formulae; export only their immutable component parameters and
        // evaluate the same formula on the device.  This is not resampling.
        if (dndx.empty() && process_id == 1000000012 &&
            cross_section->GetParametrizationName() == "Heitler") {
          for (auto const& component :
               borrowed_pointer(view.getMedium())->GetComponents()) {
            auto const weight =
                borrowed_pointer(view.getMedium())->GetSumNucleons() /
                (component.GetAtomInMolecule() * component.GetAtomicNum());
            auto const factor = PROPOSAL::NA * component.GetNucCharge() /
                                component.GetAtomicNum() * PROPOSAL::PI * PROPOSAL::RE *
                                PROPOSAL::RE / weight;
            result.dndxColumns_.push_back(
                {pdg_id,
                 process_id,
                 medium_hash,
                 static_cast<std::uint64_t>(component.GetHash()),
                 static_cast<std::uint64_t>(cross_section->GetHash()),
                 0,
                 NativeLossTransform::RelativeLog,
                 NativeKinematicModel::OnlyStochastic,
                 NativeRateModel::AnnihilationHeitler,
                 {},
                 1.,
                 view.getParticleMassMeV(),
                 component.GetNucCharge(),
                 component.GetAtomicNum(),
                 component.GetAverageNucleonWeight(),
                 factor,
                 cross_section->GetLowerEnergyLim(),
                 0.,
                 0.});
          }
        } else if (dndx.empty() && process_id == 1000000016 &&
                   cross_section->GetParametrizationName() == "Sauter") {
          for (auto const& component :
               borrowed_pointer(view.getMedium())->GetComponents()) {
            auto const weight =
                borrowed_pointer(view.getMedium())->GetSumNucleons() /
                (component.GetAtomInMolecule() * component.GetAtomicNum());
            auto const factor = PROPOSAL::NA / component.GetAtomicNum() / weight;
            auto const z_alpha = component.GetNucCharge() * PROPOSAL::ALPHA;
            auto const threshold = z_alpha * z_alpha * PROPOSAL::ME / 2.;
            result.dndxColumns_.push_back(
                {pdg_id,
                 process_id,
                 medium_hash,
                 static_cast<std::uint64_t>(component.GetHash()),
                 static_cast<std::uint64_t>(cross_section->GetHash()),
                 0,
                 NativeLossTransform::RelativeLog,
                 NativeKinematicModel::OnlyStochastic,
                 NativeRateModel::PhotoeffectSauter,
                 {},
                 1.,
                 view.getParticleMassMeV(),
                 component.GetNucCharge(),
                 component.GetAtomicNum(),
                 component.GetAverageNucleonWeight(),
                 factor,
                 threshold,
                 0.,
                 0.});
          }
        } else if (dndx.empty() && process_id == 1000000014 &&
                   cross_section->GetParametrizationName() == "HeckC7Shadowing") {
          for (auto const& component :
               borrowed_pointer(view.getMedium())->GetComponents()) {
            auto const weight =
                borrowed_pointer(view.getMedium())->GetSumNucleons() /
                (component.GetAtomInMolecule() * component.GetAtomicNum());
            auto const factor = PROPOSAL::NA / component.GetAtomicNum() * 1.e-30 / weight;
            auto const threshold =
                PROPOSAL::MPI + PROPOSAL::MPI * PROPOSAL::MPI /
                                    (2. * component.GetAverageNucleonWeight());
            result.dndxColumns_.push_back(
                {pdg_id,
                 process_id,
                 medium_hash,
                 static_cast<std::uint64_t>(component.GetHash()),
                 static_cast<std::uint64_t>(cross_section->GetHash()),
                 0,
                 NativeLossTransform::RelativeLog,
                 NativeKinematicModel::OnlyStochastic,
                 NativeRateModel::PhotoproductionHeckC7Shadowing,
                 {},
                 1.,
                 view.getParticleMassMeV(),
                 component.GetNucCharge(),
                 component.GetAtomicNum(),
                 component.GetAverageNucleonWeight(),
                 factor,
                 threshold,
                 0.,
                 0.});
          }
        } else if (dndx.empty()) {
          std::ostringstream message;
          message << "proposal-native has no device representation for direct process "
                  << cross_section->GetParametrizationName() << " (" << process_id << ")";
          throw std::invalid_argument(message.str());
        }

        auto const medium_mean_excitation_energy_MeV =
            1.e-6 * borrowed_pointer(view.getMedium())->GetI();
        if (!(medium_mean_excitation_energy_MeV > 0.) ||
            !std::isfinite(medium_mean_excitation_energy_MeV))
          throw std::invalid_argument("invalid PROPOSAL medium mean excitation energy");
        for (auto index = first_process_column; index < result.dndxColumns_.size();
             ++index) {
          result.dndxColumns_[index].mediumMeanExcitationEnergyMeV_ =
              medium_mean_excitation_energy_MeV;
          result.dndxColumns_[index].selectionOrdinal_ = selection_ordinal++;
        }

        // PROPOSAL represents a process with no interpolated continuous-loss
        // contribution by returning an empty vector.  That is a valid state
        // (notably for direct or stochastic-only processes).  A non-empty
        // export, however, must contain a complete cubic spline; accepting a
        // default/degenerate entry here would silently remove dE/dX on device.
        auto const dedx_sources = cross_section->ExportDedxInterpolation();
        for (auto const& source : dedx_sources) {
          if (source.axis.nodes < 2 || source.spline.values.size() < 2 ||
              source.spline.values.size() != source.spline.node_derivatives.size()) {
            std::ostringstream message;
            message << "invalid non-empty PROPOSAL native dE/dX export: pid=" << pdg_id
                    << ", process=" << process_id
                    << ", parametrization=" << cross_section->GetParametrizationName()
                    << ", axis_nodes=" << source.axis.nodes
                    << ", values=" << source.spline.values.size()
                    << ", derivatives=" << source.spline.node_derivatives.size();
            throw std::invalid_argument(message.str());
          }
          ++result.proposalCacheTableCount_;
          if (source.cache_hit) ++result.proposalCacheHitCount_;
          result.dedxColumns_.push_back(
              {pdg_id, process_id, medium_hash,
               static_cast<std::uint64_t>(source.cross_section_hash),
               static_cast<std::uint64_t>(source.component_index),
               static_cast<std::uint64_t>(source.table_hash),
               append_cubic(result, source.axis, source.has_value_transform,
                            source.value_transform, source.spline),
               source.component_weight, source.lower_energy_limit_MeV,
               dedx_accumulation_ordinal++, 0u});
        }
      }

      auto const total_rate =
          borrowed_pointer(view.getInteraction())->ExportTotalRateInterpolation();
      if (!total_rate.available || total_rate.axis.nodes < 2 ||
          total_rate.spline.values.size() < 2 ||
          total_rate.spline.values.size() != total_rate.spline.node_derivatives.size())
        throw std::invalid_argument(
            "proposal-native requires the interpolated Interaction "
            "mean-free-path rate table");
      ++result.proposalCacheTableCount_;
      if (total_rate.cache_hit) ++result.proposalCacheHitCount_;
      result.totalRateColumns_.push_back(
          {pdg_id, 0u, medium_hash, static_cast<std::uint64_t>(total_rate.table_hash),
           append_cubic(result, total_rate.axis, total_rate.has_value_transform,
                        total_rate.value_transform, total_rate.spline),
           total_rate.lower_energy_limit_MeV});
    }

    std::map<Code, proposal::NativeContinuousCalculatorView const*> unique_continuous;
    for (auto const& view : continuous) {
      if (!supported_native_projectile(view.getProjectile())) continue;
      if (!borrowed_pointer(view.getMedium()) ||
          !borrowed_pointer(view.getDisplacement()))
        throw std::invalid_argument("null PROPOSAL native continuous view");
      auto const found = unique_continuous.find(view.getProjectile());
      if (found == unique_continuous.end() ||
          view.getMediumHash() < found->second->getMediumHash())
        unique_continuous[view.getProjectile()] = &view;
    }
    for (auto const& entry : unique_continuous) {
      auto const& view = *entry.second;
      auto const source = borrowed_pointer(view.getDisplacement())->ExportInterpolation();
      if (!source.available)
        throw std::invalid_argument(
            "proposal-native requires interpolated displacement calculators");
      ++result.proposalCacheTableCount_;
      if (source.cache_hit) ++result.proposalCacheHitCount_;
      result.utilityColumns_.push_back(
          {static_cast<std::int32_t>(get_PDG(view.getProjectile())),
           static_cast<std::uint64_t>(view.getMediumHash()),
           static_cast<std::uint64_t>(source.table_hash),
           append_cubic(result, source.axis, source.has_value_transform,
                        source.value_transform, source.spline),
           view.getParticleMassMeV(), source.lower_energy_limit_MeV,
           source.reverse ? 1u : 0u});
    }

    std::sort(result.dndxColumns_.begin(), result.dndxColumns_.end(),
              [](auto const& left, auto const& right) {
                return std::tie(left.pdgId_, left.mediumHash_, left.processId_,
                                left.componentHash_) <
                       std::tie(right.pdgId_, right.mediumHash_, right.processId_,
                                right.componentHash_);
              });
    result.selectionColumnIndices_.resize(result.dndxColumns_.size());
    std::iota(result.selectionColumnIndices_.begin(),
              result.selectionColumnIndices_.end(), 0u);
    std::sort(result.selectionColumnIndices_.begin(),
              result.selectionColumnIndices_.end(),
              [&](std::uint32_t left_index, std::uint32_t right_index) {
                auto const& left = result.dndxColumns_[left_index];
                auto const& right = result.dndxColumns_[right_index];
                return std::tie(left.pdgId_, left.selectionOrdinal_) <
                       std::tie(right.pdgId_, right.selectionOrdinal_);
              });
    std::sort(
        result.totalRateColumns_.begin(), result.totalRateColumns_.end(),
        [](auto const& left, auto const& right) { return left.pdgId_ < right.pdgId_; });
    std::sort(result.interactionIdentities_.begin(), result.interactionIdentities_.end(),
              [](auto const& left, auto const& right) {
                return std::tie(left.pdgId_, left.mediumHash_, left.interactionHash_) <
                       std::tie(right.pdgId_, right.mediumHash_, right.interactionHash_);
              });
    std::sort(result.dedxColumns_.begin(), result.dedxColumns_.end(),
              [](auto const& left, auto const& right) {
                return std::tie(left.pdgId_, left.mediumHash_, left.processId_,
                                left.crossSectionHash_, left.componentIndex_,
                                left.tableHash_) <
                       std::tie(right.pdgId_, right.mediumHash_, right.processId_,
                                right.crossSectionHash_, right.componentIndex_,
                                right.tableHash_);
              });
    result.dedxAccumulationIndices_.resize(result.dedxColumns_.size());
    std::iota(result.dedxAccumulationIndices_.begin(),
              result.dedxAccumulationIndices_.end(), 0u);
    std::sort(result.dedxAccumulationIndices_.begin(),
              result.dedxAccumulationIndices_.end(),
              [&](std::uint32_t left_index, std::uint32_t right_index) {
                auto const& left = result.dedxColumns_[left_index];
                auto const& right = result.dedxColumns_[right_index];
                return std::tie(left.pdgId_, left.accumulationOrdinal_) <
                       std::tie(right.pdgId_, right.accumulationOrdinal_);
              });
    std::sort(result.utilityColumns_.begin(), result.utilityColumns_.end(),
              [](auto const& left, auto const& right) {
                return std::tie(left.pdgId_, left.mediumHash_, left.tableHash_) <
                       std::tie(right.pdgId_, right.mediumHash_, right.tableHash_);
              });
    result.proposalCacheAllHit_ =
        result.proposalCacheTableCount_ != 0 &&
        result.proposalCacheHitCount_ == result.proposalCacheTableCount_;
    validate_proposal_native_table(result);
    result.contentHash_ = calculate_proposal_native_hash(result);
    return result;
  }

} // namespace corsika::kokkos
