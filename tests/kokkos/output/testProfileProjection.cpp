/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/utility/LongitudinalCrossings.hpp>
#include <corsika/kokkos/output/algorithms/ProfileProjection.hpp>
#include <tests/kokkos/output/ProfileProjectionReference.hpp>
#include <tests/kokkos/em/transport/TransportChecks.hpp>

namespace current = corsika::kokkos;
namespace reference = profile_projection_reference;

namespace {
  using execution_space_type = transport_test::execution_space_type;
  using memory_space_type = transport_test::memory_space_type;
  void ensure_runtime() { static current::Runtime runtime; }

  struct CrossingInput {
    double start_;
    double end_;
    std::size_t count_;
    int mode_;
  };

  void check_crossing_device(std::vector<CrossingInput> const& inputs) {
    auto input = transport_test::upload<CrossingInput>(inputs);
    Kokkos::View<corsika::LongitudinalCrossingBins*, memory_space_type> output(
        "crossing output", inputs.size());
    Kokkos::View<reference::LongitudinalCrossingBins*, memory_space_type> expected(
        "production crossing output", inputs.size());
    Kokkos::parallel_for(
        "production crossing comparison",
        Kokkos::RangePolicy<execution_space_type>(0, inputs.size()),
        KOKKOS_LAMBDA(std::size_t i) {
          auto const q = input(i);
          output(i) = corsika::longitudinal_profile_bins(
              q.start_, q.end_, q.count_,
              static_cast<corsika::ProfileCrossingMode>(q.mode_));
          expected(i) = reference::longitudinalProfileBins(
              q.start_, q.end_, q.count_,
              static_cast<reference::ProfileCrossingMode>(q.mode_));
        });
    execution_space_type().fence();
    auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
    auto frozen = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, expected);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      auto const q = inputs[i];
      auto const host = corsika::longitudinal_profile_bins(
          q.start_, q.end_, q.count_, static_cast<corsika::ProfileCrossingMode>(q.mode_));
      REQUIRE(actual(i).begin_ == frozen(i).begin);
      REQUIRE(actual(i).end_ == frozen(i).end);
      REQUIRE(actual(i).begin_ == host.begin_);
      REQUIRE(actual(i).end_ == host.end_);
    }
  }

  void compare(current::ProjectedStepRecord const& a,
               reference::ProjectedEmStepRecord const& b) {
    REQUIRE(a.historyId_ == b.history_id);
    REQUIRE(a.pid_ == b.pid);
    REQUIRE(a.processId_ == b.process_id);
    REQUIRE(a.transportLimit_ == b.transport_limit);
    REQUIRE(a.reserved_ == b.reserved);
    REQUIRE(a.startGrammageGPerCm2_ == b.start_grammage_g_per_cm2);
    REQUIRE(a.endGrammageGPerCm2_ == b.end_grammage_g_per_cm2);
    REQUIRE(a.endEnergyGeV_ == b.end_energy_GeV);
    REQUIRE(a.depositedEnergyGeV_ == b.deposited_energy_GeV);
    REQUIRE(a.weight_ == b.weight);
    REQUIRE(a.cutDepositedEnergyGeV_ == b.cut_deposited_energy_GeV);
    REQUIRE(a.observationSurfaceReachedBeforeCut_ ==
            b.observation_surface_reached_before_cut);
    REQUIRE(a.nativeMuonPairCut_ == b.native_muon_pair_cut);
  }

  CORSIKA_HOST_DEVICE current::ProjectedStepRecord project(
      current::ProfileProjectionView const& projection,
      current::PhotonTransportRecord const& input) {
    return current::project_photon_step(projection, input);
  }
  CORSIKA_HOST_DEVICE current::ProjectedStepRecord project(
      current::ProfileProjectionView const& projection,
      current::LeptonTransportRecord const& input) {
    return current::project_lepton_step(projection, input);
  }
  CORSIKA_HOST_DEVICE reference::ProjectedEmStepRecord project(
      reference::DeviceProfileProjection const& projection,
      current::PhotonTransportRecord const& input) {
    return reference::projectPhotonStep(projection, transport_test::convert(input));
  }
  CORSIKA_HOST_DEVICE reference::ProjectedEmStepRecord project(
      reference::DeviceProfileProjection const& projection,
      current::LeptonTransportRecord const& input) {
    return reference::projectLeptonStep(projection, transport_test::convert(input));
  }

  template <class Record>
  void check_projection_device(std::vector<Record> const& inputs,
                               current::ProfileProjectionView projection) {
    auto input = transport_test::upload<Record>(inputs);
    std::vector<double> support(
        projection.axisGrammageGPerCm2_,
        projection.axisGrammageGPerCm2_ + projection.axisSupportCount_);
    auto depth = transport_test::upload<double>(support);
    projection.axisGrammageGPerCm2_ = depth.data();
    reference::DeviceProfileProjection frozenProjection{};
    for (int axis = 0; axis < 3; ++axis) {
      frozenProjection.axis_start_position_m[axis] = projection.axisStartPositionM_[axis];
      frozenProjection.axis_direction[axis] = projection.axisDirection_[axis];
    }
    frozenProjection.axis_step_length_m = projection.axisStepLengthM_;
    frozenProjection.axis_support_count = projection.axisSupportCount_;
    frozenProjection.axis_grammage_g_per_cm2 = depth.data();
    Kokkos::View<current::ProjectedStepRecord*, memory_space_type> output(
        "projected steps", inputs.size());
    Kokkos::View<reference::ProjectedEmStepRecord*, memory_space_type> expected(
        "production projected steps", inputs.size());
    for (bool reverse : {false, true}) {
      std::size_t const count = inputs.size();
      Kokkos::parallel_for(
          "production profile projection comparison",
          Kokkos::RangePolicy<execution_space_type>(0, count),
          KOKKOS_LAMBDA(std::size_t index) {
            auto const i = reverse ? count - 1 - index : index;
            output(i) = project(projection, input(i));
            expected(i) = project(frozenProjection, input(i));
          });
      execution_space_type().fence();
      auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
      auto frozen = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, expected);
      for (std::size_t i = 0; i < inputs.size(); ++i) compare(actual(i), frozen(i));
    }
  }
} // namespace

TEST_CASE("Production profile plane crossings preserve boundary and direction rules",
          "[profile_crossing_production]") {
  ensure_runtime();
#ifdef CORSIKA_EXPECT_CUDA
  STATIC_REQUIRE(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#endif
#ifdef CORSIKA_EXPECT_OPENMP
  STATIC_REQUIRE(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
  double const values[]{-HUGE_VAL, -100.,
                        -1.1,      -1.,
                        -.1,       -0.,
                        0.,        std::nextafter(0., 1.),
                        .1,        std::nextafter(1., 0.),
                        1.,        std::nextafter(1., 2.),
                        1.1,       2.,
                        17.,       100.,
                        HUGE_VAL,  NAN};
  std::size_t cases{};
  std::vector<CrossingInput> inputs;
  for (auto start : values)
    for (auto end : values)
      for (std::size_t count : {0, 1, 2, 17})
        for (int mode : {0, 1, 2, 99}) {
          auto const a = corsika::longitudinal_profile_bins(
              start, end, count, static_cast<corsika::ProfileCrossingMode>(mode));
          auto const b = reference::longitudinalProfileBins(
              start, end, count, static_cast<reference::ProfileCrossingMode>(mode));
          REQUIRE(a.begin_ == b.begin);
          REQUIRE(a.end_ == b.end);
          REQUIRE(a.begin_ <= a.end_);
          REQUIRE(a.end_ <= count);
          inputs.push_back({start, end, count, mode});
          ++cases;
        }
  auto both =
      corsika::longitudinal_profile_bins(1., 3., 10, corsika::ProfileCrossingMode::Both);
  REQUIRE(both.begin_ == 2);
  REQUIRE(both.end_ == 4);
  auto original = corsika::longitudinal_profile_bins(
      1., 3., 10, corsika::ProfileCrossingMode::OriginalC8);
  REQUIRE(original.begin_ == 1);
  REQUIRE(original.end_ == 4);
  auto reverse =
      corsika::longitudinal_profile_bins(3., 1., 10, corsika::ProfileCrossingMode::Both);
  REQUIRE(reverse.begin_ == both.begin_);
  REQUIRE(reverse.end_ == both.end_);
  check_crossing_device(inputs);
  check_crossing_device({});
  std::ofstream("crossing-counts.txt") << "cases " << cases << '\n';
}

TEST_CASE("Production photon and lepton profile projection retains all record fields",
          "[profile_projection_production]") {
  ensure_runtime();
  double const depths[]{0., 2., 9., 15., 29.};
  current::ProfileProjectionView p{{100., -10., 50.}, {.6, 0., .8}, 2., depths, 5};
  reference::DeviceProfileProjection r{{100., -10., 50.}, {.6, 0., .8}, 2., depths, 5};
  std::uint64_t photons{}, leptons{};
  std::vector<current::PhotonTransportRecord> photonInputs;
  std::vector<current::LeptonTransportRecord> leptonInputs;
  for (double s : {-100., -2., 0., .5, 1., 2., 4., 7.5, 8., 10., 100.})
    for (double t : {-2., 0., 1., 8., 30.})
      for (unsigned sample = 0; sample < 8; ++sample) {
        current::PhotonTransportRecord photon{};
        auto& first = photon.start_;
        first.pid_ = 22;
        first.historyId_ = 0xfedcba9876543210ULL + sample;
        first.parentHistoryId_ = 11;
        first.stepId_ = sample;
        first.generation_ = 2;
        first.weight_ = sample == 0 ? 0. : 1.25 * sample;
        first.energyGeV_ = .01;
        first.direction_[2] = 1.;
        for (int axis = 0; axis < 3; ++axis)
          first.positionM_[axis] =
              p.axisStartPositionM_[axis] + s * p.axisDirection_[axis];
        photon.end_ = first;
        for (int axis = 0; axis < 3; ++axis)
          photon.end_.positionM_[axis] += t * p.axisDirection_[axis];
        photon.end_.energyGeV_ = .005;
        photon.cutDepositedEnergyGeV_ = .002 * sample;
        photon.observationSurfaceReachedBeforeCut_ = sample % 2;
        photon.interaction_.processId_ = current::ComptonProcessId;
        for (int limit = 0; limit < 6; ++limit) {
          photon.limit_ = static_cast<current::PhotonTransportLimit>(limit);
          compare(current::project_photon_step(p, photon),
                  reference::projectPhotonStep(r, transport_test::convert(photon)));
          photonInputs.push_back(photon);
          ++photons;
        }
        for (int pid : {11, -11, 13, -13}) {
          current::LeptonTransportRecord lepton{};
          lepton.start_ = first;
          lepton.start_.pid_ = pid;
          if (sample % 3 == 0 && (pid == 13 || pid == -13)) {
            lepton.start_.reserved_ = current::NativeMuonPairCutFlag;
            lepton.start_.energyGeV_ = .1;
            lepton.start_.stepId_ = 0;
          }
          lepton.end_ = photon.end_;
          lepton.end_.pid_ = pid;
          lepton.continuousDepositedEnergyGeV_ = .001 * sample;
          lepton.cutDepositedEnergyGeV_ = sample == 0 ? -.0001 : .002 * sample;
          lepton.multipleScatteringApplied_ = sample % 2;
          lepton.multipleScatteringStatus_ =
              static_cast<decltype(lepton.multipleScatteringStatus_)>(sample % 3);
          lepton.multipleScatteringIterations_ = sample * 7;
          lepton.observationSurfaceReachedBeforeCut_ = sample % 2;
          lepton.interaction_.processId_ = current::BremsProcessId;
          lepton.interaction_.status_ =
              sample % 2 ? current::InteractionStatus::AtRestAnnihilation
                         : current::InteractionStatus::DistanceSampled;
          for (int limit = 0; limit < 9; ++limit) {
            lepton.limit_ = static_cast<current::LeptonTransportLimit>(limit);
            compare(current::project_lepton_step(p, lepton),
                    reference::projectLeptonStep(r, transport_test::convert(lepton)));
            leptonInputs.push_back(lepton);
            ++leptons;
          }
        }
      }
  double const a[]{100., -10., 50.};
  REQUIRE(current::project_profile_grammage(p, a) == 0.);
  check_projection_device(photonInputs, p);
  check_projection_device(leptonInputs, p);
  check_projection_device(std::vector<current::PhotonTransportRecord>{}, p);
  // A single support point clamps both ends to that value in production.
  p.axisSupportCount_ = 1;
  check_projection_device(photonInputs, p);
  check_projection_device(leptonInputs, p);
  std::ofstream("projection-counts.txt")
      << "photon " << photons << "\nlepton " << leptons << '\n';
}
