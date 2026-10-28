/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "TransportChecks.hpp"
#include "LeptonTransportReference.hpp"
#include "../continuous/ContinuousTestData.hpp"
#include "../../environment/EnvironmentQueryTestData.hpp"
#include <corsika/kokkos/em/algorithms/LeptonTransportStep.hpp>
#include <corsika/kokkos/em/execution/NativeTableStorage.hpp>
#include <array>
#include <map>

namespace lepton_transport_test {
  namespace current = corsika::kokkos;
  namespace reference = lepton_transport_reference;
  namespace environment = environment_query_test;
  using transport_test::append;
  using transport_test::check;
  using transport_test::execution_space_type;
  using transport_test::flatten;
  using transport_test::memory_space_type;
  using transport_test::Outcome;

  struct Input {
    current::InteractionRecord interaction_{};
    current::TransportGeometryConfig config_{};
    current::ExternalTransportBoundary external_{};
    double emCutMeV_{.4}, muonCutMeV_{.4};
    bool applyMoliere_{true};
  };
  struct Result {
    current::LeptonTransportRecord record_{};
    current::FallbackRequest fallback_{};
    std::uint32_t state_{};
  };
  struct Expected {
    Outcome record_, fallback_;
    std::uint32_t state_{};
  };
  KOKKOS_INLINE_FUNCTION current::LeptonTransportRecord initial_record() {
    current::LeptonTransportRecord result;
    result.distanceM_ = -17.;
    result.end_.weight_ = 31.;
    return result;
  }
  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION Result actual(current::ProposalNativeDeviceView const& native,
                                       current::EnvironmentView<TMemorySpace> const& view,
                                       Input const& input) {
    Result output;
    output.record_ = initial_record();
    output.fallback_.diagnosticValue0_ = -99.;
    current::ContinuousPhysicsView table(native, input.emCutMeV_, input.muonCutMeV_);
    output.state_ = current::transport_lepton(
        table, input.applyMoliere_, view, input.config_, input.interaction_,
        output.record_, output.fallback_, input.external_);
    return output;
  }
  KOKKOS_INLINE_FUNCTION Expected
  expected(native_table_reference::ProposalNativeDeviceView const& native,
           environment_query_reference::FullView const& view, Input const& input) {
    reference::EnvironmentSnapshot snapshot;
    snapshot.node_tree = view;
    snapshot.node_tree.maximumDeflection_ =
        input.config_.query_.maximumMagneticDeflectionRad_;
    for (int axis = 0; axis < 3; ++axis) {
      snapshot.earth_center_m[axis] = input.config_.earthCenterM_[axis];
      snapshot.point[axis] = input.config_.observation_.pointM_[axis];
      snapshot.normal[axis] = input.config_.observation_.normal_[axis];
    }
    continuous_test::Query query;
    query.emCutMeV_ = input.emCutMeV_;
    query.muonCutMeV_ = input.muonCutMeV_;
    auto record = transport_test::convert(initial_record());
    reference::ProposalFallbackEvent fallback;
    fallback.diagnostic_value0 = -99.;
    auto flag = reference::transportLepton(
        continuous_test::old_view(native, query), input.applyMoliere_, snapshot,
        continuous_test::convert(input.interaction_), record, fallback,
        {input.external_.distanceM_, input.external_.enabled_,
         input.external_.disableObservation_});
    Expected output;
    output.record_ = flatten(record);
    append(output.fallback_, fallback);
    output.state_ = flag;
    return output;
  }
  void compare(Result const& result, Expected const& old) {
    REQUIRE(old.state_ != 99); // Legacy sphere-layer branch must remain unreachable.
    REQUIRE(result.state_ == old.state_);
    check(flatten(result.record_), old.record_);
    Outcome fallback;
    append(fallback, result.fallback_);
    check(fallback, old.fallback_);
  }
  Input input(environment::Query const& query, int pid = 11, double energyMeV = 10000.) {
    Input result;
    result.interaction_ = continuous_test::interaction(pid, energyMeV);
    result.interaction_.status_ = current::InteractionStatus::DistanceSampled;
    result.interaction_.interactionGrammageGPerCm2_ = query.grammageGPerCm2_;
    result.interaction_.decayDistanceM_ = HUGE_VAL;
    auto& particle = result.interaction_.particle_;
    particle.environmentNode_ = query.logicalNode_;
    for (int axis = 0; axis < 3; ++axis) {
      particle.positionM_[axis] = query.positionM_[axis];
      particle.direction_[axis] = query.direction_[axis];
      result.config_.observation_.pointM_[axis] = query.pointM_[axis];
      result.config_.observation_.normal_[axis] = query.normal_[axis];
      result.config_.earthCenterM_[axis] = 2. * axis - 3.;
    }
    result.config_.query_.maximumMagneticDeflectionRad_ = query.maximumDeflection_;
    return result;
  }
  struct Counts {
    std::size_t cases_{}, zeroCut_{}, curved_{}, linear_{}, dualCut_{};
    std::array<std::size_t, 9> limits_{};
    std::map<int, std::size_t> reasons_;
  };
  void run_dataset(environment::Dataset const& data, std::vector<Input> const& inputs,
                   current::ProposalNativeTableSet const& table, Counts& counts) {
    environment::Uploaded<Kokkos::HostSpace> hostGeometry(data);
    environment::Uploaded<memory_space_type> deviceGeometry(data);
    current::NativeTableStorage<execution_space_type> storage;
    if (!table.utilityColumns_.empty()) storage.initialize(table);
    native_query_test::UploadedReference<Kokkos::HostSpace> oldHostStorage(table);
    native_query_test::UploadedReference<memory_space_type> oldDeviceStorage(table);
    auto const hostNative = current::make_proposal_native_host_view(table);
    auto const native = storage.isInitialized() ? storage.getDeviceView()
                                                : current::ProposalNativeDeviceView{};
    auto const oldHost = oldHostStorage.view();
    auto const oldNative = oldDeviceStorage.view();
    auto const view = deviceGeometry.view();
    auto const oldView = deviceGeometry.referenceView();
    auto queries = transport_test::upload(inputs);
    Kokkos::View<Result*, memory_space_type> outputs("lepton_steps", inputs.size());
    Kokkos::View<Expected*, memory_space_type> references("lepton_oracles",
                                                          inputs.size());
    Kokkos::parallel_for(
        "lepton_steps", Kokkos::RangePolicy<execution_space_type>(0, inputs.size()),
        KOKKOS_LAMBDA(std::size_t index) {
          outputs(index) = actual(native, view, queries(index));
          references(index) = expected(oldNative, oldView, queries(index));
        });
    auto results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, outputs);
    auto old = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, references);
    auto after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, queries);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      INFO(data.name_ << " lepton input " << index);
      auto const& request = inputs[index];
      compare(actual(hostNative, hostGeometry.view(), request),
              expected(oldHost, hostGeometry.referenceView(), request));
      compare(results(index), old(index));
      Outcome first, last;
      append(first, request.interaction_);
      append(last, after(index).interaction_);
      check(first, last);
      auto const& result = results(index);
      if (result.state_ == 1)
        ++counts.reasons_[static_cast<int>(result.fallback_.reason_)];
      else {
        ++counts.limits_.at(static_cast<unsigned>(result.record_.limit_));
        if (result.state_ == 2) ++counts.zeroCut_;
        if (result.record_.magneticBendingApplied_) ++counts.curved_;
        if (result.record_.magneticStepStatus_ ==
            static_cast<unsigned>(current::MagneticStepStatus::Linear))
          ++counts.linear_;
        if (result.record_.observationSurfaceReachedBeforeCut_) ++counts.dualCut_;
        REQUIRE(result.record_.end_.historyId_ ==
                request.interaction_.particle_.historyId_);
        REQUIRE(result.record_.multipleScatteringApplied_ == 0);
        REQUIRE(result.record_.interaction_.status_ !=
                current::InteractionStatus::AtRestAnnihilation);
      }
      ++counts.cases_;
    }
  }
  void report(Counts const& counts) {
    std::ofstream file("lepton-transport-counts.txt");
    file << "same_device_production_cases " << counts.cases_ << '\n';
    file << "zero_cut " << counts.zeroCut_ << "\ncurved " << counts.curved_ << "\nlinear "
         << counts.linear_ << "\ndual_cut " << counts.dualCut_ << '\n';
    for (std::size_t index = 0; index < counts.limits_.size(); ++index)
      file << "limit_" << index << ' ' << counts.limits_[index] << '\n';
    for (auto const& value : counts.reasons_)
      file << "fallback_" << value.first << ' ' << value.second << '\n';
  }
} // namespace lepton_transport_test

TEST_CASE("Lepton single steps preserve production NodeTree transport",
          "[transport][lepton]") {
  using namespace lepton_transport_test;
  current::Runtime runtime;
  Counts counts;
  auto simple = environment::simple("missing tables and early returns");
  environment::Query query;
  std::vector<Input> cases;
  for (int pid : {11, -11, 13, -13, 22, 2212}) {
    for (auto status :
         {current::InteractionStatus::DistanceSampled,
          current::InteractionStatus::Selected, current::InteractionStatus::ParticleCut,
          current::InteractionStatus::RequiresReselection}) {
      for (double depth :
           {0., 1., -1., HUGE_VAL, std::numeric_limits<double>::quiet_NaN()}) {
        auto value = input(query, pid);
        value.interaction_.status_ = status;
        value.interaction_.interactionGrammageGPerCm2_ = depth;
        cases.push_back(value);
      }
    }
  }
  for (int pid : {13, -13})
    for (bool split : {false, true}) {
      auto value = input(query, pid, 50.);
      value.interaction_.status_ = current::InteractionStatus::ParticleCut;
      value.interaction_.particle_.reserved_ = current::NativeMuonPairCutFlag;
      value.interaction_.particle_.generation_ = 1;
      value.interaction_.particle_.stepId_ = 0;
      value.applyMoliere_ = split;
      cases.push_back(value);
    }
  run_dataset(simple, cases, {}, counts);
#ifdef CORSIKA_LEPTON_REAL_TABLES
  auto tables = continuous_test::make_dataset();
  auto datasets = environment::datasets();
  datasets.push_back(environment::atmosphere());
  for (auto const& data : datasets) {
    cases.clear();
    for (auto const& q : data.queries_)
      for (int pid : {11, -11, 13, -13}) {
        auto base = input(q, pid);
        for (auto status : {current::InteractionStatus::DistanceSampled,
                            current::InteractionStatus::NoDiscreteInteraction,
                            current::InteractionStatus::ParticleCut}) {
          for (double depth : {q.grammageGPerCm2_, 0., 1.e-9, 1.e8}) {
            auto value = base;
            value.interaction_.status_ = status;
            value.interaction_.interactionGrammageGPerCm2_ = depth;
            cases.push_back(value);
            value.external_ = {.1, true, true};
            cases.push_back(value);
            value.interaction_.decayDistanceM_ = .01;
            cases.push_back(value);
            value.interaction_.particle_.timeS_ = .02;
            cases.push_back(value);
          }
        }
      }
    run_dataset(data, cases, tables.table_, counts);
  }
  // Nonzero parallel fields and the large-gyroradius policy both select a
  // straight step. The physical field must not be reapplied during advance.
  for (int mode : {0, 1}) {
    auto data = environment::simple("nonzero-field straight-step policy");
    data.shapes_[1].radiusM_ = 100.;
    data.models_[0].referenceDensityGPerCm3_ = 1.e-9;
    data.models_[0].magneticFieldT_[0] = 0.;
    data.models_[0].magneticFieldT_[1] = mode == 1 ? 1.e-15 : 0.;
    data.models_[0].magneticFieldT_[2] = mode == 0 ? 5.e-5 : 0.;
    cases.clear();
    for (int pid : {11, -11, 13, -13}) {
      auto value = input(query, pid);
      value.interaction_.status_ = current::InteractionStatus::NoDiscreteInteraction;
      cases.push_back(value);
    }
    run_dataset(data, cases, tables.table_, counts);
    environment::Uploaded<Kokkos::HostSpace> geometry(data);
    auto const native = current::make_proposal_native_host_view(tables.table_);
    for (auto const& value : cases) {
      auto out = actual(native, geometry.view(), value);
      REQUIRE(out.state_ == 0);
      REQUIRE(out.record_.magneticStepStatus_ ==
              static_cast<unsigned>(current::MagneticStepStatus::Linear));
      REQUIRE(out.record_.magneticBendingApplied_ == 0);
      REQUIRE(out.record_.end_.positionM_[0] ==
              value.interaction_.particle_.positionM_[0]);
      REQUIRE(out.record_.end_.direction_[0] ==
              value.interaction_.particle_.direction_[0]);
    }
  }
  // Field-free and curved steps over a dense energy/depth grid, including cuts.
  for (double field : {0., 5.e-5, 1.}) {
    auto data = environment::simple("energy-depth and magnetic-policy grid");
    data.shapes_[1].radiusM_ = 100.;
    data.models_[0].referenceDensityGPerCm3_ = 1.e-6;
    for (int axis = 0; axis < 3; ++axis)
      data.models_[0].magneticFieldT_[axis] = axis == 1 ? field : 0.;
    cases.clear();
    for (int pid : {11, -11, 13, -13})
      for (double kinetic : {.01, .4, .4001, 1., 10., 1.e3, 1.e5, 1.e7})
        for (double depth : {0., 1.e-12, 1.e-6, .001, 1., 1.e4})
          for (bool split : {false, true}) {
            auto value =
                input(query, pid, current::transport_mass_gev(pid) * 1000. + kinetic);
            value.applyMoliere_ = split;
            value.interaction_.interactionGrammageGPerCm2_ = depth;
            cases.push_back(value);
            value.interaction_.particle_.timeS_ = .02;
            cases.push_back(value);
          }
    run_dataset(data, cases, tables.table_, counts);
  }
  {
    auto data = environment::simple("escape, exact competition and zero-length decay");
    data.shapes_[1].radiusM_ = 100.;
    for (int axis = 0; axis < 3; ++axis) data.models_[0].magneticFieldT_[axis] = 0.;
    data.models_[0].referenceDensityGPerCm3_ = 1.e-9;
    cases.clear();
    for (int pid : {11, -11, 13, -13}) {
      auto value = input(query, pid);
      value.external_.disableObservation_ = true;
      value.interaction_.status_ = current::InteractionStatus::NoDiscreteInteraction;
      cases.push_back(value);
    }
    run_dataset(data, cases, tables.table_, counts);
    environment::Uploaded<Kokkos::HostSpace> host(data);
    auto const native = current::make_proposal_native_host_view(tables.table_);
    for (auto const& value : cases) {
      auto out = actual(native, host.view(), value);
      REQUIRE(out.state_ == 0);
      REQUIRE(out.record_.limit_ == current::LeptonTransportLimit::EscapedEnvironment);
      REQUIRE(out.record_.end_.environmentNode_ == 0);
    }
    data.models_[0].referenceDensityGPerCm3_ = .01;
    data.shapes_[1].radiusM_ = 1.e6;
    cases.clear();
    for (int pid : {11, -11, 13, -13}) {
      auto value = input(query, pid);
      value.external_.disableObservation_ = true;
      current::ContinuousPhysicsView view(native, .4, .4);
      auto prep = current::prepare_lepton_continuous_step(view, value.interaction_);
      REQUIRE(prep.fallbackFlag_ == 0);
      value.interaction_.interactionGrammageGPerCm2_ = prep.maximumGrammageGPerCm2_;
      cases.push_back(value); // Equality: interaction, not continuous loss, wins.
      value.interaction_.interactionGrammageGPerCm2_ = 0.;
      value.interaction_.decayDistanceM_ = 0.;
      cases.push_back(value); // For muons, decay wins equality including zero.
    }
    run_dataset(data, cases, tables.table_, counts);
    environment::Uploaded<Kokkos::HostSpace> dense(data);
    for (std::size_t index = 0; index < cases.size(); ++index) {
      auto out = actual(native, dense.view(), cases[index]);
      REQUIRE(out.state_ == 0);
      auto expectedLimit =
          index % 2 && current::is_muon_pid(cases[index].interaction_.particle_.pid_)
              ? current::LeptonTransportLimit::DecayCandidate
              : current::LeptonTransportLimit::InteractionCandidate;
      REQUIRE(out.record_.limit_ == expectedLimit);
    }
  }
#endif
  report(counts);
  REQUIRE(counts.zeroCut_ > 0);
#ifdef CORSIKA_LEPTON_REAL_TABLES
  for (auto count : counts.limits_) REQUIRE(count > 0);
  REQUIRE(counts.curved_ > 0);
  REQUIRE(counts.linear_ > 0);
  REQUIRE(counts.dualCut_ > 0);
#endif
}
