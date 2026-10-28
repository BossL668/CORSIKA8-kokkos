/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/execution/PhysicsContext.hpp>
#include <corsika/kokkos/environment/EnvironmentSearch.hpp>
#include <corsika/kokkos/environment/EnvironmentStorage.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include <catch2/catch_test_macros.hpp>

namespace corsika::kokkos::testing {
  void logPhysicsContextPublications(int publications, int showerUpdates);
}

namespace {
  namespace cp = corsika::kokkos;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using context_type = cp::PhysicsContext<execution_space_type>;

#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#else
  static_assert(Kokkos::is_execution_space<execution_space_type>::value);
#endif

  void ensure_runtime() { static cp::Runtime runtime; }

  // Sentinel arrays verify publication and lifetime, not interpolation accuracy.
  struct Inputs {
    Kokkos::View<double*, memory_space_type> values_{"context borrowed values", 2};
    Kokkos::View<cp::MoliereCubicPolynomial*, memory_space_type> polynomials_{
        "context borrowed polynomials",
        cp::MoliereInterpolationFunctionCount* cp::MoliereInterpolationIntervalCount};
    cp::EnvironmentStorage<execution_space_type> environment_;
    cp::PhysicsContextData expected_{};

    explicit Inputs(double marker) {
      auto values = Kokkos::create_mirror_view(values_);
      values(0) = marker;
      values(1) = marker + 1.;
      Kokkos::deep_copy(values_, values);
      auto polynomials = Kokkos::create_mirror_view(polynomials_);
      for (std::size_t polynomial = 0; polynomial < polynomials.extent(0); ++polynomial) {
        for (int coefficient = 0; coefficient < 4; ++coefficient) {
          polynomials(polynomial).c_[coefficient] = marker + polynomial + coefficient;
        }
      }
      Kokkos::deep_copy(polynomials_, polynomials);
      cp::EnvironmentNode node;
      cp::EnvironmentShape shape;
      shape.kind_ = cp::EnvironmentShape::Kind::Universe;
      cp::EnvironmentDataView source;
      source.nodes_ = &node;
      source.shapes_ = &shape;
      source.nodeCount_ = 1;
      environment_ = cp::EnvironmentStorage<execution_space_type>(source);
      expected_.environment_ = cp::EnvironmentBorrowedView(environment_.getDataView());
      cp::NativeArrayView<double> numerical(values_);
      cp::ProposalNativeDeviceView tables({}, {}, {}, {}, {}, {}, numerical, numerical,
                                          numerical, numerical, numerical, numerical,
                                          numerical, {});
      expected_.physics_ = cp::InteractionPhysicsView(
          cp::ContinuousPhysicsView(tables, marker + 2., marker + 3.), 7u);
      expected_.moliereInterpolation_ = cp::MoliereInterpolationView(
          cp::NativeArrayView<cp::MoliereCubicPolynomial>(polynomials_));
      for (std::size_t coefficient = 0; coefficient < cp::MoliereSeriesCoefficientCount;
           ++coefficient) {
        expected_.electronMoliere_.c1_[coefficient] = marker + 100. + coefficient;
        expected_.muonMoliere_.c2_[coefficient] = marker + 200. + coefficient;
      }
      expected_.electronMoliere_.particleMassMeV_ = .51099895;
      expected_.muonMoliere_.particleMassMeV_ = 105.6583755;
      expected_.photonPairLpm_.eLpmMeV_ = marker + 300.;
      expected_.bremsLpm_.muonELpmMeV_ = marker + 400.;
      expected_.geometry_.query_.maximumMagneticDeflectionRad_ = .2;
      expected_.geometry_.observation_.pointM_[2] = marker + 500.;
      expected_.geometry_.observation_.normal_[2] = 1.;
      expected_.geometry_.earthCenterM_[1] = marker + 600.;
      expected_.profileProjection_.axisGrammageGPerCm2_ = values_.data();
      expected_.profileProjection_.axisSupportCount_ = 2;
      expected_.profileProjection_.axisStepLengthM_ = marker + 700.;
      expected_.profileProjection_.axisDirection_[2] = -1.;
      expected_.thinning_.enabled_ = 1;
      expected_.thinning_.thresholdGeV_ = marker + 800.;
      expected_.thinning_.maximumWeight_ = marker + 900.;
      expected_.randomSeed_ = (std::uint64_t{1} << 63) + 97;
      expected_.showerId_ = (std::uint64_t{1} << 61) + 53;
      expected_.muonMoliereAvailable_ = marker > 1. ? 0 : 1;
    }
  };

  void initialize(context_type& context, cp::PhysicsContextData const& x,
                  execution_space_type const& execution) {
    context.initialize(x.physics_, x.environment_, x.geometry_, x.electronMoliere_,
                       x.muonMoliere_, x.moliereInterpolation_,
                       x.muonMoliereAvailable_ != 0, x.photonPairLpm_, x.bremsLpm_,
                       x.thinning_, x.profileProjection_, x.randomSeed_, x.showerId_,
                       execution);
  }

  void check(context_type const& context, cp::PhysicsContextData const& expected,
             double marker, execution_space_type const& execution) {
    auto view = context.getDeviceView();
    Kokkos::View<std::uint64_t * [5], memory_space_type> keys("context keys", 1);
    Kokkos::View<double * [10], memory_space_type> values("context observations", 1);
    Kokkos::parallel_for(
        "read context after publication",
        Kokkos::RangePolicy<execution_space_type>(execution, 0, 1), KOKKOS_LAMBDA(int) {
          auto const& x = view();
          keys(0, 0) = x.randomSeed_;
          keys(0, 1) = x.showerId_;
          keys(0, 2) = x.muonMoliereAvailable_;
          keys(0, 3) = x.physics_.getPhysicsSource();
          double const point[3]{};
          keys(0, 4) = cp::locate_node(x.environment_, point);
          values(0, 0) =
              x.physics_.getContinuousView().getNativeView().getCubicValues()[1];
          values(0, 1) = x.profileProjection_.axisGrammageGPerCm2_[1];
          values(0, 2) = x.moliereInterpolation_.getF2()[0].c_[0];
          values(0, 3) = x.electronMoliere_.c1_[cp::MoliereSeriesCoefficientCount - 1];
          values(0, 4) = x.muonMoliere_.c2_[cp::MoliereSeriesCoefficientCount - 1];
          values(0, 5) = x.photonPairLpm_.eLpmMeV_;
          values(0, 6) = x.bremsLpm_.muonELpmMeV_;
          values(0, 7) = x.geometry_.observation_.pointM_[2];
          values(0, 8) = x.geometry_.query_.maximumMagneticDeflectionRad_;
          values(0, 9) = x.thinning_.thresholdGeV_;
        });
    auto hostKeys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, keys);
    auto hostValues = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);
    REQUIRE(hostKeys(0, 0) == expected.randomSeed_);
    REQUIRE(hostKeys(0, 1) == expected.showerId_);
    REQUIRE(hostKeys(0, 2) == expected.muonMoliereAvailable_);
    REQUIRE(hostKeys(0, 3) == expected.physics_.getPhysicsSource());
    REQUIRE(hostKeys(0, 4) == 0);
    REQUIRE(hostValues(0, 0) == marker + 1.);
    REQUIRE(hostValues(0, 1) == marker + 1.);
    REQUIRE(hostValues(0, 2) == marker + cp::MoliereInterpolationIntervalCount);
    REQUIRE(hostValues(0, 3) ==
            expected.electronMoliere_.c1_[cp::MoliereSeriesCoefficientCount - 1]);
    REQUIRE(hostValues(0, 4) ==
            expected.muonMoliere_.c2_[cp::MoliereSeriesCoefficientCount - 1]);
    REQUIRE(hostValues(0, 5) == expected.photonPairLpm_.eLpmMeV_);
    REQUIRE(hostValues(0, 6) == expected.bremsLpm_.muonELpmMeV_);
    REQUIRE(hostValues(0, 7) == expected.geometry_.observation_.pointM_[2]);
    REQUIRE(hostValues(0, 8) == expected.geometry_.query_.maximumMagneticDeflectionRad_);
    REQUIRE(hostValues(0, 9) == expected.thinning_.thresholdGeV_);
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
    REQUIRE(host().physics_.getContinuousView().getEmTransportCutMeV() == marker + 2.);
    REQUIRE(host().physics_.getContinuousView().getMuonTransportCutMeV() == marker + 3.);
    REQUIRE(host().thinning_.maximumWeight_ == expected.thinning_.maximumWeight_);
    REQUIRE(host().thinning_.enabled_ == expected.thinning_.enabled_);
    REQUIRE(host().thinning_.eraseZeroWeight_ == expected.thinning_.eraseZeroWeight_);
    REQUIRE(host().geometry_.earthCenterM_[1] == expected.geometry_.earthCenterM_[1]);
    REQUIRE(host().geometry_.observation_.normal_[2] == 1.);
    REQUIRE(host().profileProjection_.axisSupportCount_ == 2);
    REQUIRE(host().profileProjection_.axisStepLengthM_ ==
            expected.profileProjection_.axisStepLengthM_);
    REQUIRE(host().profileProjection_.axisDirection_[2] == -1.);
  }
} // namespace

TEST_CASE("Physics context rejects shower updates before initialization",
          "[kokkos][context]") {
  ensure_runtime();
  execution_space_type execution;
  context_type context;
  REQUIRE_FALSE(context.isInitialized());
  REQUIRE(context.getDeviceBytes() == 0);
  REQUIRE(context.getDeviceView().data() == nullptr);
  REQUIRE(context.projectedDeviceBytes() == sizeof(cp::PhysicsContextData));
  REQUIRE_THROWS_AS(context.beginShower({}, 7, 11, execution), std::logic_error);
}

TEST_CASE("Physics context publishes keys and preserves borrowed inputs",
          "[kokkos][context]") {
  ensure_runtime();
  execution_space_type execution;
  context_type context;
  std::size_t checks{};
  cp::PhysicsContextData const* firstAllocation{};
  for (double marker : {1., 1001.}) {
    Inputs input(marker);
    auto expected = input.expected_;
    initialize(context, expected, execution);
    REQUIRE(context.isInitialized());
    REQUIRE(context.getDeviceBytes() == context.projectedDeviceBytes());
    auto const address = context.getDeviceView().data();
    if (firstAllocation) { REQUIRE(address == firstAllocation); }
    firstAllocation = address;
    check(context, expected, marker, execution);
    ++checks;
    for (std::uint64_t shower = 0; shower < 32; ++shower) {
      expected.randomSeed_ += 17;
      expected.showerId_ += 19;
      expected.thinning_.enabled_ = shower % 2;
      expected.thinning_.eraseZeroWeight_ = (shower + 1) % 2;
      expected.thinning_.thresholdGeV_ += 1.;
      expected.thinning_.maximumWeight_ += 2.;
      context.beginShower(expected.thinning_, expected.randomSeed_, expected.showerId_,
                          execution);
      REQUIRE(context.getDeviceView().data() == address);
      check(context, expected, marker, execution);
      ++checks;
    }
    // All uses finish before the borrowed owners leave this iteration.
    execution.fence("context preview owner lifetime");
  }
  REQUIRE(checks == 66);
  cp::testing::logPhysicsContextPublications(checks, checks - 2);
}
