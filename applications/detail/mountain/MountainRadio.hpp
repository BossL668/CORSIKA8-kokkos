/*
 * Internal mountain radio adapter: homogeneous, convex, in-rock paths only.
 * Straight-path physics follows c8_ice_cascade's homogeneous propagator;
 * unlike that application it does not reuse a discretized atmospheric integral.
 */
#pragma once

#include <corsika/accelerator/radio/common/RadioSnapshotBuilder.hpp>
#include <corsika/modules/radio/propagators/RadioPropagator.hpp>
#include <corsika/framework/core/PhysicalConstants.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <stdexcept>
#include <utility>

namespace corsika::applications::mountain {

/** Explicit homogeneous device propagation. No secondary atmospheric table is
 * constructed: both horizontal and inclined paths use n R/c analytically.
 * Observer membership must be checked against the same convex solid used by
 * transport. GPU tracks must terminate at its surface; exterior radio is not
 * supported by this configuration.
 */
template <typename TCoreasDetector, typename TZhsDetector, typename TInside>
gpu::radio::GpuRadioConfig makeHomogeneousMountainRadioConfig(
    double const refractiveIndex, CoordinateSystemPtr const& coordinates,
    TCoreasDetector const& coreas, TZhsDetector const& zhs,
    TInside const& insideConvexRock) {
  if (!std::isfinite(refractiveIndex) || !(refractiveIndex > 0.)) {
    throw std::invalid_argument("mountain radio requires a finite positive refractive index");
  }
  auto check = [&](auto const& detector) {
    for (auto const& observer : detector.getObservers()) {
      if (!insideConvexRock(observer.getLocation())) {
        throw std::invalid_argument("mountain radio observers must be inside the convex rock; external refraction is unsupported");
      }
    }
  };
  check(coreas);
  check(zhs);
  gpu::radio::GpuRadioConfig result;
  result.coreas_enabled = coreas.size() != 0;
  result.zhs_enabled = zhs.size() != 0;
  result.enabled = result.coreas_enabled || result.zhs_enabled;
  result.propagation.homogeneous_refractive_index = refractiveIndex;
  result.coreas_observers =
      gpu::radio::snapshot_detail::makeObservers(coreas, coordinates);
  result.zhs_observers =
      gpu::radio::snapshot_detail::makeObservers(zhs, coordinates);
  return result;
}

/** All source/observer pairs must lie in a convex homogeneous solid.
 * Convexity makes endpoint membership sufficient to keep the complete straight
 * path in rock. Refraction, attenuation, reflection, diffraction and emission
 * after escape are deliberately outside this adapter's contract.
 */
template <typename TEnvironment>
class HomogeneousMountainRadioPropagator final
    : public RadioPropagator<HomogeneousMountainRadioPropagator<TEnvironment>,
                             TEnvironment> {
  using Base = RadioPropagator<HomogeneousMountainRadioPropagator<TEnvironment>,
                               TEnvironment>;
  using SignalPathCollection = typename Base::SignalPathCollection;

 public:
  HomogeneousMountainRadioPropagator(
      TEnvironment const& environment, double const refractiveIndex,
      std::function<bool(Point const&)> insideConvexRock)
      : Base(environment), refractiveIndex_(refractiveIndex),
        insideConvexRock_(std::move(insideConvexRock)) {
    if (!std::isfinite(refractiveIndex_) || !(refractiveIndex_ > 0.) ||
        !insideConvexRock_) {
      throw std::invalid_argument("mountain radio requires finite positive n and a convex rock predicate");
    }
  }

  template <typename TParticle>
  SignalPathCollection propagate(TParticle const&, Point const& source,
                                 Point const& destination) {
    if (!insideConvexRock_(source) || !insideConvexRock_(destination)) {
      throw std::runtime_error("mountain homogeneous radio cannot propagate outside rock or across interfaces");
    }
    auto const displacement = destination - source;
    auto const distance = displacement.getNorm();
    if (!(distance > 0_m) || !std::isfinite(distance / 1_m)) {
      throw std::runtime_error("mountain radio source and observer must have finite nonzero separation");
    }
    auto const emit = displacement.normalized();
    TimeType const time = refractiveIndex_ * distance / constants::c;
    return {SignalPath(time, refractiveIndex_, refractiveIndex_,
                       refractiveIndex_, emit, -emit, distance,
                       std::deque<Point>{source, destination})};
  }

 private:
  double refractiveIndex_;
  std::function<bool(Point const&)> insideConvexRock_;
};

/** Merge raw device E (CoREAS) and A (ZHS) before RadioProcess::endOfShower.
 * No differentiation belongs here: the normal ZHS output path takes exactly
 * one time derivative after CPU and device contributions have been combined.
 */
template <typename TCoreasDetector, typename TZhsDetector>
void mergeMountainGpuRadioWaveforms(
    gpu::radio::GpuRadioWaveforms const& waveforms,
    TCoreasDetector& coreas, TZhsDetector& zhs) {
  auto check = [](auto const& incoming, auto const& detector) {
    auto const& observers = detector.getObservers();
    if (incoming.size() != observers.size()) {
      throw std::runtime_error("mountain radio waveform observer count mismatch");
    }
    for (std::size_t i = 0; i < incoming.size(); ++i) {
      auto const size = observers[i].getWaveformX().size();
      auto const& row = incoming[i];
      if (row.x.size() != size || row.y.size() != size || row.z.size() != size) {
        throw std::runtime_error("mountain radio waveform sample count mismatch");
      }
      for (auto const* axis : {&row.x, &row.y, &row.z}) {
        if (!std::all_of(axis->begin(), axis->end(),
                         [](double value) { return std::isfinite(value); })) {
          throw std::runtime_error("mountain radio waveform contains a non-finite sample");
        }
      }
    }
  };
  // Validate both algorithms before modifying either detector.
  check(waveforms.coreas, coreas);
  check(waveforms.zhs, zhs);
  auto merge = [](auto const& incoming, auto& detector) {
    auto& observers = detector.getObservers();
    for (std::size_t i = 0; i < incoming.size(); ++i) {
      auto const& row = incoming[i];
      observers[i].addWaveform(row.x, row.y, row.z);
    }
  };
  merge(waveforms.coreas, coreas);
  merge(waveforms.zhs, zhs);
}

}  // namespace corsika::applications::mountain
