/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/radio/firn/IGeometricalOpticsAccumulator.hpp>
#include <corsika/kokkos/radio/firn/DeviceStratifiedRaySolver.hpp>
#include <corsika/kokkos/radio/firn/ExponentialProfile.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <corsika/kokkos/radio/algorithms/TimeDomainEmission.hpp>
#include <corsika/kokkos/radio/firn/Fft.hpp>
#include <Kokkos_Complex.hpp>
#include <chrono>
#include <exception>
#include <optional>

namespace corsika::kokkos::radio::firn {
  namespace {
    namespace go = detail::go;
    using complex_type = Kokkos::complex<double>;
    using time_ray_type = time_domain_detail::TimeDomainRay<complex_type>;
    using clock_type = std::chrono::steady_clock;
    template <typename T, typename TExecutionSpace>
    using view_type = Kokkos::View<T*, TExecutionSpace>;
    struct Child {
      go::Vector midpoint_;
      go::Vector displacement_;
      double durationS_;
      double midpointTimeS_;
      double chargeWeight_;
      int reuseParent_{-1};
    };

    double elapsed_ms(clock_type::time_point start) {
      return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
    }
    go::Vector device_vector(vector_type v) { return {v[0], v[1], v[2]}; }
    PathDiagnostics host_path(go::Path const& p) {
      PathDiagnostics out;
      static char const* names[]{
          "direct", "surface_reflected",       "turning_short", "turning_long", "unused",
          "unused", "horizontal_uniform_layer"};
      out.valid_ = p.valid_;
      out.topology_ = names[p.topology_];
      out.invariantP_ = p.invariantP_;
      out.sourceRefractiveIndex_ = p.sourceRefractiveIndex_;
      out.observerRefractiveIndex_ = p.observerRefractiveIndex_;
      out.geometricLengthM_ = p.geometricLengthM_;
      out.opticalPathM_ = p.opticalPathM_;
      out.travelTimeS_ = p.travelTimeS_;
      out.attenuationExponent_ = p.attenuationExponent_;
      out.phaseOffsetRad_ = p.phaseOffsetRad_;
      out.horizontalResidualM_ = p.horizontalResidualM_;
      out.fermatRelativeResidual_ = p.fermatRelativeResidual_;
      out.snellResidual_ = p.snellResidual_;
      out.geometricSpreadingPerM_ = p.geometricSpreadingPerM_;
      for (int i = 0; i < 3; ++i) {
        out.emitDirection_[i] = p.emitDirection_[i];
        out.receiveDirection_[i] = p.receiveDirection_[i];
      }
      return out;
    }
    std::size_t checked_product(std::size_t a, std::size_t b) {
      if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("GO allocation size overflow");
      return a * b;
    }
    template <typename T, typename TExecutionSpace>
    void upload(view_type<T, TExecutionSpace> const& target,
                std::vector<T> const& source) {
      using host_type = Kokkos::View<T const*, Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      Kokkos::deep_copy(
          Kokkos::subview(target, std::make_pair(std::size_t(0), source.size())),
          host_type(source.data(), source.size()));
    }
    KOKKOS_INLINE_FUNCTION complex_type polar(double magnitude, double phase) {
      return {magnitude * std::cos(phase), magnitude * std::sin(phase)};
    }
    struct SpectrumTerm {
      complex_type projection_[3]{};
      double propagationScale_;
      double travelTimeS_;
      double phaseOffsetRad_;
      double retardedDurationS_;
      double emissionTimeS_;
      double chargeWeight_;
      int topology_;
    };
    using PreparedQuery = go::BranchList<SpectrumTerm>;

    // The exported exponential profile has frequency-independent refraction,
    // attenuation and Fresnel coefficients. Reuse their projection at every bin.
    KOKKOS_INLINE_FUNCTION SpectrumTerm prepare_term(go::Path const& path,
                                                     Child const& child,
                                                     double timeOrigin,
                                                     double surfaceIndex,
                                                     double exteriorIndex) {
      auto emit = go::normalized(path.emitDirection_);
      auto receive = go::normalized(path.receiveDirection_);
      go::Vector horizontal{emit[0], emit[1], 0.};
      if (go::norm(horizontal) < 1.e-14) horizontal = {receive[0], receive[1], 0.};
      auto s = go::norm(horizontal) < 1.e-14
                   ? go::Vector{0., 1., 0.}
                   : go::normalized(go::Vector{-horizontal[1], horizontal[0], 0.});
      if (path.topology_ == 1) {
        s = {-path.emitDirection_[1], path.emitDirection_[0], 0.};
        if (go::norm(s) < 1.e-14) s = {0., 1., 0.};
        s = go::normalized(s);
        emit = path.emitDirection_;
        receive = path.receiveDirection_;
      }
      auto ps = go::normalized(go::cross(s, emit));
      auto po = go::normalized(go::cross(s, receive));
      complex_type rs{1., 0.}, rp{1., 0.};
      if (path.topology_ == 1) {
        double const sine = path.invariantP_ / surfaceIndex;
        double const ci = std::sqrt(go::maximum(0., 1. - sine * sine));
        // exp(+i omega t) synthesis: Im(k_z)<0 decays away from the ice.
        auto ct = Kokkos::conj(Kokkos::sqrt(
            complex_type(1. - std::pow(path.invariantP_ / exteriorIndex, 2.), 0.)));
        rs = (surfaceIndex * ci - exteriorIndex * ct) /
             (surfaceIndex * ci + exteriorIndex * ct);
        rp = (exteriorIndex * ci - surfaceIndex * ct) /
             (exteriorIndex * ci + surfaceIndex * ct);
      }
      SpectrumTerm term;
      term.propagationScale_ =
          path.geometricSpreadingPerM_ * std::exp(-path.attenuationExponent_);
      term.travelTimeS_ = path.travelTimeS_;
      term.phaseOffsetRad_ = path.phaseOffsetRad_;
      term.retardedDurationS_ =
          child.durationS_ - path.sourceRefractiveIndex_ *
                                 go::dot(path.emitDirection_, child.displacement_) /
                                 c_m_per_s;
      term.emissionTimeS_ = child.midpointTimeS_ - timeOrigin;
      term.chargeWeight_ = child.chargeWeight_;
      term.topology_ = path.topology_;
      for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
          term.projection_[row] +=
              (rs * s[row] * s[col] + rp * po[row] * ps[col]) * child.displacement_[col];
      return term;
    }
    KOKKOS_INLINE_FUNCTION void contribution(SpectrumTerm const& term, double frequency,
                                             complex_type out[3]) {
      auto propagation =
          polar(term.propagationScale_,
                -2. * pi * frequency * term.travelTimeS_ + term.phaseOffsetRad_);
      double const argument = pi * frequency * term.retardedDurationS_;
      double const sinc = std::abs(argument) < 1.e-8 ? 1. - argument * argument / 6.
                                                     : std::sin(argument) / argument;
      auto common = mu0_h_per_m * elementary_charge_c * term.chargeWeight_ / (4. * pi) *
                    complex_type(0., -2. * pi * frequency) * sinc *
                    polar(1., -2. * pi * frequency * term.emissionTimeS_);
      for (int row = 0; row < 3; ++row)
        out[row] = common * (propagation * term.projection_[row]);
    }

    template <typename TExecutionSpace>
    struct PrepareSpectrum {
      view_type<Child, TExecutionSpace> children_;
      view_type<go::Query, TExecutionSpace> queries_;
      view_type<PreparedQuery, TExecutionSpace> prepared_;
      std::size_t observerCount_;
      double timeOriginS_, surfaceIndex_, exteriorIndex_;
      KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
        PreparedQuery result;
        auto const& query = queries_(i);
        auto const& child = children_(i / observerCount_);
        for (int b = 0; b < query.paths_.size(); ++b)
          result.push_back(prepare_term(query.paths_[b], child, timeOriginS_,
                                        surfaceIndex_, exteriorIndex_));
        prepared_(i) = result;
      }
    };

    template <typename TExecutionSpace>
    struct ParentRays {
      view_type<go::Vector, TExecutionSpace> sources_, observers_;
      view_type<go::Query, TExecutionSpace> queries_;
      std::size_t observerCount_;
      go::RaySolver solver_;
      GeometricalOpticsConfig config_;
      KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
        queries_(i) =
            solver_.evaluate(sources_(i / observerCount_), observers_(i % observerCount_),
                             config_.includeSurfaceReflection_, config_.includeTurning_,
                             config_.foldGuardTimeS_);
      }
    };

    template <typename TExecutionSpace>
    struct ChildRays {
      view_type<Child, TExecutionSpace> children_;
      view_type<go::Vector, TExecutionSpace> observers_;
      view_type<go::Query, TExecutionSpace> queries_, parents_;
      std::size_t observerCount_;
      go::RaySolver solver_;
      GeometricalOpticsConfig config_;
      KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
        auto child = children_(i / observerCount_);
        queries_(i) =
            child.reuseParent_ >= 0
                ? parents_(child.reuseParent_ * observerCount_ + i % observerCount_)
                : solver_.evaluate(child.midpoint_, observers_(i % observerCount_),
                                   config_.includeSurfaceReflection_,
                                   config_.includeTurning_, config_.foldGuardTimeS_);
      }
    };

    struct ParentSummary {
      double maximumSpreadingPerM_{};
      go::Status status_{go::Status::Success};
      bool hasPaths_{false};
    };

    // The host subdivision formula only needs the strongest spreading factor.
    // Keep all ray records resident, and preserve the original host arithmetic
    // and track order for the curvature estimate and ceil-based subdivision.
    template <typename TExecutionSpace>
    struct SummarizeParents {
      view_type<go::Query, TExecutionSpace> queries_;
      view_type<ParentSummary, TExecutionSpace> summaries_;
      std::size_t observerCount_;
      KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
        ParentSummary result;
        for (std::size_t o = 0; o < observerCount_; ++o) {
          auto const& query = queries_(i * observerCount_ + o);
          if (query.status_ != go::Status::Success) {
            result.status_ = query.status_;
            break;
          }
          for (int b = 0; b < query.paths_.size(); ++b) {
            result.hasPaths_ = true;
            double const spreading = query.paths_[b].geometricSpreadingPerM_;
            if (spreading > result.maximumSpreadingPerM_)
              result.maximumSpreadingPerM_ = spreading;
          }
        }
        summaries_(i) = result;
      }
    };

    struct QueryDiagnostics {
      std::uint64_t branchCounts_[7]{};
      double minimumTimeS_{std::numeric_limits<double>::infinity()};
      double maximumTimeS_{};
      double maximumSnellResidual_{};
      double maximumFermatResidual_{};
      std::size_t worstQuery_{std::numeric_limits<std::size_t>::max()};
      int worstBranch_{};
      std::size_t firstFailure_{std::numeric_limits<std::size_t>::max()};
      go::Status status_{go::Status::Success};
    };

    template <typename TExecutionSpace>
    struct SummarizeQueries {
      using value_type = QueryDiagnostics;
      view_type<go::Query, TExecutionSpace> queries_;
      KOKKOS_INLINE_FUNCTION void init(value_type& out) const { out = value_type{}; }
      KOKKOS_INLINE_FUNCTION void join(value_type& out, value_type const& in) const {
        for (int b = 0; b < 7; ++b) out.branchCounts_[b] += in.branchCounts_[b];
        if (in.minimumTimeS_ < out.minimumTimeS_) out.minimumTimeS_ = in.minimumTimeS_;
        if (in.maximumTimeS_ > out.maximumTimeS_) out.maximumTimeS_ = in.maximumTimeS_;
        if (in.maximumSnellResidual_ > out.maximumSnellResidual_)
          out.maximumSnellResidual_ = in.maximumSnellResidual_;
        // Resolve ties in the original query order for reproducible diagnostics.
        if (in.maximumFermatResidual_ > out.maximumFermatResidual_ ||
            (in.maximumFermatResidual_ == out.maximumFermatResidual_ &&
             in.worstQuery_ < out.worstQuery_)) {
          out.maximumFermatResidual_ = in.maximumFermatResidual_;
          out.worstQuery_ = in.worstQuery_;
          out.worstBranch_ = in.worstBranch_;
        }
        if (in.firstFailure_ < out.firstFailure_) {
          out.firstFailure_ = in.firstFailure_;
          out.status_ = in.status_;
        }
      }
      KOKKOS_INLINE_FUNCTION void operator()(std::size_t i, value_type& out) const {
        auto const& query = queries_(i);
        if (query.status_ != go::Status::Success) {
          if (i < out.firstFailure_) {
            out.firstFailure_ = i;
            out.status_ = query.status_;
          }
          return;
        }
        for (int b = 0; b < query.paths_.size(); ++b) {
          auto const& path = query.paths_[b];
          ++out.branchCounts_[path.topology_];
          if (path.travelTimeS_ < out.minimumTimeS_)
            out.minimumTimeS_ = path.travelTimeS_;
          if (path.travelTimeS_ > out.maximumTimeS_)
            out.maximumTimeS_ = path.travelTimeS_;
          if (path.snellResidual_ > out.maximumSnellResidual_)
            out.maximumSnellResidual_ = path.snellResidual_;
          if (path.fermatRelativeResidual_ > out.maximumFermatResidual_ ||
              (path.fermatRelativeResidual_ > 0. &&
               path.fermatRelativeResidual_ == out.maximumFermatResidual_ &&
               (i < out.worstQuery_ || (i == out.worstQuery_ && b < out.worstBranch_)))) {
            out.maximumFermatResidual_ = path.fermatRelativeResidual_;
            out.worstQuery_ = i;
            out.worstBranch_ = b;
          }
        }
      }
    };

    template <typename TExecutionSpace>
    struct CheckEndpoints {
      using value_type = std::size_t;
      view_type<go::Query, TExecutionSpace> queries_, starts_, ends_;
      std::size_t count_;
      KOKKOS_INLINE_FUNCTION void init(value_type& out) const { out = ~value_type{0}; }
      KOKKOS_INLINE_FUNCTION void join(value_type& out, value_type const& in) const {
        if (in < out) out = in;
      }
      KOKKOS_INLINE_FUNCTION void operator()(std::size_t i, value_type& out) const {
        auto const& expected = queries_(i).paths_;
        for (int side = 0; side < 2; ++side) {
          auto const& query = side == 0 ? starts_(i) : ends_(i);
          auto const& actual = query.paths_;
          std::size_t error = static_cast<std::size_t>(query.status_);
          if (query.status_ == go::Status::Success) {
            if (actual.size() != expected.size()) {
              error = 8;
            } else {
              for (int branch = 0; branch < expected.size(); ++branch) {
                int matches = 0;
                for (int candidate = 0; candidate < actual.size(); ++candidate)
                  if (actual[candidate].topology_ == expected[branch].topology_)
                    ++matches;
                if (matches != 1) {
                  error = 9;
                  break;
                }
              }
            }
          }
          if (error) {
            // The old host loop checks every start before checking the ends.
            auto const key = (side * count_ + i) * 16 + error;
            if (key < out) out = key;
          }
        }
      }
    };

    KOKKOS_INLINE_FUNCTION time_ray_type
    prepare_time_ray(go::Path const& path, Child const& child, double timeOrigin,
                     double surfaceIndex, double exteriorIndex, double emissionTime) {
      auto const term =
          prepare_term(path, child, timeOrigin, surfaceIndex, exteriorIndex);
      time_ray_type result;
      auto const transfer = polar(term.propagationScale_, term.phaseOffsetRad_);
      for (int component = 0; component < 3; ++component)
        result.projectedDisplacement_[component] = term.projection_[component] * transfer;
      result.arrivalTimeS_ = emissionTime - timeOrigin + term.travelTimeS_;
      result.retardedDurationS_ = term.retardedDurationS_;
      result.topology_ = term.topology_;
      return result;
    }

    template <typename TExecutionSpace>
    struct EndpointRays {
      view_type<Child, TExecutionSpace> children_;
      view_type<go::Vector, TExecutionSpace> observers_;
      view_type<go::Query, TExecutionSpace> starts_, ends_;
      std::size_t observerCount_;
      go::RaySolver solver_;
      GeometricalOpticsConfig config_;
      KOKKOS_INLINE_FUNCTION void operator()(std::size_t index) const {
        auto const child = children_(index / observerCount_);
        auto start = child.midpoint_, end = child.midpoint_;
        for (int component = 0; component < 3; ++component) {
          start[component] -= .5 * child.displacement_[component];
          end[component] += .5 * child.displacement_[component];
        }
        auto const observer = observers_(index % observerCount_);
        starts_(index) =
            solver_.evaluate(start, observer, config_.includeSurfaceReflection_,
                             config_.includeTurning_, config_.foldGuardTimeS_);
        ends_(index) = solver_.evaluate(end, observer, config_.includeSurfaceReflection_,
                                        config_.includeTurning_, config_.foldGuardTimeS_);
      }
    };

    template <typename TExecutionSpace>
    struct TimeReceiver {
      view_type<double, TExecutionSpace> values_;
      std::size_t observer_, observerCount_, bins_;
      bool branches_;
      KOKKOS_INLINE_FUNCTION void operator()(std::size_t bin, complex_type const* field,
                                             int topology) const {
        for (int component = 0; component < 3; ++component) {
          auto const index = ((observer_ * 3 + component) * bins_ + bin) * 2;
          Kokkos::atomic_add(&values_(index), field[component].real());
          Kokkos::atomic_add(&values_(index + 1), field[component].imag());
          if (branches_) {
            auto const branchIndex =
                index + (topology + 1) * observerCount_ * 3 * bins_ * 2;
            Kokkos::atomic_add(&values_(branchIndex), field[component].real());
            Kokkos::atomic_add(&values_(branchIndex + 1), field[component].imag());
          }
        }
      }
    };

    template <typename TExecutionSpace>
    struct TimeDomainEmission {
      view_type<Child, TExecutionSpace> children_;
      view_type<go::Query, TExecutionSpace> middle_, starts_, ends_;
      view_type<double, TExecutionSpace> waveforms_;
      std::size_t observerCount_;
      double surfaceIndex_;
      GeometricalOpticsConfig config_;
      KOKKOS_INLINE_FUNCTION void operator()(std::size_t index) const {
        auto const child = children_(index / observerCount_);
        auto const& paths = middle_(index).paths_;
        auto const& radio = config_.radio_;
        TimeReceiver<TExecutionSpace> receive{
            waveforms_, index % observerCount_, observerCount_,
            radio.grid_.sampleCount_ + 1, radio.writeBranchComponents_};
        for (int branch = 0; branch < paths.size(); ++branch) {
          auto const middle =
              prepare_time_ray(paths[branch], child, radio.timeOriginS_, surfaceIndex_,
                               config_.exteriorRefractiveIndex_, child.midpointTimeS_);
          if (radio.emissionFormalism_ == EmissionFormalism::ZHS) {
            ::corsika::kokkos::radio::time_domain_detail::deposit_zhs(
                middle, child.chargeWeight_, radio.grid_.sampleIntervalS_,
                radio.grid_.sampleCount_, receive);
          } else {
            // Host validation has checked that these ray families match one-to-one.
            go::Path startPath{}, endPath{};
            for (int candidate = 0; candidate < starts_(index).paths_.size(); ++candidate)
              if (starts_(index).paths_[candidate].topology_ == middle.topology_)
                startPath = starts_(index).paths_[candidate];
            for (int candidate = 0; candidate < ends_(index).paths_.size(); ++candidate)
              if (ends_(index).paths_[candidate].topology_ == middle.topology_)
                endPath = ends_(index).paths_[candidate];
            auto const start =
                prepare_time_ray(startPath, child, radio.timeOriginS_, surfaceIndex_,
                                 config_.exteriorRefractiveIndex_,
                                 child.midpointTimeS_ - .5 * child.durationS_);
            auto const end =
                prepare_time_ray(endPath, child, radio.timeOriginS_, surfaceIndex_,
                                 config_.exteriorRefractiveIndex_,
                                 child.midpointTimeS_ + .5 * child.durationS_);
            ::corsika::kokkos::radio::time_domain_detail::deposit_coreas(
                start, end, middle, child.durationS_, child.chargeWeight_,
                radio.grid_.sampleIntervalS_, radio.grid_.sampleCount_, receive,
                radio.coreasApproximationThreshold_);
          }
        }
      }
    };

    template <typename TExecutionSpace>
    struct CoherentSpectrum {
      view_type<PreparedQuery, TExecutionSpace> prepared_;
      view_type<complex_type, TExecutionSpace> spectra_, branches_;
      std::size_t observerCount_, frequencyCount_, childCount_, outputCount_;
      GeometricalOpticsConfig config_;
      KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
        auto o = i / (frequencyCount_ - 1), f = i % (frequencyCount_ - 1) + 1;
        double const frequency =
            double(f) * (1. / (config_.radio_.grid_.sampleIntervalS_ *
                               config_.radio_.grid_.sampleCount_));
        complex_type total[3]{};
        complex_type family[7][3]{};
        for (int c = 0; c < 3; ++c) {
          auto index = (o * 3 + c) * frequencyCount_ + f;
          total[c] = spectra_(index);
          if (config_.radio_.writeBranchComponents_)
            for (int b = 0; b < 7; ++b)
              family[b][c] = branches_(b * outputCount_ + index);
        }
        for (std::size_t child = 0; child < childCount_; ++child) {
          auto const& q = prepared_(child * observerCount_ + o);
          for (int b = 0; b < q.size(); ++b) {
            complex_type v[3]{};
            auto const& path = q[b];
            contribution(path, frequency, v);
            for (int c = 0; c < 3; ++c) {
              total[c] += v[c];
              if (config_.radio_.writeBranchComponents_)
                family[path.topology_][c] += v[c];
            }
          }
        }
        for (int c = 0; c < 3; ++c) {
          auto index = (o * 3 + c) * frequencyCount_ + f;
          spectra_(index) = total[c];
          if (config_.radio_.writeBranchComponents_)
            for (int b = 0; b < 7; ++b)
              branches_(b * outputCount_ + index) = family[b][c];
        }
      }
    };

    template <typename TExecutionSpace>
    class GeometricalOpticsAccumulator final : public IGeometricalOpticsAccumulator {
    public:
      GeometricalOpticsAccumulator(ExponentialOpticalParameters profile,
                                   std::vector<vector_type> observers,
                                   GeometricalOpticsConfig config, int threads,
                                   int device)
          : profile_(profile)
          , observers_(std::move(observers))
          , config_(config)
          , threads_(threads)
          , device_(device) {
        config_.radio_.validate();
        // Validate the exported model with the CPU model's existing contract.
        ExponentialProfile check(profile.deepIndex_, profile.deltaIndex_,
                                 profile.scaleDepthM_, profile.surfaceZM_,
                                 profile.attenuationPerM_, profile.minimumZM_);
        for (double v :
             {profile.deepIndex_, profile.deltaIndex_, profile.scaleDepthM_,
              profile.surfaceZM_, profile.attenuationPerM_, profile.minimumZM_})
          if (!std::isfinite(v)) throw std::invalid_argument("non-finite GO profile");
        if (observers_.empty() || !config_.batchSegments_ ||
            !(config_.exteriorRefractiveIndex_ >= 1.) ||
            !std::isfinite(config_.exteriorRefractiveIndex_) ||
            !(config_.foldGuardTimeS_ >= 0.) || !std::isfinite(config_.foldGuardTimeS_))
          throw std::invalid_argument("invalid GO configuration");
        for (auto const& o : observers_)
          if (!finite(o) || o[2] < profile.minimumZM_ || o[2] > profile.surfaceZM_)
            throw std::invalid_argument("GO observer lies outside the optical profile");
        queryCapacity_ = checked_product(config_.batchSegments_, observers_.size());
        outputCount_ = checked_product(checked_product(observers_.size(), 3),
                                       config_.radio_.grid_.frequencyCount());
        if (queryCapacity_ > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            outputCount_ > static_cast<std::size_t>(std::numeric_limits<int>::max()))
          throw std::invalid_argument("GO execution range too large");
        std::size_t bytes = 0;
        auto addBytes = [&](std::size_t n, std::size_t width) {
          auto value = checked_product(n, width);
          if (value > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::invalid_argument("GO allocation size overflow");
          bytes += value;
        };
        addBytes(queryCapacity_, 2 * sizeof(go::Query));
        addBytes(config_.batchSegments_, sizeof(Child) + sizeof(go::Vector));
        addBytes(config_.batchSegments_, sizeof(ParentSummary));
        addBytes(1, sizeof(QueryDiagnostics) + sizeof(std::size_t));
        addBytes(observers_.size(), sizeof(go::Vector));
        if (config_.radio_.emissionFormalism_ == EmissionFormalism::FiniteTrackSpectrum) {
          addBytes(queryCapacity_, sizeof(PreparedQuery));
          addBytes(outputCount_, sizeof(complex_type) *
                                     (config_.radio_.writeBranchComponents_ ? 8 : 1));
        } else {
          addBytes(checked_product(checked_product(observers_.size(), 3),
                                   config_.radio_.grid_.sampleCount_ + 1),
                   2 * sizeof(double) * (config_.radio_.writeBranchComponents_ ? 8 : 1));
          if (config_.radio_.emissionFormalism_ == EmissionFormalism::CoREAS)
            addBytes(queryCapacity_, 2 * sizeof(go::Query));
        }
        if (bytes > config_.memoryLimitBytes_)
          throw std::invalid_argument("GO workspace exceeds configured memory limit");
        statistics_.allocatedBytes_ = bytes;
        pending_.reserve(config_.batchSegments_);
      }
      void append(GeometricalOpticsTrack const& track) final {
        requireHealthy();
        if (!finite(track.startM_) || !finite(track.endM_) ||
            !std::isfinite(track.startTimeS_) || !std::isfinite(track.endTimeS_) ||
            !std::isfinite(track.chargeWeight_) ||
            !(track.endTimeS_ > track.startTimeS_) ||
            !std::isfinite(track.endTimeS_ - track.startTimeS_) ||
            !std::isfinite(norm(subtract(track.endM_, track.startM_))) ||
            !(norm(subtract(track.endM_, track.startM_)) > 0.) ||
            track.chargeWeight_ == 0.)
          throw std::invalid_argument("invalid GO finite track");
        for (auto z : {track.startM_[2], track.endM_[2]})
          if (z < profile_.minimumZM_ || z > profile_.surfaceZM_)
            throw std::invalid_argument("GO track lies outside optical profile");
        pending_.push_back(track);
        if (pending_.size() == config_.batchSegments_) flush();
      }
      void flush() final {
        requireHealthy();
        if (pending_.empty()) return;
        try {
          flushPending();
        } catch (...) {
          failure_ = std::current_exception();
          throw;
        }
      }
      std::vector<std::vector<PathDiagnostics>> paths(
          std::vector<vector_type> const& sources) final {
        requireHealthy();
        ensureDevice();
        std::vector<std::vector<PathDiagnostics>> out;
        for (std::size_t begin = 0; begin < sources.size();
             begin += config_.batchSegments_) {
          auto end = std::min(sources.size(), begin + config_.batchSegments_);
          std::vector<go::Vector> batch;
          for (auto i = begin; i < end; ++i) batch.push_back(device_vector(sources[i]));
          auto q = initialQueries(batch);
          for (std::size_t i = 0; i < batch.size() * observers_.size(); ++i) {
            checkQuery(q(i));
            out.emplace_back();
            for (int j = 0; j < q(i).paths_.size(); ++j)
              out.back().push_back(host_path(q(i).paths_[j]));
          }
        }
        return out;
      }
      GeometricalOpticsSpectra download() final {
        flush();
        GeometricalOpticsSpectra result;
        result.total_.resize(observers_.size());
        if (config_.radio_.writeBranchComponents_)
          result.branches_.resize(observers_.size());
        if (!ready_) {
          for (auto& observer : result.total_)
            for (auto& component : observer) component.resize(frequencies());
          for (auto& observer : result.branches_)
            for (auto& family : observer)
              for (auto& component : family) component.resize(frequencies());
          return result;
        }
        if (config_.radio_.emissionFormalism_ != EmissionFormalism::FiniteTrackSpectrum)
          return downloadTimeDomain();
        auto start = clock_type::now();
        auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, spectra_);
        auto branches =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, branchSpectra_);
        for (std::size_t o = 0; o < observers_.size(); ++o)
          for (int c = 0; c < 3; ++c) {
            auto& output = result.total_[o][c];
            output.resize(frequencies());
            for (std::size_t f = 0; f < frequencies(); ++f) {
              auto const index = (o * 3 + c) * frequencies() + f;
              if (ready_) {
                auto v = host(index);
                output[f] = {v.real(), v.imag()};
              }
              if (config_.radio_.writeBranchComponents_)
                for (int b = 0; b < 7; ++b) {
                  auto& values = result.branches_[o][b][c];
                  if (f == 0) values.resize(frequencies());
                  if (ready_) {
                    auto v = branches(b * outputCount_ + index);
                    values[f] = {v.real(), v.imag()};
                  }
                }
            }
          }
        statistics_.transferMs_ += elapsed_ms(start);
        return result;
      }
      void reset() final {
        pending_.clear();
        failure_ = nullptr;
        auto bytes = statistics_.allocatedBytes_;
        statistics_ = {};
        statistics_.allocatedBytes_ = bytes;
        if (ready_) clearSpectra();
      }
      GeometricalOpticsStatistics const& statistics() const final { return statistics_; }
      std::string executionSpaceName() const final { return TExecutionSpace::name(); }

    private:
      using range_type = Kokkos::RangePolicy<TExecutionSpace>;
      GeometricalOpticsSpectra downloadTimeDomain() {
        auto const start = clock_type::now();
        auto host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, timeWaveforms_);
        statistics_.transferMs_ += elapsed_ms(start);
        auto const transformStart = clock_type::now();
        auto const samples = config_.radio_.grid_.sampleCount_;
        auto const interval = config_.radio_.grid_.sampleIntervalS_;
        bool const potential =
            config_.radio_.emissionFormalism_ == EmissionFormalism::ZHS;
        GeometricalOpticsSpectra result;
        result.total_.resize(observers_.size());
        if (config_.radio_.writeBranchComponents_)
          result.branches_.resize(observers_.size());
        std::size_t const families = config_.radio_.writeBranchComponents_ ? 8 : 1;
        for (std::size_t family = 0; family < families; ++family)
          for (std::size_t observer = 0; observer < observers_.size(); ++observer)
            for (int component = 0; component < 3; ++component) {
              std::vector<std::complex<double>> values(samples);
              auto const base =
                  ((family * observers_.size() + observer) * 3 + component) *
                  (samples + 1) * 2;
              for (std::size_t sample = 0; sample < samples; ++sample) {
                auto const index = base + 2 * sample;
                values[sample] = {host(index), host(index + 1)};
                if (potential)
                  values[sample] =
                      (values[sample] -
                       std::complex<double>(host(index + 2), host(index + 3))) /
                      interval;
              }
              fft_in_place(values, false);
              auto& output = family == 0
                                 ? result.total_[observer][component]
                                 : result.branches_[observer][family - 1][component];
              output.resize(frequencies());
              for (std::size_t frequency = 0; frequency < frequencies(); ++frequency) {
                // C8 writes -dA/dt at bin midpoints. Keep that known half-bin offset;
                // no fitted time shift or sign change is applied to comparisons.
                double const phase = potential ? -pi * double(frequency) / samples : 0.;
                output[frequency] = values[frequency] * interval *
                                    std::exp(std::complex<double>(0., phase));
                if (!std::isfinite(output[frequency].real()) ||
                    !std::isfinite(output[frequency].imag()))
                  throw std::runtime_error("non-finite time-domain radio output");
              }
            }
        statistics_.finalizationMs_ += elapsed_ms(transformStart);
        return result;
      }
      void requireHealthy() const {
        // Output finalization must report the original failure, not hide it.
        if (failure_) std::rethrow_exception(failure_);
      }
      std::size_t frequencies() const { return config_.radio_.grid_.frequencyCount(); }
      void ensureDevice() {
        if (ready_) return;
        runtime_.emplace(threads_, device_);
        deviceObservers_ =
            view_type<go::Vector, TExecutionSpace>("GO observers", observers_.size());
        std::vector<go::Vector> observers;
        for (auto const& o : observers_) observers.push_back(device_vector(o));
        upload(deviceObservers_, observers);
        sources_ = view_type<go::Vector, TExecutionSpace>("GO parent sources",
                                                          config_.batchSegments_);
        parents_ =
            view_type<go::Query, TExecutionSpace>("GO parent geometry", queryCapacity_);
        queries_ =
            view_type<go::Query, TExecutionSpace>("GO child geometry", queryCapacity_);
        children_ =
            view_type<Child, TExecutionSpace>("GO children", config_.batchSegments_);
        parentSummaries_ = view_type<ParentSummary, TExecutionSpace>(
            "GO parent summaries", config_.batchSegments_);
        queryDiagnostics_ =
            Kokkos::View<QueryDiagnostics, TExecutionSpace>("GO query diagnostics");
        endpointFailure_ =
            Kokkos::View<std::size_t, TExecutionSpace>("GO endpoint failure");
        if (config_.radio_.emissionFormalism_ == EmissionFormalism::FiniteTrackSpectrum) {
          prepared_ = view_type<PreparedQuery, TExecutionSpace>("GO projected responses",
                                                                queryCapacity_);
          spectra_ = view_type<complex_type, TExecutionSpace>("GO spectra", outputCount_);
          if (config_.radio_.writeBranchComponents_)
            branchSpectra_ = view_type<complex_type, TExecutionSpace>("GO branch spectra",
                                                                      7 * outputCount_);
        } else {
          auto const values =
              observers_.size() * 3 * (config_.radio_.grid_.sampleCount_ + 1) * 2;
          timeWaveforms_ = view_type<double, TExecutionSpace>(
              "firn time-domain quadratures",
              values * (config_.radio_.writeBranchComponents_ ? 8 : 1));
          if (config_.radio_.emissionFormalism_ == EmissionFormalism::CoREAS) {
            startQueries_ = view_type<go::Query, TExecutionSpace>(
                "firn start endpoint rays", queryCapacity_);
            endQueries_ = view_type<go::Query, TExecutionSpace>("firn end endpoint rays",
                                                                queryCapacity_);
          }
        }
        clearSpectra();
        ready_ = true;
      }
      void clearSpectra() {
        if (timeWaveforms_.extent(0)) Kokkos::deep_copy(timeWaveforms_, 0.);
        if (spectra_.extent(0)) Kokkos::deep_copy(spectra_, complex_type{});
        if (branchSpectra_.extent(0)) Kokkos::deep_copy(branchSpectra_, complex_type{});
      }
      void solveParents(std::vector<go::Vector> const& sources) {
        auto start = clock_type::now();
        upload(sources_, sources);
        statistics_.transferMs_ += elapsed_ms(start);
        auto points = sources_;
        auto observers = deviceObservers_;
        auto queries = parents_;
        auto n = observers_.size();
        go::RaySolver solver(profile_);
        auto config = config_;
        start = clock_type::now();
        Kokkos::parallel_for(
            "firn GO parent rays", range_type(0, sources.size() * n),
            ParentRays<TExecutionSpace>{points, observers, queries, n, solver, config});
        Kokkos::fence("firn GO parent rays complete");
        statistics_.geometryMs_ += elapsed_ms(start);
      }
      typename view_type<go::Query, TExecutionSpace>::HostMirror initialQueries(
          std::vector<go::Vector> const& sources) {
        solveParents(sources);
        auto start = clock_type::now();
        // Full records are only requested by the explicit geometry-audit API.
        auto host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{},
            Kokkos::subview(
                parents_,
                std::make_pair(std::size_t(0), sources.size() * observers_.size())));
        statistics_.transferMs_ += elapsed_ms(start);
        return host;
      }
      typename view_type<ParentSummary, TExecutionSpace>::HostMirror parentSummaries(
          std::vector<go::Vector> const& sources) {
        solveParents(sources);
        auto start = clock_type::now();
        Kokkos::parallel_for("firn GO summarize parent rays",
                             range_type(0, sources.size()),
                             SummarizeParents<TExecutionSpace>{parents_, parentSummaries_,
                                                               observers_.size()});
        Kokkos::fence("firn GO parent summaries complete");
        statistics_.geometryMs_ += elapsed_ms(start);
        start = clock_type::now();
        auto host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{},
            Kokkos::subview(parentSummaries_,
                            std::make_pair(std::size_t(0), sources.size())));
        statistics_.transferMs_ += elapsed_ms(start);
        return host;
      }
      void checkQuery(go::Query const& q) {
        if (q.status_ == go::Status::Success) return;
        ++statistics_.failedQueries_;
        char const* reason = "invalid geometry";
        switch (q.status_) {
          case go::Status::OutsideProfile:
            reason = "endpoints outside optical profile";
            break;
          case go::Status::NoPath:
            reason = "no ordinary GO path; full-wave response required";
            break;
          case go::Status::FoldGuard:
            reason = "turning branches inside fold guard; full-wave response required";
            break;
          case go::Status::BranchOverflow:
            reason = "branch capacity exceeded";
            break;
          default:
            break;
        }
        throw std::runtime_error(std::string("Kokkos firn GO: ") + reason);
      }
      void recordQueries(std::size_t count) {
        auto start = clock_type::now();
        Kokkos::parallel_reduce("firn GO summarize child rays", range_type(0, count),
                                SummarizeQueries<TExecutionSpace>{queries_},
                                queryDiagnostics_);
        Kokkos::fence("firn GO child summaries complete");
        statistics_.geometryMs_ += elapsed_ms(start);
        start = clock_type::now();
        auto host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, queryDiagnostics_);
        statistics_.transferMs_ += elapsed_ms(start);
        auto const& summary = host();
        go::Query status;
        status.status_ = summary.status_;
        checkQuery(status);
        for (int b = 0; b < 7; ++b) {
          statistics_.branchCounts_[b] += summary.branchCounts_[b];
          statistics_.branchContributions_ += summary.branchCounts_[b];
        }
        statistics_.minimumTravelTimeS_ =
            std::min(statistics_.minimumTravelTimeS_, summary.minimumTimeS_);
        statistics_.maximumTravelTimeS_ =
            std::max(statistics_.maximumTravelTimeS_, summary.maximumTimeS_);
        statistics_.maximumSnellResidual_ =
            std::max(statistics_.maximumSnellResidual_, summary.maximumSnellResidual_);
        if (summary.maximumFermatResidual_ >
            statistics_.worstFermatPath_.fermatRelativeResidual_) {
          start = clock_type::now();
          auto worst = Kokkos::create_mirror_view_and_copy(
              Kokkos::HostSpace{},
              Kokkos::subview(queries_, std::make_pair(summary.worstQuery_,
                                                       summary.worstQuery_ + 1)));
          statistics_.transferMs_ += elapsed_ms(start);
          statistics_.worstFermatPath_ = host_path(worst(0).paths_[summary.worstBranch_]);
        }
      }
      void flushPending() {
        ensureDevice();
        auto const& radio = config_.radio_;
        std::vector<go::Vector> midpoints;
        for (auto const& t : pending_)
          midpoints.push_back(
              device_vector(add(t.startM_, multiply(subtract(t.endM_, t.startM_), .5))));
        auto geometry = parentSummaries(midpoints);
        std::vector<Child> children;
        children.reserve(config_.batchSegments_);
        for (std::size_t i = 0; i < pending_.size(); ++i) {
          auto const& t = pending_[i];
          auto displacement = subtract(t.endM_, t.startM_);
          double const length = norm(displacement),
                       duration = t.endTimeS_ - t.startTimeS_;
          double curvature = 0.;
          auto const& summary = geometry(i);
          go::Query status;
          status.status_ = summary.status_;
          checkQuery(status);
          if (summary.hasPaths_) {
            double const distance = 1. / std::max(summary.maximumSpreadingPerM_, 1.e-30);
            curvature = radio.grid_.nyquistHz() * radio.maximumRefractiveIndex_ * length *
                        length / (8. * c_m_per_s * distance);
          }
          double const required = std::max(
              1., std::max(std::ceil(length / radio.maximumLinearizationLengthM_),
                           std::ceil(std::sqrt(curvature / radio.maximumPhaseCycles_))));
          if (!std::isfinite(required) || required > radio.maximumSubdivisions_)
            throw std::runtime_error("radio phase subdivision exceeds configured limit");
          auto count = static_cast<std::size_t>(required);
          auto queries = checked_product(count == 1 ? 1 : count + 1, observers_.size());
          if (queries > radio.maximumResponseQueries_ ||
              statistics_.responseQueries_ > radio.maximumResponseQueries_ - queries)
            throw std::runtime_error("radio response-query budget exceeded");
          statistics_.responseQueries_ += queries;
          ++statistics_.inputSegments_;
          statistics_.subdividedSegments_ += count;
          statistics_.maximumSubdivisions_ =
              std::max(statistics_.maximumSubdivisions_, count);
          statistics_.maximumPhaseCycles_ = std::max(statistics_.maximumPhaseCycles_,
                                                     curvature / (double(count) * count));
          for (std::size_t c = 0; c < count; ++c) {
            double const f0 = double(c) / count, f1 = double(c + 1) / count,
                         fm = .5 * (f0 + f1);
            auto start = add(t.startM_, multiply(displacement, f0));
            auto end = add(t.startM_, multiply(displacement, f1));
            children.push_back({device_vector(add(t.startM_, multiply(displacement, fm))),
                                device_vector(subtract(end, start)), duration / count,
                                t.startTimeS_ + fm * duration, t.chargeWeight_,
                                count == 1 ? static_cast<int>(i) : -1});
            if (children.size() == config_.batchSegments_) {
              processChildren(children);
              children.clear();
            }
          }
        }
        if (!children.empty()) processChildren(children);
        pending_.clear();
        ++statistics_.batches_;
      }
      void processChildren(std::vector<Child> const& hostChildren) {
        auto start = clock_type::now();
        upload(children_, hostChildren);
        statistics_.transferMs_ += elapsed_ms(start);
        auto children = children_;
        auto observers = deviceObservers_;
        auto queries = queries_;
        auto parents = parents_;
        auto config = config_;
        auto n = observers_.size(), count = hostChildren.size(), nf = frequencies();
        go::RaySolver solver(profile_);
        start = clock_type::now();
        Kokkos::parallel_for("firn GO child rays", range_type(0, count * n),
                             ChildRays<TExecutionSpace>{children, observers, queries,
                                                        parents, n, solver, config});
        Kokkos::fence("firn GO child rays complete");
        statistics_.geometryMs_ += elapsed_ms(start);
        recordQueries(count * n);
        if (config.radio_.emissionFormalism_ != EmissionFormalism::FiniteTrackSpectrum) {
          if (config.radio_.emissionFormalism_ == EmissionFormalism::CoREAS) {
            start = clock_type::now();
            Kokkos::parallel_for(
                "firn CoREAS endpoint rays", range_type(0, count * n),
                EndpointRays<TExecutionSpace>{children, observers, startQueries_,
                                              endQueries_, n, solver, config});
            Kokkos::fence("firn CoREAS endpoint rays complete");
            statistics_.geometryMs_ += elapsed_ms(start);
            auto const extraQueries = checked_product(count * n, 2);
            if (extraQueries > config.radio_.maximumResponseQueries_ ||
                statistics_.responseQueries_ >
                    config.radio_.maximumResponseQueries_ - extraQueries)
              throw std::runtime_error("CoREAS endpoint query budget exceeded");
            statistics_.responseQueries_ += extraQueries;
            start = clock_type::now();
            Kokkos::parallel_reduce("firn CoREAS check endpoint rays",
                                    range_type(0, count * n),
                                    CheckEndpoints<TExecutionSpace>{
                                        queries_, startQueries_, endQueries_, count * n},
                                    endpointFailure_);
            Kokkos::fence("firn CoREAS endpoint checks complete");
            statistics_.geometryMs_ += elapsed_ms(start);
            start = clock_type::now();
            auto failure = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                               endpointFailure_);
            statistics_.transferMs_ += elapsed_ms(start);
            if (failure() != std::numeric_limits<std::size_t>::max()) {
              auto const error = failure() % 16;
              if (error == 8)
                throw std::runtime_error(
                    "CoREAS ray family changes across track; refine geometry or use full "
                    "wave");
              if (error == 9)
                throw std::runtime_error("CoREAS endpoint ray topology is not unique");
              go::Query status;
              status.status_ = static_cast<go::Status>(error);
              checkQuery(status);
            }
          }
          double const surface =
              go::Profile{profile_}.refractiveIndexAtZ(profile_.surfaceZM_);
          start = clock_type::now();
          Kokkos::parallel_for("firn time-domain emission", range_type(0, count * n),
                               TimeDomainEmission<TExecutionSpace>{
                                   children, queries, startQueries_, endQueries_,
                                   timeWaveforms_, n, surface, config});
          Kokkos::fence("firn time-domain emission complete");
          statistics_.spectrumMs_ += elapsed_ms(start);
          return;
        }
        auto spectra = spectra_;
        auto branches = branchSpectra_;
        auto values = outputCount_;
        double const surface =
            go::Profile{profile_}.refractiveIndexAtZ(profile_.surfaceZM_);
        start = clock_type::now();
        Kokkos::parallel_for(
            "firn GO projected responses", range_type(0, count * n),
            PrepareSpectrum<TExecutionSpace>{children, queries, prepared_, n,
                                             config.radio_.timeOriginS_, surface,
                                             config.exteriorRefractiveIndex_});
        Kokkos::parallel_for(
            "firn GO coherent spectrum", range_type(0, n * (nf - 1)),
            CoherentSpectrum<TExecutionSpace>{prepared_, spectra, branches, n, nf, count,
                                              values, config});
        Kokkos::fence("firn GO spectrum complete");
        statistics_.spectrumMs_ += elapsed_ms(start);
      }
      // Runtime must be destroyed after the device views.
      std::optional<Runtime> runtime_;
      ExponentialOpticalParameters profile_;
      std::vector<vector_type> observers_;
      GeometricalOpticsConfig config_;
      int threads_, device_;
      bool ready_{false};
      std::exception_ptr failure_;
      std::size_t queryCapacity_{}, outputCount_{};
      GeometricalOpticsStatistics statistics_{};
      std::vector<GeometricalOpticsTrack> pending_;
      view_type<go::Vector, TExecutionSpace> deviceObservers_, sources_;
      view_type<go::Query, TExecutionSpace> parents_, queries_;
      view_type<ParentSummary, TExecutionSpace> parentSummaries_;
      Kokkos::View<QueryDiagnostics, TExecutionSpace> queryDiagnostics_;
      Kokkos::View<std::size_t, TExecutionSpace> endpointFailure_;
      view_type<PreparedQuery, TExecutionSpace> prepared_;
      view_type<Child, TExecutionSpace> children_;
      view_type<complex_type, TExecutionSpace> spectra_, branchSpectra_;
      view_type<go::Query, TExecutionSpace> startQueries_, endQueries_;
      view_type<double, TExecutionSpace> timeWaveforms_;
    };
  } // namespace
  std::shared_ptr<IGeometricalOpticsAccumulator> make_geometrical_optics_accumulator(
      ExponentialOpticalParameters profile, std::vector<vector_type> observers,
      GeometricalOpticsConfig config, int threads, int device, bool useHostExecution) {
    if (useHostExecution)
      return std::make_shared<
          GeometricalOpticsAccumulator<Kokkos::DefaultHostExecutionSpace>>(
          profile, std::move(observers), config, threads, device);
    return std::make_shared<GeometricalOpticsAccumulator<Kokkos::DefaultExecutionSpace>>(
        profile, std::move(observers), config, threads, device);
  }
} // namespace corsika::kokkos::radio::firn
