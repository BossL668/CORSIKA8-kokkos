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
#include <Kokkos_Complex.hpp>
#include <chrono>
#include <optional>

namespace corsika::kokkos::radio::firn {
  namespace {
    namespace go = detail::go;
    using complex_type = Kokkos::complex<double>;
    using clock_type = std::chrono::steady_clock;
    template <typename T>
    using view_type = Kokkos::View<T*>;
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
    template <typename T>
    void upload(view_type<T> const& target, std::vector<T> const& source) {
      using host_type = Kokkos::View<T const*, Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      Kokkos::deep_copy(
          Kokkos::subview(target, std::make_pair(std::size_t(0), source.size())),
          host_type(source.data(), source.size()));
    }
    KOKKOS_INLINE_FUNCTION complex_type polar(double magnitude, double phase) {
      return {magnitude * std::cos(phase), magnitude * std::sin(phase)};
    }
    KOKKOS_INLINE_FUNCTION void contribution(go::Path const& path, Child const& child,
                                             double frequency, double timeOrigin,
                                             double surfaceIndex, double exteriorIndex,
                                             complex_type out[3]) {
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
      auto propagation =
          polar(path.geometricSpreadingPerM_ * std::exp(-path.attenuationExponent_),
                -2. * pi * frequency * path.travelTimeS_ + path.phaseOffsetRad_);
      double const retarded =
          child.durationS_ - path.sourceRefractiveIndex_ *
                                 go::dot(path.emitDirection_, child.displacement_) /
                                 c_m_per_s;
      double const argument = pi * frequency * retarded;
      double const sinc = std::abs(argument) < 1.e-8 ? 1. - argument * argument / 6.
                                                     : std::sin(argument) / argument;
      auto common = mu0_h_per_m * elementary_charge_c * child.chargeWeight_ / (4. * pi) *
                    complex_type(0., -2. * pi * frequency) * sinc *
                    polar(1., -2. * pi * frequency * (child.midpointTimeS_ - timeOrigin));
      for (int row = 0; row < 3; ++row) {
        complex_type projected{};
        for (int col = 0; col < 3; ++col)
          projected += propagation * (rs * s[row] * s[col] + rp * po[row] * ps[col]) *
                       child.displacement_[col];
        out[row] = common * projected;
      }
    }

    struct ParentRays {
      view_type<go::Vector> sources_, observers_;
      view_type<go::Query> queries_;
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

    struct ChildRays {
      view_type<Child> children_;
      view_type<go::Vector> observers_;
      view_type<go::Query> queries_, parents_;
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

    struct CoherentSpectrum {
      view_type<Child> children_;
      view_type<go::Query> queries_;
      view_type<complex_type> spectra_, branches_;
      std::size_t observerCount_, frequencyCount_, childCount_, outputCount_;
      double surfaceIndex_;
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
          auto const& q = queries_(child * observerCount_ + o);
          for (int b = 0; b < q.paths_.size(); ++b) {
            complex_type v[3]{};
            auto const& path = q.paths_[b];
            contribution(path, children_(child), frequency, config_.radio_.timeOriginS_,
                         surfaceIndex_, config_.exteriorRefractiveIndex_, v);
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
        addBytes(observers_.size(), sizeof(go::Vector));
        addBytes(outputCount_,
                 sizeof(complex_type) * (config_.radio_.writeBranchComponents_ ? 8 : 1));
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
          failed_ = true;
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
        failed_ = false;
        auto bytes = statistics_.allocatedBytes_;
        statistics_ = {};
        statistics_.allocatedBytes_ = bytes;
        if (ready_) clearSpectra();
      }
      GeometricalOpticsStatistics const& statistics() const final { return statistics_; }
      std::string executionSpaceName() const final {
        return Kokkos::DefaultExecutionSpace::name();
      }

    private:
      void requireHealthy() const {
        if (failed_) throw std::runtime_error("GO event failed; reset before reuse");
      }
      std::size_t frequencies() const { return config_.radio_.grid_.frequencyCount(); }
      void ensureDevice() {
        if (ready_) return;
        runtime_.emplace(threads_, device_);
        deviceObservers_ = view_type<go::Vector>("GO observers", observers_.size());
        std::vector<go::Vector> observers;
        for (auto const& o : observers_) observers.push_back(device_vector(o));
        upload(deviceObservers_, observers);
        sources_ = view_type<go::Vector>("GO parent sources", config_.batchSegments_);
        parents_ = view_type<go::Query>("GO parent geometry", queryCapacity_);
        queries_ = view_type<go::Query>("GO child geometry", queryCapacity_);
        children_ = view_type<Child>("GO children", config_.batchSegments_);
        spectra_ = view_type<complex_type>("GO spectra", outputCount_);
        if (config_.radio_.writeBranchComponents_)
          branchSpectra_ = view_type<complex_type>("GO branch spectra", 7 * outputCount_);
        clearSpectra();
        ready_ = true;
      }
      void clearSpectra() {
        Kokkos::deep_copy(spectra_, complex_type{});
        if (branchSpectra_.extent(0)) Kokkos::deep_copy(branchSpectra_, complex_type{});
      }
      typename view_type<go::Query>::HostMirror initialQueries(
          std::vector<go::Vector> const& sources) {
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
        Kokkos::parallel_for("firn GO parent rays", sources.size() * n,
                             ParentRays{points, observers, queries, n, solver, config});
        Kokkos::fence("firn GO parent rays complete");
        statistics_.geometryMs_ += elapsed_ms(start);
        start = clock_type::now();
        auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, parents_);
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
      void record(go::Path const& path) {
        ++statistics_.branchContributions_;
        ++statistics_.branchCounts_[path.topology_];
        statistics_.minimumTravelTimeS_ =
            std::min(statistics_.minimumTravelTimeS_, path.travelTimeS_);
        statistics_.maximumTravelTimeS_ =
            std::max(statistics_.maximumTravelTimeS_, path.travelTimeS_);
        if (path.fermatRelativeResidual_ >
            statistics_.worstFermatPath_.fermatRelativeResidual_)
          statistics_.worstFermatPath_ = host_path(path);
        statistics_.maximumSnellResidual_ =
            std::max(statistics_.maximumSnellResidual_, path.snellResidual_);
      }
      void flushPending() {
        ensureDevice();
        auto const& radio = config_.radio_;
        std::vector<go::Vector> midpoints;
        for (auto const& t : pending_)
          midpoints.push_back(
              device_vector(add(t.startM_, multiply(subtract(t.endM_, t.startM_), .5))));
        auto geometry = initialQueries(midpoints);
        std::vector<Child> children;
        children.reserve(config_.batchSegments_);
        for (std::size_t i = 0; i < pending_.size(); ++i) {
          auto const& t = pending_[i];
          auto displacement = subtract(t.endM_, t.startM_);
          double const length = norm(displacement),
                       duration = t.endTimeS_ - t.startTimeS_;
          double curvature = 0.;
          for (std::size_t o = 0; o < observers_.size(); ++o) {
            auto const& q = geometry(i * observers_.size() + o);
            checkQuery(q);
            for (int b = 0; b < q.paths_.size(); ++b) {
              double const distance =
                  1. / std::max(q.paths_[b].geometricSpreadingPerM_, 1.e-30);
              curvature = std::max(curvature, radio.grid_.nyquistHz() *
                                                  radio.maximumRefractiveIndex_ * length *
                                                  length / (8. * c_m_per_s * distance));
            }
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
        Kokkos::parallel_for(
            "firn GO child rays", count * n,
            ChildRays{children, observers, queries, parents, n, solver, config});
        Kokkos::fence("firn GO child rays complete");
        statistics_.geometryMs_ += elapsed_ms(start);
        start = clock_type::now();
        auto host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{},
            Kokkos::subview(queries_, std::make_pair(std::size_t(0), count * n)));
        statistics_.transferMs_ += elapsed_ms(start);
        for (std::size_t i = 0; i < count * n; ++i) {
          checkQuery(host(i));
          for (int b = 0; b < host(i).paths_.size(); ++b) record(host(i).paths_[b]);
        }
        auto spectra = spectra_;
        auto branches = branchSpectra_;
        auto values = outputCount_;
        double const surface =
            go::Profile{profile_}.refractiveIndexAtZ(profile_.surfaceZM_);
        start = clock_type::now();
        Kokkos::parallel_for("firn GO coherent spectrum", n * (nf - 1),
                             CoherentSpectrum{children, queries, spectra, branches, n, nf,
                                              count, values, surface, config});
        Kokkos::fence("firn GO spectrum complete");
        statistics_.spectrumMs_ += elapsed_ms(start);
      }
      // Runtime must be destroyed after the device views.
      std::optional<Runtime> runtime_;
      ExponentialOpticalParameters profile_;
      std::vector<vector_type> observers_;
      GeometricalOpticsConfig config_;
      int threads_, device_;
      bool ready_{false}, failed_{false};
      std::size_t queryCapacity_{}, outputCount_{};
      GeometricalOpticsStatistics statistics_{};
      std::vector<GeometricalOpticsTrack> pending_;
      view_type<go::Vector> deviceObservers_, sources_;
      view_type<go::Query> parents_, queries_;
      view_type<Child> children_;
      view_type<complex_type> spectra_, branchSpectra_;
    };
  } // namespace
  std::shared_ptr<IGeometricalOpticsAccumulator> make_geometrical_optics_accumulator(
      ExponentialOpticalParameters profile, std::vector<vector_type> observers,
      GeometricalOpticsConfig config, int threads, int device) {
    return std::make_shared<GeometricalOpticsAccumulator>(profile, std::move(observers),
                                                          config, threads, device);
  }
} // namespace corsika::kokkos::radio::firn
