/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/radio/firn/IFullWaveAccumulator.hpp>
#include <corsika/kokkos/radio/firn/Sha256.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <Kokkos_Core.hpp>
#include <cnpy.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace corsika::kokkos::radio::firn {
  namespace {
    template <typename TValue>
    class DeviceBuffer {
    public:
      using view_type = Kokkos::View<TValue*>;
      void allocate(std::size_t count) {
        view_ = view_type(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "firn full-wave"), count);
      }
      TValue* get() const { return view_.data(); }
      std::size_t bytes() const { return view_.extent(0) * sizeof(TValue); }
      view_type view_;
    };
    using clock_type = std::chrono::steady_clock;
    double elapsed_ms(clock_type::time_point start) {
      return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
    }
    struct DeviceChild {
      double midpoint_[3];
      double displacement_[3];
      double midpointTimeS_;
      double chargeWeight_;
    };

    struct FrequencyInterpolation {
      std::uint32_t node0_;
      std::uint32_t node1_;
      double a_;
      double b_;
      double second0_;
      double second1_;
      double frequencyHz_;
    };

    struct DeviceQuery {
      std::uint32_t zi_;
      std::uint32_t ri_;
      double wz_;
      double wr_;
      double radialX_;
      double radialY_;
      double directPathM_;
      double boundaryPathM_;
      std::uint32_t valid_;
    };

    struct DeviceTable {
      double const* sourceZ_;
      double const* sourceIndex_;
      double const* radiusM_;
      double const* directPathM_;
      double const* boundaryPathM_;
      double const* directReal_;
      double const* directImag_;
      double const* directSecondReal_;
      double const* directSecondImag_;
      double const* boundaryReal_;
      double const* boundaryImag_;
      double const* boundarySecondReal_;
      double const* boundarySecondImag_;
      std::uint32_t sourceCount_;
      std::uint32_t radiusCount_;
      std::uint32_t nodeCount_;
      double observerZM_;
    };

    struct ComplexValue {
      double real_;
      double imag_;
    };

    KOKKOS_INLINE_FUNCTION ComplexValue make_complex(double real, double imag) {
      return {real, imag};
    }
    KOKKOS_INLINE_FUNCTION void sine_cosine(double angle, double* sine, double* cosine) {
#if defined(__CUDA_ARCH__)
      ::sincos(angle, sine, cosine);
#else
      *sine = ::sin(angle);
      *cosine = ::cos(angle);
#endif
    }

    KOKKOS_INLINE_FUNCTION ComplexValue add(ComplexValue a, ComplexValue b) {
      return {a.real_ + b.real_, a.imag_ + b.imag_};
    }

    KOKKOS_INLINE_FUNCTION ComplexValue multiply(ComplexValue a, ComplexValue b) {
      return {a.real_ * b.real_ - a.imag_ * b.imag_,
              a.real_ * b.imag_ + a.imag_ * b.real_};
    }

    KOKKOS_INLINE_FUNCTION ComplexValue scale(ComplexValue value, double factor) {
      return {value.real_ * factor, value.imag_ * factor};
    }

    KOKKOS_INLINE_FUNCTION std::uint32_t bracket(double const* axis, std::uint32_t count,
                                                 double value) {
      std::uint32_t lo = 0;
      std::uint32_t hi = count - 1;
      while (hi - lo > 1) {
        std::uint32_t const middle = lo + (hi - lo) / 2;
        if (value < axis[middle])
          hi = middle;
        else
          lo = middle;
      }
      return lo < count - 2 ? lo : count - 2;
    }

    KOKKOS_INLINE_FUNCTION double bilinear_surface(double const* values,
                                                   DeviceTable const& table,
                                                   DeviceQuery const& query) {
      auto value = [&](std::uint32_t dz, std::uint32_t dr) {
        return values[(query.zi_ + dz) * table.radiusCount_ + query.ri_ + dr];
      };
      double const lower = (1.0 - query.wr_) * value(0, 0) + query.wr_ * value(0, 1);
      double const upper = (1.0 - query.wr_) * value(1, 0) + query.wr_ * value(1, 1);
      return (1.0 - query.wz_) * lower + query.wz_ * upper;
    }

    KOKKOS_INLINE_FUNCTION void build_query(std::size_t flat, DeviceChild const* children,
                                            std::size_t childCount,
                                            double const* observers,
                                            std::size_t observerCount, DeviceTable table,
                                            DeviceQuery* queries, unsigned int* invalid) {
      std::size_t const count = childCount * observerCount;
      if (flat >= count) return;
      std::size_t const childIndex = flat / observerCount;
      std::size_t const observerIndex = flat % observerCount;
      DeviceChild const child = children[childIndex];
      double const* observer = observers + 3 * observerIndex;
      double const dx = observer[0] - child.midpoint_[0];
      double const dy = observer[1] - child.midpoint_[1];
      double const radius = hypot(dx, dy);
      double const z = child.midpoint_[2];
      bool const covered = z >= table.sourceZ_[0] &&
                           z <= table.sourceZ_[table.sourceCount_ - 1] &&
                           radius >= table.radiusM_[0] &&
                           radius <= table.radiusM_[table.radiusCount_ - 1] &&
                           fabs(observer[2] - table.observerZM_) <= 1.0e-9;
      DeviceQuery query{};
      query.valid_ = covered ? 1U : 0U;
      if (!covered) {
        Kokkos::atomic_add(invalid, 1U);
        queries[flat] = query;
        return;
      }
      query.zi_ = bracket(table.sourceZ_, table.sourceCount_, z);
      query.ri_ = bracket(table.radiusM_, table.radiusCount_, radius);
      query.wz_ = (z - table.sourceZ_[query.zi_]) /
                  (table.sourceZ_[query.zi_ + 1] - table.sourceZ_[query.zi_]);
      query.wr_ = (radius - table.radiusM_[query.ri_]) /
                  (table.radiusM_[query.ri_ + 1] - table.radiusM_[query.ri_]);
      query.radialX_ = radius > 0.0 ? dx / radius : 1.0;
      query.radialY_ = radius > 0.0 ? dy / radius : 0.0;
      query.directPathM_ = bilinear_surface(table.directPathM_, table, query);
      query.boundaryPathM_ = bilinear_surface(table.boundaryPathM_, table, query);
      queries[flat] = query;
    }

    KOKKOS_INLINE_FUNCTION std::size_t table_index(DeviceTable const& table,
                                                   std::uint32_t zi, std::uint32_t ri,
                                                   std::uint32_t row, std::uint32_t col,
                                                   std::uint32_t node) {
      return (((static_cast<std::size_t>(zi) * table.radiusCount_ + ri) * 3 + row) * 3 +
              col) *
                 table.nodeCount_ +
             node;
    }

    KOKKOS_INLINE_FUNCTION double cubic_spatial(double const* values,
                                                double const* second,
                                                DeviceTable const& table,
                                                FrequencyInterpolation const& frequency,
                                                DeviceQuery const& query,
                                                std::uint32_t row, std::uint32_t col) {
      double result = 0.0;
      for (std::uint32_t dz = 0; dz < 2; ++dz) {
        double const weightZ = dz ? query.wz_ : 1.0 - query.wz_;
        for (std::uint32_t dr = 0; dr < 2; ++dr) {
          double const weightR = dr ? query.wr_ : 1.0 - query.wr_;
          std::uint32_t const z = query.zi_ + dz;
          std::uint32_t const r = query.ri_ + dr;
          std::size_t const index0 = table_index(table, z, r, row, col, frequency.node0_);
          std::size_t const index1 = table_index(table, z, r, row, col, frequency.node1_);
          double const interpolated =
              frequency.a_ * values[index0] + frequency.b_ * values[index1] +
              frequency.second0_ * second[index0] + frequency.second1_ * second[index1];
          result += weightZ * weightR * interpolated;
        }
      }
      return result;
    }

    KOKKOS_INLINE_FUNCTION void project_branch(DeviceTable const& table,
                                               FrequencyInterpolation const& frequency,
                                               DeviceQuery const& query,
                                               DeviceChild const& child, bool direct,
                                               ComplexValue output[3]) {
      double const* real = direct ? table.directReal_ : table.boundaryReal_;
      double const* imag = direct ? table.directImag_ : table.boundaryImag_;
      double const* secondReal =
          direct ? table.directSecondReal_ : table.boundarySecondReal_;
      double const* secondImag =
          direct ? table.directSecondImag_ : table.boundarySecondImag_;
      double const pathM = direct ? query.directPathM_ : query.boundaryPathM_;
      double const localDisplacement[3]{query.radialX_ * child.displacement_[0] +
                                            query.radialY_ * child.displacement_[1],
                                        -query.radialY_ * child.displacement_[0] +
                                            query.radialX_ * child.displacement_[1],
                                        child.displacement_[2]};
      double restoreSin = 0.0;
      double restoreCos = 0.0;
      double const restorePhase = -2.0 * pi * frequency.frequencyHz_ * pathM / c_m_per_s;
      sine_cosine(restorePhase, &restoreSin, &restoreCos);
      ComplexValue const restore{restoreCos, restoreSin};
      ComplexValue localProjected[3]{};
      for (std::uint32_t row = 0; row < 3; ++row) {
        for (std::uint32_t col = 0; col < 3; ++col) {
          ComplexValue reduced{
              cubic_spatial(real, secondReal, table, frequency, query, row, col),
              cubic_spatial(imag, secondImag, table, frequency, query, row, col)};
          ComplexValue const green =
              scale(multiply(reduced, restore), 1.0 / frequency.frequencyHz_);
          localProjected[row] =
              add(localProjected[row], scale(green, localDisplacement[col]));
        }
      }
      output[0] = add(scale(localProjected[0], query.radialX_),
                      scale(localProjected[1], -query.radialY_));
      output[1] = add(scale(localProjected[0], query.radialY_),
                      scale(localProjected[1], query.radialX_));
      output[2] = localProjected[2];
    }

    KOKKOS_INLINE_FUNCTION void accumulate_spectrum(
        std::size_t flat, DeviceChild const* children, std::size_t childCount,
        DeviceQuery const* queries, std::size_t observerCount,
        FrequencyInterpolation const* frequencies, std::size_t frequencyCount,
        DeviceTable table, double timeOriginS, ComplexValue* total, ComplexValue* direct,
        ComplexValue* boundary, bool retainBranches) {
      std::size_t const observer = flat / (frequencyCount - 1);
      std::size_t const frequencyIndex = flat % (frequencyCount - 1) + 1;
      bool const active = observer < observerCount && frequencyIndex < frequencyCount;
      FrequencyInterpolation frequency{};
      ComplexValue totalValue[3]{}, directValue[3]{}, boundaryValue[3]{};
      if (active) {
        frequency = frequencies[frequencyIndex];
        for (std::size_t component = 0; component < 3; ++component) {
          std::size_t const output =
              (observer * 3 + component) * frequencyCount + frequencyIndex;
          totalValue[component] = {total[output].real_, total[output].imag_};
          if (retainBranches) {
            directValue[component] = {direct[output].real_, direct[output].imag_};
            boundaryValue[component] = {boundary[output].real_, boundary[output].imag_};
          }
        }
      }
      for (std::size_t childIndex = 0; childIndex < childCount; ++childIndex) {
        DeviceChild const sharedChild = children[childIndex];
        DeviceQuery const sharedQuery = queries[childIndex * observerCount + observer];
        if (active && sharedQuery.valid_) {
          ComplexValue directProjected[3]{}, boundaryProjected[3]{};
          project_branch(table, frequency, sharedQuery, sharedChild, true,
                         directProjected);
          project_branch(table, frequency, sharedQuery, sharedChild, false,
                         boundaryProjected);
          double timeSin = 0.0;
          double timeCos = 0.0;
          double const timePhase = -2.0 * pi * frequency.frequencyHz_ *
                                   (sharedChild.midpointTimeS_ - timeOriginS);
          sine_cosine(timePhase, &timeSin, &timeCos);
          ComplexValue const timeCarrier{timeCos, timeSin};
          ComplexValue const derivative{0.0, -2.0 * pi * frequency.frequencyHz_};
          double const normalization =
              mu0_h_per_m * elementary_charge_c * sharedChild.chargeWeight_ / (4.0 * pi);
          ComplexValue const common =
              scale(multiply(derivative, timeCarrier), normalization);
          for (std::size_t component = 0; component < 3; ++component) {
            ComplexValue const directContribution =
                multiply(common, directProjected[component]);
            ComplexValue const boundaryContribution =
                multiply(common, boundaryProjected[component]);
            directValue[component] = add(directValue[component], directContribution);
            boundaryValue[component] =
                add(boundaryValue[component], boundaryContribution);
            totalValue[component] =
                add(totalValue[component], add(directContribution, boundaryContribution));
          }
        }
      }
      if (active) {
        for (std::size_t component = 0; component < 3; ++component) {
          std::size_t const output =
              (observer * 3 + component) * frequencyCount + frequencyIndex;
          total[output] =
              make_complex(totalValue[component].real_, totalValue[component].imag_);
          if (retainBranches) {
            direct[output] =
                make_complex(directValue[component].real_, directValue[component].imag_);
            boundary[output] = make_complex(boundaryValue[component].real_,
                                            boundaryValue[component].imag_);
          }
        }
      }
    }

    std::pair<std::size_t, double> host_bracket(std::vector<double> const& axis,
                                                double value) {
      auto upper = std::upper_bound(axis.begin(), axis.end(), value);
      std::size_t hi = static_cast<std::size_t>(upper - axis.begin());
      hi = std::clamp<std::size_t>(hi, 1, axis.size() - 1);
      std::size_t const lo = hi - 1;
      return {lo, (value - axis[lo]) / (axis[hi] - axis[lo])};
    }

    double host_surface(std::vector<double> const& values,
                        std::vector<double> const& sourceZ,
                        std::vector<double> const& radiusM, double z, double radius) {
      auto const zBracket = host_bracket(sourceZ, z);
      auto const rBracket = host_bracket(radiusM, radius);
      std::size_t const zi = zBracket.first;
      std::size_t const ri = rBracket.first;
      double const wz = zBracket.second;
      double const wr = rBracket.second;
      auto value = [&](std::size_t dz, std::size_t dr) {
        return values[(zi + dz) * radiusM.size() + ri + dr];
      };
      return (1.0 - wz) * ((1.0 - wr) * value(0, 0) + wr * value(0, 1)) +
             wz * ((1.0 - wr) * value(1, 0) + wr * value(1, 1));
    }

    class FullWaveAccumulator final : public IFullWaveAccumulator {
    public:
      FullWaveAccumulator(std::filesystem::path const& manifest,
                          std::string const& fingerprint,
                          std::vector<vector_type> observers, FullWaveDeviceConfig config,
                          int numThreads, int deviceId)
          : numThreads_(numThreads)
          , deviceId_(deviceId)
          , config_(config)
          , observers_(std::move(observers)) {
        config_.validate();
        std::size_t const maximum = std::numeric_limits<int>::max();
        if (observers_.size() > maximum || config_.batchSegments_ > maximum ||
            config_.grid_.frequencyCount() > maximum ||
            (!observers_.empty() &&
             (config_.batchSegments_ > maximum / observers_.size() ||
              config_.grid_.frequencyCount() > maximum / observers_.size())))
          throw std::invalid_argument("full-wave execution range exceeds supported size");
        if (observers_.empty())
          throw std::invalid_argument("full-wave radio needs observers");
        for (auto const& observer : observers_)
          if (!finite(observer)) throw std::invalid_argument("non-finite radio observer");
        loadTable(manifest, fingerprint);
        prepareFrequencyInterpolation();
        auto const estimate = allocationEstimate();
        if (estimate.totalDeviceBytes_ > config_.memoryLimitBytes_) {
          throw std::runtime_error("Kokkos full-wave allocation exceeds memory limit");
        }
        hostChildren_.reserve(config_.batchSegments_);
        initialized_ = true;
        reset();
      }
      void append(FullWaveChildSegment const& segment) final {
        if (!finite(segment.midpointM_) || !finite(segment.displacementM_) ||
            !std::isfinite(segment.midpointTimeS_) ||
            !std::isfinite(segment.chargeWeight_))
          throw std::invalid_argument("non-finite full-wave segment");
        hostChildren_.push_back(
            {{segment.midpointM_[0], segment.midpointM_[1], segment.midpointM_[2]},
             {segment.displacementM_[0], segment.displacementM_[1],
              segment.displacementM_[2]},
             segment.midpointTimeS_,
             segment.chargeWeight_});
        ++statistics_.inputSegments_;
        if (hostChildren_.size() == config_.batchSegments_) flush();
      }
      void flush() final {
        if (hostChildren_.empty()) return;
        ensureDevice();
        std::size_t const count = hostChildren_.size();
        statistics_.maximumBatchSegments_ =
            std::max(statistics_.maximumBatchSegments_, count);
        auto start = clock_type::now();
        using host_children_type = Kokkos::View<DeviceChild const*, Kokkos::HostSpace,
                                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        Kokkos::deep_copy(
            Kokkos::subview(deviceChildren_.view_, std::make_pair(std::size_t(0), count)),
            host_children_type(hostChildren_.data(), count));
        Kokkos::deep_copy(deviceInvalid_.view_, 0U);
        statistics_.hostToDeviceMiB_ += count * sizeof(DeviceChild) / (1024. * 1024.);
        statistics_.transferMs_ += elapsed_ms(start);
        auto const children = deviceChildren_.get();
        auto const observers = deviceObservers_.get();
        auto const queries = deviceQueries_.get();
        auto const invalid = deviceInvalid_.get();
        auto const table = deviceTable();
        auto const observerCount = observers_.size();
        start = clock_type::now();
        Kokkos::parallel_for(
            "firn full-wave geometry", count * observerCount,
            KOKKOS_LAMBDA(std::size_t i) {
              build_query(i, children, count, observers, observerCount, table, queries,
                          invalid);
            });
        Kokkos::fence("firn geometry complete");
        statistics_.geometryKernelMs_ += elapsed_ms(start);
        auto invalidHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                               deviceInvalid_.view_);
        statistics_.tableDomainFailures_ += invalidHost(0);
        if (invalidHost(0))
          throw std::runtime_error("full-wave batch contains out-of-domain queries");
        auto const frequencies = deviceFrequencies_.get();
        auto const frequencyCount = config_.grid_.frequencyCount();
        auto const total = deviceTotal_.get();
        auto const direct = deviceDirect_.get();
        auto const boundary = deviceBoundary_.get();
        auto const origin = config_.timeOriginS_;
        auto const retain = config_.retainBranchSpectra_;
        start = clock_type::now();
        Kokkos::parallel_for(
            "firn full-wave spectrum", observerCount * (frequencyCount - 1),
            KOKKOS_LAMBDA(std::size_t i) {
              accumulate_spectrum(i, children, count, queries, observerCount, frequencies,
                                  frequencyCount, table, origin, total, direct, boundary,
                                  retain);
            });
        Kokkos::fence("firn spectrum complete");
        statistics_.spectrumKernelMs_ += elapsed_ms(start);
        statistics_.responseQueries_ += count * observerCount;
        ++statistics_.batches_;
        hostChildren_.clear();
      }
      FullWaveSpectra download() final {
        flush();
        auto const start = clock_type::now();
        FullWaveSpectra out;
        if (!deviceReady_) {
          // An empty event must not initialize Kokkos ahead of the EM backend.
          for (auto* family : {&out.total_, &out.direct_, &out.boundary_}) {
            if (family != &out.total_ && !config_.retainBranchSpectra_) continue;
            family->resize(observers_.size());
            for (auto& observer : *family)
              for (auto& component : observer)
                component.resize(config_.grid_.frequencyCount());
          }
          return out;
        }
        auto retrieve = [&](DeviceBuffer<ComplexValue> const& device, auto& target) {
          auto host =
              Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device.view_);
          target.resize(observers_.size());
          for (std::size_t o = 0; o < observers_.size(); ++o)
            for (std::size_t c = 0; c < 3; ++c) {
              target[o][c].resize(config_.grid_.frequencyCount());
              for (std::size_t f = 0; f < config_.grid_.frequencyCount(); ++f) {
                auto const value = host((o * 3 + c) * config_.grid_.frequencyCount() + f);
                target[o][c][f] = {value.real_, value.imag_};
              }
            }
          statistics_.deviceToHostMiB_ += device.bytes() / (1024. * 1024.);
        };
        retrieve(deviceTotal_, out.total_);
        if (config_.retainBranchSpectra_) {
          retrieve(deviceDirect_, out.direct_);
          retrieve(deviceBoundary_, out.boundary_);
        }
        statistics_.transferMs_ += elapsed_ms(start);
        return out;
      }
      void reset() final {
        hostChildren_.clear();
        auto const saved = statistics_;
        statistics_ = {};
        statistics_.tableBytes_ = saved.tableBytes_;
        statistics_.spectrumBytes_ = saved.spectrumBytes_;
        statistics_.workspaceBytes_ = saved.workspaceBytes_;
        statistics_.totalDeviceBytes_ = saved.totalDeviceBytes_;
        if (deviceReady_) clearSpectra();
      }
      FullWaveDeviceStatistics const& statistics() const final { return statistics_; }
      std::string executionSpaceName() const final {
        return Kokkos::DefaultExecutionSpace::name();
      }
      double sourceRefractiveIndex(double z) const final {
        requireInitialized();
        if (!std::isfinite(z) || z < sourceZ_.front() || z > sourceZ_.back()) {
          throw std::runtime_error("source z=" + std::to_string(z) +
                                   " m lies outside Kokkos full-wave table [" +
                                   std::to_string(sourceZ_.front()) + ", " +
                                   std::to_string(sourceZ_.back()) + "] m");
        }
        auto const [index, weight] = host_bracket(sourceZ_, z);
        return (1.0 - weight) * sourceIndex_[index] + weight * sourceIndex_[index + 1];
      }

      bool covers(vector_type const& source, vector_type const& observer) const final {
        if (!initialized_ || !finite(source) || !finite(observer)) return false;
        double const radius =
            std::hypot(observer[0] - source[0], observer[1] - source[1]);
        return source[2] >= sourceZ_.front() && source[2] <= sourceZ_.back() &&
               radius >= radiusM_.front() && radius <= radiusM_.back() &&
               std::abs(observer[2] - observerZM_) <= 1.0e-9 &&
               config_.grid_.nyquistHz() <= maximumFrequencyHz_ + 1.0e-6;
      }

      std::array<double, 2> opticalPathMeters(vector_type const& source,
                                              vector_type const& observer) const final {
        if (!covers(source, observer)) {
          throw std::runtime_error("path query lies outside Kokkos full-wave table");
        }
        double const radius =
            std::hypot(observer[0] - source[0], observer[1] - source[1]);
        return {host_surface(directPath_, sourceZ_, radiusM_, source[2], radius),
                host_surface(boundaryPath_, sourceZ_, radiusM_, source[2], radius)};
      }

    private:
      struct HostBand {
        std::vector<double> frequency_;
        std::array<std::vector<double>, 8> arrays_;
        std::size_t nodeOffset_{};
      };

      void requireInitialized() const {
        if (!initialized_) {
          throw std::runtime_error(
              "Kokkos coherent full-wave accumulator is not initialized");
        }
      }

      static void increasing(std::vector<double> const& values) {
        if (values.size() < 2 ||
            values.size() > std::numeric_limits<std::uint32_t>::max())
          throw std::runtime_error("invalid full-wave axis size");
        for (std::size_t i = 1; i < values.size(); ++i)
          if (!(values[i] > values[i - 1]))
            throw std::runtime_error("non-increasing full-wave axis");
      }
      static std::vector<double> vector(cnpy::npz_t const& archive,
                                        std::string const& key) {
        auto found = archive.find(key);
        if (found == archive.end() || found->second.word_size != sizeof(double) ||
            found->second.fortran_order) {
          throw std::runtime_error("invalid Kokkos full-wave NPZ array: " + key);
        }
        auto values = found->second.as_vec<double>();
        if (!std::all_of(values.begin(), values.end(),
                         [](double v) { return std::isfinite(v); }))
          throw std::runtime_error("non-finite full-wave array: " + key);
        return values;
      }

      void loadTable(std::filesystem::path const& manifestPath,
                     std::string const& expectedFingerprint) {
        YAML::Node const manifest = YAML::LoadFile(manifestPath.string());
        if (manifest["schema_version"].as<int>() != 2 ||
            manifest["dyadic_basis"].as<std::string>() != "cylindrical_r_phi_z" ||
            manifest["frequency_interpolation"].as<std::string>() !=
                "banded_not_a_knot_cubic_phase_reduced_fG") {
          throw std::runtime_error("unsupported Kokkos coherent full-wave manifest");
        }
        if (manifest["profile_fingerprint"].as<std::string>() != expectedFingerprint) {
          throw std::runtime_error("Kokkos full-wave profile fingerprint mismatch");
        }
        observerZM_ = manifest["observer_z_m"].as<double>();
        if (!std::isfinite(observerZM_))
          throw std::runtime_error("invalid observer plane");
        std::filesystem::path npzPath = manifest["npz"].as<std::string>();
        if (npzPath.is_relative()) npzPath = manifestPath.parent_path() / npzPath;
        if (sha256_file(npzPath) != manifest["npz_sha256"].as<std::string>()) {
          throw std::runtime_error("Kokkos full-wave response SHA-256 mismatch");
        }
        cnpy::npz_t const archive = cnpy::npz_load(npzPath.string());
        sourceZ_ = vector(archive, "source_z_m");
        radiusM_ = vector(archive, "horizontal_radius_m");
        sourceIndex_ = vector(archive, "source_refractive_index");
        increasing(sourceZ_);
        increasing(radiusM_);
        if (radiusM_.front() < 0. ||
            !std::all_of(sourceIndex_.begin(), sourceIndex_.end(),
                         [](double n) { return n > 0.; }))
          throw std::runtime_error("invalid full-wave spatial model");
        directPath_ = vector(archive, "direct_reduction_optical_path_m");
        boundaryPath_ = vector(archive, "boundary_reduction_optical_path_m");
        if (sourceZ_.size() < 2 || radiusM_.size() < 2 ||
            sourceIndex_.size() != sourceZ_.size() ||
            directPath_.size() != sourceZ_.size() * radiusM_.size() ||
            boundaryPath_.size() != directPath_.size()) {
          throw std::runtime_error("Kokkos full-wave table axis/path shape differs");
        }
        std::array<char const*, 8> const suffixes{
            "direct_reduced_fg_real_per_m_hz",   "direct_reduced_fg_imag_per_m_hz",
            "direct_second_real_per_m_per_hz",   "direct_second_imag_per_m_per_hz",
            "boundary_reduced_fg_real_per_m_hz", "boundary_reduced_fg_imag_per_m_hz",
            "boundary_second_real_per_m_per_hz", "boundary_second_imag_per_m_per_hz"};
        std::size_t totalNodes = 0;
        for (std::size_t bandIndex = 0; bandIndex < 4; ++bandIndex) {
          HostBand band;
          std::string const prefix = "band" + std::to_string(bandIndex) + "_";
          band.frequency_ = vector(archive, prefix + "frequency_hz");
          if (band.frequency_.size() < 2) {
            throw std::runtime_error("Kokkos full-wave frequency band is too short");
          }
          increasing(band.frequency_);
          band.nodeOffset_ = totalNodes;
          totalNodes += band.frequency_.size();
          std::size_t const expected =
              band.frequency_.size() * sourceZ_.size() * radiusM_.size() * 9;
          for (std::size_t field = 0; field < suffixes.size(); ++field) {
            band.arrays_[field] = vector(archive, prefix + suffixes[field]);
            if (band.arrays_[field].size() != expected) {
              throw std::runtime_error("Kokkos full-wave tensor shape differs");
            }
          }
          bands_.push_back(std::move(band));
        }
        maximumFrequencyHz_ = bands_.back().frequency_.back();
        if (totalNodes > std::numeric_limits<std::uint32_t>::max())
          throw std::runtime_error("full-wave frequency index overflow");
        totalNodeCount_ = totalNodes;
        std::size_t const spatialCount = sourceZ_.size() * radiusM_.size() * 9;
        for (auto& field : transposed_) field.assign(spatialCount * totalNodes, 0.0);
        for (auto const& band : bands_) {
          std::size_t const bandNodes = band.frequency_.size();
          for (std::size_t f = 0; f < bandNodes; ++f) {
            for (std::size_t z = 0; z < sourceZ_.size(); ++z) {
              for (std::size_t r = 0; r < radiusM_.size(); ++r) {
                for (std::size_t row = 0; row < 3; ++row) {
                  for (std::size_t col = 0; col < 3; ++col) {
                    std::size_t const input =
                        ((((f * sourceZ_.size() + z) * radiusM_.size() + r) * 3 + row) *
                             3 +
                         col);
                    std::size_t const space =
                        (((z * radiusM_.size() + r) * 3 + row) * 3 + col);
                    std::size_t const output = space * totalNodes + band.nodeOffset_ + f;
                    for (std::size_t field = 0; field < transposed_.size(); ++field)
                      transposed_[field][output] = band.arrays_[field][input];
                  }
                }
              }
            }
          }
        }
      }

      void prepareFrequencyInterpolation() {
        frequencyInterpolation_.resize(config_.grid_.frequencyCount());
        for (std::size_t output = 1; output < config_.grid_.frequencyCount(); ++output) {
          double const frequency = config_.grid_.frequencyHz(output);
          HostBand const* selected = nullptr;
          for (auto const& band : bands_) {
            if (frequency >= band.frequency_.front() - 1.0e-6 &&
                frequency <= band.frequency_.back() + 1.0e-6) {
              selected = &band;
              break;
            }
          }
          if (!selected) {
            throw std::runtime_error("Kokkos output frequency lies outside table bands");
          }
          auto const [index, weight] = host_bracket(selected->frequency_, frequency);
          double const h = selected->frequency_[index + 1] - selected->frequency_[index];
          double const a = 1.0 - weight;
          double const b = weight;
          frequencyInterpolation_[output] = {
              static_cast<std::uint32_t>(selected->nodeOffset_ + index),
              static_cast<std::uint32_t>(selected->nodeOffset_ + index + 1),
              a,
              b,
              (a * a * a - a) * h * h / 6.0,
              (b * b * b - b) * h * h / 6.0,
              frequency};
        }
      }

      template <typename TValue>
      static void copy(DeviceBuffer<TValue>& destination,
                       std::vector<TValue> const& source, char const*) {
        destination.allocate(source.size());
        using host_type = Kokkos::View<TValue const*, Kokkos::HostSpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        Kokkos::deep_copy(destination.view_, host_type(source.data(), source.size()));
      }
      FullWaveDeviceStatistics allocationEstimate() const {
        std::size_t const tableBytes =
            (sourceZ_.size() + sourceIndex_.size() + radiusM_.size() +
             directPath_.size() + boundaryPath_.size()) *
                sizeof(double) +
            [&] {
              std::size_t total = 0;
              for (auto const& field : transposed_)
                total += field.size() * sizeof(double);
              return total;
            }();
        std::size_t const spectrumBytes = outputValueCount() * sizeof(ComplexValue) *
                                          (config_.retainBranchSpectra_ ? 3 : 1);
        std::size_t const workspaceBytes =
            observers_.size() * 3 * sizeof(double) +
            frequencyInterpolation_.size() * sizeof(FrequencyInterpolation) +
            config_.batchSegments_ * sizeof(DeviceChild) +
            config_.batchSegments_ * observers_.size() * sizeof(DeviceQuery) +
            sizeof(unsigned int);
        FullWaveDeviceStatistics estimate;
        estimate.tableBytes_ = tableBytes;
        estimate.spectrumBytes_ = spectrumBytes;
        estimate.workspaceBytes_ = workspaceBytes;
        estimate.totalDeviceBytes_ = tableBytes + spectrumBytes + workspaceBytes;
        return estimate;
      }
      void ensureDevice() {
        if (deviceReady_) return;
        // EM initializes first in a hybrid cascade. Borrow that runtime, or own
        // initialization when radio is the only Kokkos process.
        if (!runtime_) runtime_.emplace(numThreads_, deviceId_);
        allocateDevice();
        clearSpectra();
        deviceReady_ = true;
        bands_.clear();
        bands_.shrink_to_fit();
        for (auto& field : transposed_) {
          field.clear();
          field.shrink_to_fit();
        }
        frequencyInterpolation_.clear();
        frequencyInterpolation_.shrink_to_fit();
      }
      void clearSpectra() {
        Kokkos::deep_copy(deviceTotal_.view_, ComplexValue{});
        if (config_.retainBranchSpectra_) {
          Kokkos::deep_copy(deviceDirect_.view_, ComplexValue{});
          Kokkos::deep_copy(deviceBoundary_.view_, ComplexValue{});
        }
      }
      void allocateDevice() {
        auto const estimate = allocationEstimate();
        statistics_.tableBytes_ = estimate.tableBytes_;
        statistics_.spectrumBytes_ = estimate.spectrumBytes_;
        statistics_.workspaceBytes_ = estimate.workspaceBytes_;
        statistics_.totalDeviceBytes_ = estimate.totalDeviceBytes_;
        std::vector<double> flatObservers;
        flatObservers.reserve(observers_.size() * 3);
        for (auto const& observer : observers_)
          flatObservers.insert(flatObservers.end(), observer.begin(), observer.end());
        copy(deviceSourceZ_, sourceZ_, "copy Kokkos source-z axis");
        copy(deviceSourceIndex_, sourceIndex_, "copy Kokkos source-index axis");
        copy(deviceRadius_, radiusM_, "copy Kokkos radius axis");
        copy(deviceDirectPath_, directPath_, "copy Kokkos direct paths");
        copy(deviceBoundaryPath_, boundaryPath_, "copy Kokkos boundary paths");
        copy(deviceDirectReal_, transposed_[0], "copy Kokkos direct real table");
        copy(deviceDirectImag_, transposed_[1], "copy Kokkos direct imag table");
        copy(deviceDirectSecondReal_, transposed_[2],
             "copy Kokkos direct real second derivatives");
        copy(deviceDirectSecondImag_, transposed_[3],
             "copy Kokkos direct imag second derivatives");
        copy(deviceBoundaryReal_, transposed_[4], "copy Kokkos boundary real table");
        copy(deviceBoundaryImag_, transposed_[5], "copy Kokkos boundary imag table");
        copy(deviceBoundarySecondReal_, transposed_[6],
             "copy Kokkos boundary real second derivatives");
        copy(deviceBoundarySecondImag_, transposed_[7],
             "copy Kokkos boundary imag second derivatives");
        copy(deviceObservers_, flatObservers, "copy Kokkos observers");
        copy(deviceFrequencies_, frequencyInterpolation_,
             "copy Kokkos frequency interpolation map");
        deviceChildren_.allocate(config_.batchSegments_);
        deviceQueries_.allocate(config_.batchSegments_ * observers_.size());
        deviceInvalid_.allocate(1);
        deviceTotal_.allocate(outputValueCount());
        if (config_.retainBranchSpectra_) {
          deviceDirect_.allocate(outputValueCount());
          deviceBoundary_.allocate(outputValueCount());
        }
        hostChildren_.reserve(config_.batchSegments_);
      }
      std::size_t outputValueCount() const {
        return observers_.size() * 3 * config_.grid_.frequencyCount();
      }

      DeviceTable deviceTable() const {
        return {deviceSourceZ_.get(),
                deviceSourceIndex_.get(),
                deviceRadius_.get(),
                deviceDirectPath_.get(),
                deviceBoundaryPath_.get(),
                deviceDirectReal_.get(),
                deviceDirectImag_.get(),
                deviceDirectSecondReal_.get(),
                deviceDirectSecondImag_.get(),
                deviceBoundaryReal_.get(),
                deviceBoundaryImag_.get(),
                deviceBoundarySecondReal_.get(),
                deviceBoundarySecondImag_.get(),
                static_cast<std::uint32_t>(sourceZ_.size()),
                static_cast<std::uint32_t>(radiusM_.size()),
                static_cast<std::uint32_t>(totalNodeCount_),
                observerZM_};
      }

      // Declared before device buffers so it is released after their destruction.
      std::optional<Runtime> runtime_;
      int numThreads_{};
      int deviceId_{-1};
      bool initialized_{};
      bool deviceReady_{};
      FullWaveDeviceConfig config_{};
      std::vector<vector_type> observers_{};
      std::vector<double> sourceZ_, radiusM_, sourceIndex_;
      std::vector<double> directPath_, boundaryPath_;
      std::vector<HostBand> bands_;
      std::array<std::vector<double>, 8> transposed_{};
      std::vector<FrequencyInterpolation> frequencyInterpolation_{};
      std::vector<DeviceChild> hostChildren_{};
      std::size_t totalNodeCount_{};
      double maximumFrequencyHz_{};
      double observerZM_{};
      FullWaveDeviceStatistics statistics_{};
      DeviceBuffer<double> deviceSourceZ_, deviceRadius_, deviceDirectPath_,
          deviceBoundaryPath_;
      DeviceBuffer<double> deviceSourceIndex_;
      DeviceBuffer<double> deviceDirectReal_, deviceDirectImag_, deviceDirectSecondReal_,
          deviceDirectSecondImag_;
      DeviceBuffer<double> deviceBoundaryReal_, deviceBoundaryImag_,
          deviceBoundarySecondReal_, deviceBoundarySecondImag_;
      DeviceBuffer<double> deviceObservers_;
      DeviceBuffer<FrequencyInterpolation> deviceFrequencies_;
      DeviceBuffer<DeviceChild> deviceChildren_;
      DeviceBuffer<DeviceQuery> deviceQueries_;
      DeviceBuffer<unsigned int> deviceInvalid_;
      DeviceBuffer<ComplexValue> deviceTotal_, deviceDirect_, deviceBoundary_;
    };
  } // namespace
  std::shared_ptr<IFullWaveAccumulator> make_full_wave_accumulator(
      std::filesystem::path const& manifest, std::string const& fingerprint,
      std::vector<vector_type> observers, FullWaveDeviceConfig config, int numThreads,
      int deviceId) {
    return std::make_shared<FullWaveAccumulator>(
        manifest, fingerprint, std::move(observers), config, numThreads, deviceId);
  }
} // namespace corsika::kokkos::radio::firn
