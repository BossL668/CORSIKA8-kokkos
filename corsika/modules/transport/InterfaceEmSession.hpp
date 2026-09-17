/* Independent two-sided material transport. No application, YAML or air-backend
 * dependency. Each bank supplies its own density profile and uniform B vector.
 * Runtime ownership is exclusive; all views die before runtime finalization. */
#pragma once
#include <corsika/modules/transport/InterfaceEmTypes.hpp>
#include <functional>
#include <array>

namespace corsika::interfaces {
struct ResidentStatistics {
  std::uint64_t uploaded_particles{}, upload_batches{}, wavefronts{};
  std::uint64_t advanced_particles{}, device_enqueued_particles{};
  std::size_t peak_pending_particles{};
  std::uint64_t cascade_calls{}, control_downloads{}, record_downloads{}, downloaded_records{};
  std::size_t maximum_call_wavefronts{}, peak_buffered_records{};
  // Drained, CPU fallback, scalar interleave, record capacity, wavefront limit.
  std::array<std::uint64_t,5> checkpoint_counts{};
  std::array<std::uint64_t,17> wavefront_size_log2{};
};

enum class ResidentCheckpoint { Drained, CpuFallback, ScalarInterleave, RecordCapacity, WavefrontLimit };
struct ResidentCascadeResult {
  std::vector<EmStep> records;
  std::size_t wavefronts{};
  ResidentCheckpoint checkpoint{ResidentCheckpoint::Drained};
};

class InterfaceEmSession {
 public:
  InterfaceEmSession(terrain::FlatTerrainData const&, std::vector<EmMaterial> const&, EmConfig const&);
  ~InterfaceEmSession();
  InterfaceEmSession(InterfaceEmSession const&) = delete;
  InterfaceEmSession& operator=(InterfaceEmSession const&) = delete;

  // Stateless reference: never mutates the resident queue. Useful for step replay.
  std::vector<EmStep> advance(std::vector<em::EmParticleState> const&,
                              std::uint64_t first_child_history);
  // Only CPU-originated particles enter here. The entire input is validated
  // before upload; rejection leaves the logical queue unchanged.
  void submit(std::vector<em::EmParticleState> const&);
  // Advance a stable FIFO prefix. Survivors/children are compacted and appended
  // device-to-device. Returned records serve diagnostics and specified CPU
  // fallback; they must NOT be re-enqueued by the caller.
  // Reserve exactly 3 * nextResidentBatchSize() histories before calling.
  std::vector<EmStep> advanceResident(std::uint64_t first_child_history);
  // Air-style backend-owned multi-wavefront loop. Only a small control POD is
  // read each wavefront. Output records are buffered on device until the queue
  // drains, CPU fallback/interleave is needed, or a bounded checkpoint is hit.
  // reserve_histories receives the exact number of child slots for that front;
  // it must only reserve IDs and must not submit particles/re-enter the session.
  ResidentCascadeResult runResidentCascade(
      std::function<std::uint64_t(std::size_t)> const& reserve_histories,
      std::size_t minimum_pending = 1, std::size_t maximum_wavefronts = 1024);
  std::size_t nextResidentBatchSize() const;
  std::size_t pendingParticles() const;
  std::size_t residentCapacity() const;
  ResidentStatistics const& residentStatistics() const;
  std::size_t deviceBytes() const;
  std::size_t projectedPeakDeviceBytes() const;
  std::string executionSpace() const;
  int executionConcurrency() const;
  bool radioEnabled() const;
  void accumulateRadioCpu(std::vector<corsika::radio::interface::Track> const&);
  // Enabling interface radio always accumulates both algorithms on each source.
  corsika::radio::interface::PairedResult finishRadio();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  void launch(std::size_t count, std::uint64_t first_child_history);
  std::vector<EmStep> execute(std::size_t count, std::uint64_t first_child_history);
};
} // namespace corsika::interfaces
