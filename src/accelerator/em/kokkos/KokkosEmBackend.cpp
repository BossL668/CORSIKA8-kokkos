/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/detail/KokkosBackendInstance.hpp>
#include <utility>

namespace corsika::accelerator::em {
  class KokkosEmBackend::Impl {
   public:
    explicit Impl(KokkosRuntimeConfig const& config) : backend(make(config)) {}
    static std::unique_ptr<detail::KokkosBackendInstance> make(
        KokkosRuntimeConfig const& config) {
      auto const selected = resolveKokkosExecutionBackend(config.execution_backend);
      if (config.cooperative_policy != "legacy" &&
          (config.cooperative_policy != "adaptive" || selected != "cuda-openmp"))
        throw std::invalid_argument("Adaptive scheduling requires Kokkos cuda-openmp execution");
      (void)selected;
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
    if (selected == "cuda-openmp" || selected == "openmp-cuda")
      return detail::makeCooperativeBackendInstance(config);
    if (selected == "openmp") return detail::makeOpenMPBackendInstance(config);
    return detail::makeCudaBackendInstance(config);
#elif defined(CORSIKA8_KOKKOS_BACKEND_OPENMP)
    return detail::makeOpenMPBackendInstance(config);
#elif defined(CORSIKA8_KOKKOS_BACKEND_CUDA)
    return detail::makeCudaBackendInstance(config);
#elif defined(CORSIKA8_KOKKOS_BACKEND_HIP)
    return detail::makeHipBackendInstance(config);
#elif defined(CORSIKA8_KOKKOS_BACKEND_SYCL)
    return detail::makeSyclBackendInstance(config);
#endif
    }
    std::unique_ptr<detail::KokkosBackendInstance> backend;
  };

  KokkosEmBackend::KokkosEmBackend(KokkosRuntimeConfig const& config)
      : impl_(std::make_unique<Impl>(config)) {}
  KokkosEmBackend::~KokkosEmBackend() = default;
  KokkosEmBackend::KokkosEmBackend(KokkosEmBackend&&) noexcept = default;
  KokkosEmBackend& KokkosEmBackend::operator=(KokkosEmBackend&&) noexcept = default;

  void KokkosEmBackend::initialize(
      gpu::em::EnvironmentSnapshot const& environment,
      gpu::em::tables::ProposalNativeTableSet const& table,
      gpu::em::tables::ProposalNativeAuxData const& auxiliary,
      gpu::em::GpuEmConfig const& config) {
    return impl_->backend->initialize(environment, table, auxiliary, config);
  }

  gpu::em::EmInteractionBatchResult
  KokkosEmBackend::selectInteractionsForValidation(
      std::vector<gpu::em::EmParticleState> const& particles) {
    return impl_->backend->selectInteractionsForValidation(particles);
  }

  gpu::em::PhotonTransportBatchResult
  KokkosEmBackend::transportPhotonsForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions) {
    return impl_->backend->transportPhotonsForValidation(interactions);
  }

  gpu::em::LeptonTransportBatchResult
  KokkosEmBackend::transportLeptonsForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions) {
    return impl_->backend->transportLeptonsForValidation(interactions);
  }

  gpu::em::LeptonVertexSelectionBatchResult
  KokkosEmBackend::selectLeptonVerticesForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& candidates) {
    return impl_->backend->selectLeptonVerticesForValidation(candidates);
  }

  gpu::em::EmFinalStateBatchResult
  KokkosEmBackend::generatePhotonFinalStatesForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions,
      std::uint64_t const first_secondary_history_id) {
    return impl_->backend->generatePhotonFinalStatesForValidation(interactions, first_secondary_history_id);
  }

  gpu::em::BremsFinalStateBatchResult
  KokkosEmBackend::generateLeptonFinalStatesForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions,
      std::uint64_t const first_secondary_history_id) {
    return impl_->backend->generateLeptonFinalStatesForValidation(interactions, first_secondary_history_id);
  }

  gpu::radio::GpuRadioWaveforms KokkosEmBackend::projectRadioForValidation(
      std::vector<gpu::em::LeptonTransportRecord> const& records) {
    return impl_->backend->projectRadioForValidation(records);
  }

  void KokkosEmBackend::beginShower(
      AcceleratedEmShowerConfig const& shower) {
    return impl_->backend->beginShower(shower);
  }

  bool KokkosEmBackend::canTransport(
      gpu::em::EmParticleState const& particle) const {
    return impl_->backend->canTransport(particle);
  }

  bool KokkosEmBackend::hasProposalTable() const {
    return impl_->backend->hasProposalTable();
  }

  std::size_t KokkosEmBackend::minimumBatchSize() const {
    return impl_->backend->minimumBatchSize();
  }

  std::size_t KokkosEmBackend::maximumResidentPhotonBatchSize() const {
    return impl_->backend->maximumResidentPhotonBatchSize();
  }

  std::size_t KokkosEmBackend::maximumResidentLeptonBatchSize() const {
    return impl_->backend->maximumResidentLeptonBatchSize();
  }

  std::size_t KokkosEmBackend::maximumResidentInputBatchSize() const {
    return impl_->backend->maximumResidentInputBatchSize();
  }

  std::size_t KokkosEmBackend::pendingPhotonCount() const noexcept {
    return impl_->backend->pendingPhotonCount();
  }

  std::size_t KokkosEmBackend::pendingLeptonCount() const noexcept {
    return impl_->backend->pendingLeptonCount();
  }

  bool KokkosEmBackend::independentSubshowersEnabled() const noexcept {
    return impl_->backend->independentSubshowersEnabled();
  }
  bool KokkosEmBackend::independentSubshowersReady() const noexcept {
    return impl_->backend->independentSubshowersReady();
  }
  bool KokkosEmBackend::batchIndependentSpecifiedFallbacks() const noexcept {
    return impl_->backend->batchIndependentSpecifiedFallbacks();
  }
  std::size_t KokkosEmBackend::advanceIndependentSubshowers(
      std::vector<gpu::em::EmParticleState> const& particles,
      detail::SubshowerCallbacks const& callbacks) {
    return impl_->backend->advanceIndependentSubshowers(particles, callbacks);
  }

  gpu::em::ResidentPhotonCascadeResult KokkosEmBackend::runPhotonWavefront(
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id,
      std::size_t maximum_wavefronts,
      std::size_t minimum_resident_batch_size) {
    return impl_->backend->runPhotonWavefront(particles, first_secondary_history_id, maximum_wavefronts, minimum_resident_batch_size);
  }

  gpu::em::ResidentLeptonCascadeResult KokkosEmBackend::runLeptonWavefront(
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id,
      std::size_t maximum_wavefronts,
      std::uint64_t secondary_history_id_limit_exclusive,
      std::size_t minimum_resident_batch_size) {
    return impl_->backend->runLeptonWavefront(particles, first_secondary_history_id, maximum_wavefronts, secondary_history_id_limit_exclusive, minimum_resident_batch_size);
  }

  BackendCapabilities KokkosEmBackend::capabilities() const {
    return impl_->backend->capabilities();
  }

  AcceleratedEmStatistics const& KokkosEmBackend::statistics() const {
    return impl_->backend->statistics();
  }

  bool KokkosEmBackend::gpuProfileEnabled() const noexcept {
    return impl_->backend->gpuProfileEnabled();
  }

  gpu::em::GpuProfileResult KokkosEmBackend::downloadProfile() {
    return impl_->backend->downloadProfile();
  }

  bool KokkosEmBackend::gpuRadioEnabled() const noexcept {
    return impl_->backend->gpuRadioEnabled();
  }

  gpu::radio::GpuRadioWaveforms KokkosEmBackend::downloadRadioWaveforms() {
    return impl_->backend->downloadRadioWaveforms();
  }

  std::optional<gpu::em::GpuFirstInteractionSnapshot>
  KokkosEmBackend::downloadFirstInteractionSnapshot() {
    return impl_->backend->downloadFirstInteractionSnapshot();
  }

  KokkosRuntimeInfo const& KokkosEmBackend::runtimeInfo() const noexcept {
    return impl_->backend->runtimeInfo();
  }

} // namespace corsika::accelerator::em
