/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/em/execution/EmBackend.hpp>
#include <corsika/kokkos/em/execution/BackendInstance.hpp>
#include <corsika/kokkos/runtime/BackendRuntime.hpp>
#include <utility>

namespace corsika::kokkos {

  class EmBackend::Impl {
  public:
    explicit Impl(RuntimeConfig const& config)
        : backend_(make(config)) {}
    static std::unique_ptr<BackendInstance> make(RuntimeConfig const& config) {
      auto const selected = resolve_execution_backend(config.executionBackend_);
      if (config.cooperativePolicy_ != "legacy" &&
          (config.cooperativePolicy_ != "adaptive" || selected != "cuda-openmp"))
        throw std::invalid_argument(
            "Adaptive scheduling requires Kokkos cuda-openmp execution");
      (void)selected;
#if defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
      if (selected == "cuda-openmp" || selected == "openmp-cuda")
        return make_cooperative_backend_instance(config);
      if (selected == "openmp") return make_openmp_backend_instance(config);
      return make_cuda_backend_instance(config);
#elif defined(CORSIKA_KOKKOS_BACKEND_OPENMP)
      return make_openmp_backend_instance(config);
#elif defined(CORSIKA_KOKKOS_BACKEND_CUDA)
      return make_cuda_backend_instance(config);
#elif defined(CORSIKA_KOKKOS_BACKEND_HIP)
      return make_hip_backend_instance(config);
#elif defined(CORSIKA_KOKKOS_BACKEND_SYCL)
      return make_sycl_backend_instance(config);
#endif
    }
    std::unique_ptr<BackendInstance> backend_;
  };

  EmBackend::EmBackend(RuntimeConfig const& config)
      : impl_(std::make_unique<Impl>(config)) {}
  EmBackend::~EmBackend() = default;
  EmBackend::EmBackend(EmBackend&&) noexcept = default;
  EmBackend& EmBackend::operator=(EmBackend&&) noexcept = default;

  void EmBackend::initialize(EnvironmentDataView const& environment,
                             TransportGeometryConfig const& geometry,
                             ProposalNativeTableSet const& table,
                             ProposalNativeAuxData const& auxiliary,
                             BackendConfig const& config) {
    return impl_->backend_->initialize(environment, geometry, table, auxiliary, config);
  }

  InteractionBatchResult EmBackend::selectInteractionsForValidation(
      std::vector<ParticleState> const& particles) {
    return impl_->backend_->selectInteractionsForValidation(particles);
  }

  PhotonTransportBatchResult EmBackend::transportPhotonsForValidation(
      std::vector<InteractionRecord> const& interactions) {
    return impl_->backend_->transportPhotonsForValidation(interactions);
  }

  LeptonTransportBatchResult EmBackend::transportLeptonsForValidation(
      std::vector<InteractionRecord> const& interactions) {
    return impl_->backend_->transportLeptonsForValidation(interactions);
  }

  LeptonVertexSelectionBatchResult EmBackend::selectLeptonVerticesForValidation(
      std::vector<InteractionRecord> const& candidates) {
    return impl_->backend_->selectLeptonVerticesForValidation(candidates);
  }

  PhotonFinalStateBatchResult EmBackend::generatePhotonFinalStatesForValidation(
      std::vector<InteractionRecord> const& interactions,
      std::uint64_t const firstSecondaryHistoryId) {
    return impl_->backend_->generatePhotonFinalStatesForValidation(
        interactions, firstSecondaryHistoryId);
  }

  LeptonFinalStateBatchResult EmBackend::generateLeptonFinalStatesForValidation(
      std::vector<InteractionRecord> const& interactions,
      std::uint64_t const firstSecondaryHistoryId) {
    return impl_->backend_->generateLeptonFinalStatesForValidation(
        interactions, firstSecondaryHistoryId);
  }

  radio::RadioWaveforms EmBackend::projectRadioForValidation(
      std::vector<LeptonTransportRecord> const& records) {
    return impl_->backend_->projectRadioForValidation(records);
  }

  void EmBackend::enqueueRadioTracks(boost::span<LeptonTransportRecord const> records) {
    impl_->backend_->enqueueRadioTracks(records);
  }

  void EmBackend::beginShower(ShowerConfig const& shower) {
    return impl_->backend_->beginShower(shower);
  }

  bool EmBackend::canTransport(ParticleState const& particle) const {
    return impl_->backend_->canTransport(particle);
  }

  bool EmBackend::hasProposalTable() const { return impl_->backend_->hasProposalTable(); }

  std::size_t EmBackend::minimumBatchSize() const {
    return impl_->backend_->minimumBatchSize();
  }

  std::size_t EmBackend::maximumResidentPhotonBatchSize() const {
    return impl_->backend_->maximumResidentPhotonBatchSize();
  }

  std::size_t EmBackend::maximumResidentLeptonBatchSize() const {
    return impl_->backend_->maximumResidentLeptonBatchSize();
  }

  std::size_t EmBackend::maximumResidentInputBatchSize() const {
    return impl_->backend_->maximumResidentInputBatchSize();
  }

  std::size_t EmBackend::pendingPhotonCount() const noexcept {
    return impl_->backend_->pendingPhotonCount();
  }

  std::size_t EmBackend::pendingLeptonCount() const noexcept {
    return impl_->backend_->pendingLeptonCount();
  }

  bool EmBackend::independentSubshowersEnabled() const noexcept {
    return impl_->backend_->independentSubshowersEnabled();
  }
  bool EmBackend::independentSubshowersReady() const noexcept {
    return impl_->backend_->independentSubshowersReady();
  }
  bool EmBackend::batchIndependentSpecifiedFallbacks() const noexcept {
    return impl_->backend_->batchIndependentSpecifiedFallbacks();
  }
  std::size_t EmBackend::advanceIndependentSubshowers(
      std::vector<ParticleState> const& particles, SubshowerCallbacks const& callbacks) {
    return impl_->backend_->advanceIndependentSubshowers(particles, callbacks);
  }

  PhotonCascadeResult EmBackend::runPhotonWavefront(
      std::vector<ParticleState> const& particles, std::uint64_t firstSecondaryHistoryId,
      std::size_t maximumWavefronts, std::size_t minimumResidentBatchSize) {
    return impl_->backend_->runPhotonWavefront(
        particles, firstSecondaryHistoryId, maximumWavefronts, minimumResidentBatchSize);
  }

  LeptonCascadeResult EmBackend::runLeptonWavefront(
      std::vector<ParticleState> const& particles, std::uint64_t firstSecondaryHistoryId,
      std::size_t maximumWavefronts, std::uint64_t secondaryHistoryIdLimitExclusive,
      std::size_t minimumResidentBatchSize) {
    return impl_->backend_->runLeptonWavefront(
        particles, firstSecondaryHistoryId, maximumWavefronts,
        secondaryHistoryIdLimitExclusive, minimumResidentBatchSize);
  }

  BackendCapabilities EmBackend::capabilities() const {
    return impl_->backend_->capabilities();
  }

  BackendStatistics const& EmBackend::statistics() const {
    return impl_->backend_->statistics();
  }

  bool EmBackend::gpuProfileEnabled() const noexcept {
    return impl_->backend_->gpuProfileEnabled();
  }

  ProfileResult EmBackend::downloadProfile() {
    return impl_->backend_->downloadProfile();
  }

  bool EmBackend::gpuRadioEnabled() const noexcept {
    return impl_->backend_->gpuRadioEnabled();
  }

  radio::RadioWaveforms EmBackend::downloadRadioWaveforms() {
    return impl_->backend_->downloadRadioWaveforms();
  }

  std::optional<FirstInteractionRecord> EmBackend::downloadFirstInteractionSnapshot() {
    return impl_->backend_->downloadFirstInteractionSnapshot();
  }

  RuntimeInfo const& EmBackend::runtimeInfo() const noexcept {
    return impl_->backend_->runtimeInfo();
  }

} // namespace corsika::kokkos
