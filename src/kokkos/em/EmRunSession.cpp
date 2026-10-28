/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/EmRunSession.hpp>
#include <corsika/kokkos/em/proposal/NativeRequirements.hpp>
#include <corsika/kokkos/em/proposal/NativeTableExporter.hpp>
#include <corsika/kokkos/em/proposal/ProposalNativeAux.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshotKey.hpp>
#include <corsika/kokkos/environment/TransportGeometryConfig.hpp>
#include <optional>
#include <stdexcept>
#include <utility>
namespace corsika::kokkos {
  class EmRunSession::Impl {
  public:
    Impl(RuntimeConfig const& runtimeConfig, std::filesystem::path auxiliaryCache)
        : runtimeConfig_{runtimeConfig}
        , auxiliaryCache_{std::move(auxiliaryCache)} {}
    RuntimeConfig runtimeConfig_{};
    std::filesystem::path auxiliaryCache_{};
    std::optional<ProposalNativeTableSet> table_{};
    std::unique_ptr<EmBackend> backend_{};
    std::vector<double> environmentKey_{};
  };
  EmRunSession::EmRunSession(RuntimeConfig const& runtimeConfig,
                             std::filesystem::path auxiliaryCache)
      : impl_(std::make_unique<Impl>(runtimeConfig, std::move(auxiliaryCache))) {}
  EmRunSession::~EmRunSession() = default;
  EmRunSession::EmRunSession(EmRunSession&&) noexcept = default;
  EmRunSession& EmRunSession::operator=(EmRunSession&&) noexcept = default;
  SessionBeginResult EmRunSession::beginProposalNative(
      EnvironmentDataView const& environment, TransportGeometryConfig const& geometry,
      BackendConfig const& config, PhysicsRequirements const& requirements,
      std::vector<proposal::NativeInteractionCalculatorView> const& interactions,
      std::vector<proposal::NativeContinuousCalculatorView> const& continuous) {
    if (config.physicsSource_ != PhysicsSource::ProposalNative) {
      throw std::invalid_argument("Kokkos session supports proposal-native physics only");
    }
    auto const reused = isInitialized();
    auto environmentKey = make_environment_snapshot_key(environment);
    environmentKey.push_back(geometry.query_.maximumMagneticDeflectionRad_);
    for (int a = 0; a < 3; ++a) {
      environmentKey.push_back(geometry.observation_.pointM_[a]);
      environmentKey.push_back(geometry.observation_.normal_[a]);
    }
    if (reused && impl_->environmentKey_ != environmentKey)
      throw std::invalid_argument(
          "Environment changed in a reused Kokkos session; create a new session to "
          "upload it");
    if (!impl_->backend_) {
      impl_->table_.emplace(export_proposal_native_tables(interactions, continuous));
      validate_proposal_native_requirements(*impl_->table_, requirements);
      auto const auxiliary =
          load_or_create_proposal_native_aux(interactions, impl_->auxiliaryCache_);
      impl_->backend_ = std::make_unique<EmBackend>(impl_->runtimeConfig_);
      impl_->backend_->initialize(environment, geometry, *impl_->table_, auxiliary,
                                  config);
      impl_->environmentKey_ = std::move(environmentKey);
    } else {
      impl_->backend_->beginShower(make_shower_config(config));
    }
    return {impl_->backend_.get(), reused,
            impl_->backend_->capabilities().muonTransport_};
  }
  bool EmRunSession::isInitialized() const noexcept { return impl_->backend_ != nullptr; }
  EmBackend& EmRunSession::getBackend() {
    if (!impl_->backend_) throw std::logic_error("Kokkos EM session is not initialized");
    return *impl_->backend_;
  }
  ProposalNativeTableSet const* EmRunSession::getLoadedNativeTable() const noexcept {
    return impl_->table_ ? &*impl_->table_ : nullptr;
  }
} // namespace corsika::kokkos
