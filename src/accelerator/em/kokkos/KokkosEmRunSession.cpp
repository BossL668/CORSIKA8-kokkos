/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/detail/KokkosEmRunSession.hpp>

#include <optional>
#include <stdexcept>
#include <utility>

#include <corsika/accelerator/em/ProposalNativeRequirements.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeAux.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTableExporter.hpp>
#include <corsika/accelerator/em/common/EnvironmentSnapshotStorage.hpp>

namespace corsika::accelerator::em::detail {

  class KokkosEmRunSession::Impl {
  public:
    Impl(KokkosRuntimeConfig const& runtime_config,
         std::filesystem::path auxiliary_cache)
        : runtime_config_{runtime_config},
          auxiliary_cache_{std::move(auxiliary_cache)} {}

    KokkosRuntimeConfig runtime_config_{};
    std::filesystem::path auxiliary_cache_{};
    std::optional<gpu::em::tables::ProposalNativeTableSet> table_{};
    std::unique_ptr<KokkosEmBackend> backend_{};
    gpu::em::EnvironmentGeometry geometry_{};
    std::vector<double> environment_key_{};
  };

  KokkosEmRunSession::KokkosEmRunSession(
      KokkosRuntimeConfig const& runtime_config,
      std::filesystem::path auxiliary_cache)
      : impl_(std::make_unique<Impl>(
            runtime_config, std::move(auxiliary_cache))) {}
  KokkosEmRunSession::~KokkosEmRunSession() = default;
  KokkosEmRunSession::KokkosEmRunSession(KokkosEmRunSession&&) noexcept =
      default;
  KokkosEmRunSession& KokkosEmRunSession::operator=(
      KokkosEmRunSession&&) noexcept = default;

  KokkosSessionBeginResult KokkosEmRunSession::beginProposalNative(
      gpu::em::EnvironmentSnapshot const& environment,
      gpu::em::GpuEmConfig const& config,
      AcceleratedPhysicsRequirements const& requirements,
      std::vector<proposal::NativeInteractionCalculatorView> const& interactions,
      std::vector<proposal::NativeContinuousCalculatorView> const& continuous) {
    if (config.physics_source != gpu::em::GpuPhysicsSource::ProposalNative) {
      throw std::invalid_argument(
          "Kokkos session supports proposal-native physics only");
    }
    auto const reused = initialized();
    auto environment_key = gpu::em::environment_tree::snapshotKey(environment.node_tree);
    if (environment.geometry == gpu::em::EnvironmentGeometry::NodeTree) {
      environment_key.push_back(environment.maximum_magnetic_deflection_rad);
      for (int a = 0; a < 3; ++a) {
        environment_key.push_back(environment.observation_plane_point_m[a]);
        environment_key.push_back(environment.observation_plane_normal[a]);
      }
    }
    if (reused && (impl_->geometry_ != environment.geometry ||
        (environment.geometry == gpu::em::EnvironmentGeometry::NodeTree &&
         impl_->environment_key_ != environment_key)))
      throw std::invalid_argument("Environment changed in a reused Kokkos session; create a new session to upload it");
    if (!impl_->backend_) {
      impl_->table_.emplace(
          gpu::em::tables::exportProposalNativeTables(
              interactions, continuous));
      validateProposalNativeRequirements(*impl_->table_, requirements);
      auto const auxiliary =
          gpu::em::tables::loadOrCreateProposalNativeAux(
              interactions, impl_->auxiliary_cache_);
      impl_->backend_ =
          std::make_unique<KokkosEmBackend>(impl_->runtime_config_);
      impl_->backend_->initialize(
          environment, *impl_->table_, auxiliary, config);
      impl_->geometry_ = environment.geometry;
      impl_->environment_key_ = std::move(environment_key);
    } else {
      impl_->backend_->beginShower(gpu::em::makeGpuEmShowerConfig(config));
    }
    return {impl_->backend_.get(), reused,
            impl_->backend_->capabilities().muon_transport};
  }



  bool KokkosEmRunSession::initialized() const noexcept {
    return impl_->backend_ != nullptr;
  }

  KokkosEmBackend& KokkosEmRunSession::backend() {
    if (!impl_->backend_) {
      throw std::logic_error("Kokkos EM session is not initialized");
    }
    return *impl_->backend_;
  }

  gpu::em::tables::ProposalNativeTableSet const*
  KokkosEmRunSession::loadedNativeTable() const noexcept {
    return impl_->table_ ? &*impl_->table_ : nullptr;
  }

} // namespace corsika::accelerator::em::detail
