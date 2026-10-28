/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <PROPOSAL/medium/Medium.h>
#include <corsika/kokkos/em/EmRunSession.hpp>
#include <corsika/kokkos/em/proposal/NativeTableExporter.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshotData.hpp>
#include <corsika/kokkos/environment/TransportGeometryConfig.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tests/common/SetupTestEnvironment.hpp>
#include <utility>
#include <vector>

namespace session_test {
  namespace cp = corsika::kokkos;
  using namespace corsika;
  using environment_interface_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  using environment_type = Environment<environment_interface_type>;

  inline unsigned checks{};
  inline void require(bool value, std::string const& label) {
    ++checks;
    if (!value) throw std::runtime_error(label);
  }
  template <typename TException, typename TFunction>
  std::string rejects(TFunction&& function) {
    ++checks;
    try {
      function();
    } catch (TException const& error) { return error.what(); }
    throw std::runtime_error("Expected production exception was not raised");
  }

  // These are the same reduced real-PROPOSAL grids as NativeTableFixture.cpp.
  // Each CTest case is a fresh process; no table/model implementation is mocked.
  class SmallTableSettings {
  public:
    explicit SmallTableSettings(std::filesystem::path const& directory);
    ~SmallTableSettings();

  private:
    std::string path_;
    double upper_;
    unsigned dndxEnergy_, dndxLoss_, dedx_, utility_, rate_;
  };

  struct Calculator {
    std::unique_ptr<PROPOSAL::Interaction> interaction_;
    std::unique_ptr<PROPOSAL::Displacement> displacement_;
    std::unique_ptr<PROPOSAL::crosssection::PhotoPairLPM> photon_;
    std::unique_ptr<PROPOSAL::crosssection::BremsLPM> brems_;
    Calculator(Code code, PROPOSAL::Medium& medium);
    ~Calculator();
  };

  struct EnvironmentCopy {
    std::vector<cp::EnvironmentNode> nodes_;
    std::vector<cp::EnvironmentShape> shapes_;
    std::vector<std::int32_t> children_, exclusions_;
    std::vector<cp::EnvironmentModel> models_;
    std::vector<cp::EnvironmentComponent> composition_;
    explicit EnvironmentCopy(cp::EnvironmentDataView const& view)
        : nodes_(view.nodes_, view.nodes_ + view.nodeCount_)
        , shapes_(view.shapes_, view.shapes_ + view.nodeCount_)
        , models_(view.models_, view.models_ + view.modelCount_)
        , composition_(view.composition_, view.composition_ + view.componentCount_) {
      if (view.childCount_)
        children_.assign(view.children_, view.children_ + view.childCount_);
      if (view.exclusionCount_)
        exclusions_.assign(view.exclusions_, view.exclusions_ + view.exclusionCount_);
    }
    cp::EnvironmentDataView view() const {
      return {nodes_.data(),
              shapes_.data(),
              children_.data(),
              exclusions_.data(),
              models_.data(),
              composition_.data(),
              static_cast<std::uint32_t>(nodes_.size()),
              static_cast<std::uint32_t>(children_.size()),
              static_cast<std::uint32_t>(exclusions_.size()),
              static_cast<std::uint32_t>(models_.size()),
              static_cast<std::uint32_t>(composition_.size())};
    }
  };

  struct Fixture {
    SmallTableSettings settings_;
    PROPOSAL::Air medium_;
    std::vector<std::unique_ptr<Calculator>> calculators_;
    std::vector<proposal::NativeInteractionCalculatorView> interactions_;
    std::vector<proposal::NativeContinuousCalculatorView> continuous_;
    std::unique_ptr<environment_type> environment_;
    CoordinateSystemPtr coordinates_;
    environment_type::BaseNodeType* node_{};
    std::unique_ptr<cp::EnvironmentSnapshot<environment_interface_type>> snapshot_;
    cp::TransportGeometryConfig geometry_{};
    cp::BackendConfig config_{};
    cp::PhysicsRequirements requirements_{1.e6, 100., 100., .4};
    explicit Fixture(std::filesystem::path const& directory);
    ~Fixture();
    cp::EnvironmentDataView view() const { return cp::bind_environment_data(*snapshot_); }
    cp::SessionBeginResult begin(cp::EmRunSession& session) const {
      return session.beginProposalNative(view(), geometry_, config_, requirements_,
                                         interactions_, continuous_);
    }
    std::vector<cp::ParticleState> particles() const {
      std::vector<cp::ParticleState> result;
      for (int pid : {22, 11, -11, 13, -13}) {
        cp::ParticleState particle{};
        particle.pid_ = pid;
        particle.energyGeV_ = 100.;
        particle.environmentNode_ = snapshot_->getTopology().getNodeIndex(*(node_));
        particle.mediumId_ = static_cast<int>(Medium::AirDry1Atm);
        particle.direction_[2] = -1.;
        particle.weight_ = 1.;
        particle.historyId_ = 100 + result.size();
        particle.parentHistoryId_ = 1;
        particle.generation_ = 1;
        particle.stepId_ = 7;
        result.push_back(particle);
      }
      return result;
    }
  };
} // namespace session_test
