/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "MaterialBoundaryFixture.hpp"
#include <corsika/framework/geometry/SeparationPlane.hpp>
#include <corsika/framework/geometry/StraightTrajectory.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshotData.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/ExponentialFirn.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>

namespace material_boundary_test {
  corsika::kokkos::EnvironmentDataView Fixture::view() const {
    return {nodes_.data(),
            shapes_.data(),
            children_.data(),
            exclusions_.data(),
            models_.data(),
            composition_.data(),
            static_cast<unsigned>(nodes_.size()),
            static_cast<unsigned>(children_.size()),
            static_cast<unsigned>(exclusions_.size()),
            static_cast<unsigned>(models_.size()),
            static_cast<unsigned>(composition_.size())};
  }
  Fixture make_fixture() {
    using namespace corsika;
    namespace cp = corsika::kokkos;
    using interface_type =
        IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
    using medium_type =
        UniformMagneticField<MediumPropertyModel<ExponentialFirn<interface_type>>>;
    using environment_type = Environment<interface_type>;
    auto const frame = get_root_CoordinateSystem();
    environment_type environment;
    auto child = environment_type::createNode<SeparationPlane>(
        Plane(Point(frame, 0_m, 0_m, 0_m), DirectionVector(frame, {0., 0., 1.})));
    child->setModelProperties<medium_type>(
        MagneticFieldVector(frame, {0_T, 0_T, 0_T}), Medium::WaterIce, frame,
        NuclearComposition({Code::Hydrogen, Code::Oxygen}, {2. / 3., 1. / 3.}));
    environment.getUniverse()->addChild(std::move(child));
    cp::EnvironmentSnapshot<interface_type> const snapshot(*environment.getUniverse(),
                                                           frame);
    auto const data = cp::bind_environment_data(snapshot);
    Fixture result;
    result.nodes_.assign(data.nodes_, data.nodes_ + data.nodeCount_);
    result.shapes_.assign(data.shapes_, data.shapes_ + data.nodeCount_);
    result.children_.assign(data.children_, data.children_ + data.childCount_);
    if (data.exclusionCount_)
      result.exclusions_.assign(data.exclusions_,
                                data.exclusions_ + data.exclusionCount_);
    result.models_.assign(data.models_, data.models_ + data.modelCount_);
    result.composition_.assign(data.composition_,
                               data.composition_ + data.componentCount_);
    for (int pid : {11, -11, 13, -13}) {
      for (double depth : {.001, .01, .05}) {
        cp::InteractionRecord input;
        input.inputIndex_ = result.inputs_.size();
        input.status_ = cp::InteractionStatus::NoDiscreteInteraction;
        input.decayDistanceM_ = HUGE_VAL;
        auto& particle = input.particle_;
        particle.pid_ = pid;
        particle.environmentNode_ = 1;
        particle.mediumId_ = static_cast<int>(Medium::WaterIce);
        particle.energyGeV_ = .5;
        particle.weight_ = 1.5;
        particle.direction_[2] = 1.;
        particle.positionM_[2] = -depth;
        particle.historyId_ = 100 + input.inputIndex_;
        particle.stepId_ = 7;
        Point const start(frame, 0_m, 0_m, -depth * 1_m);
        DirectionVector const direction(frame, {0., 0., 1.});
        auto const* node = environment.getUniverse()->getContainingNode(start);
        StraightTrajectory const line(Line(start, direction * (1_m / 1_s)), depth * 1_s);
        result.inputs_.push_back(input);
        result.distances_.push_back(depth);
        result.grammages_.push_back(
            node->getModelProperties().getIntegratedGrammage(line) /
            (1_g / square(1_cm)));
      }
    }
    return result;
  }
} // namespace material_boundary_test
