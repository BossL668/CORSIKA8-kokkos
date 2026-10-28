/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "EnvironmentQueryDataset.hpp"

#include <corsika/kokkos/environment/EnvironmentSnapshot.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>

#include <array>
#include <stdexcept>

namespace environment_query_test {

  namespace {
    using namespace corsika;
    using model_interface_type =
        IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
    using environment_type = Environment<model_interface_type>;
    template <typename TModel>
    using extra_type =
        GladstoneDaleRefractiveIndex<MediumPropertyModel<UniformMagneticField<TModel>>>;
  } // namespace

  Dataset atmosphere() {
    using namespace corsika;
    auto const common = get_root_CoordinateSystem();
    auto const local = make_translation(common, {1_km, -2_km, 3_km});
    Point const center(local, 0_m, 0_m, 0_m);
    Point const reference(local, 0_m, 0_m, constants::EarthRadius::Mean);
    environment_type environment;
    create_5layer_atmosphere<model_interface_type, extra_type>(
        environment, AtmosphereId::LinsleyUSStd, center, 1.000312, reference,
        Medium::AirDry1Atm, MagneticFieldVector(local, {20_uT, -30_uT, 40_uT}));
    kokkos::EnvironmentSnapshot<model_interface_type> const snapshot(
        *environment.getUniverse(), common);
    Dataset result;
    result.name_ = "native-five-layer-atmosphere";
    result.nodes_ = snapshot.getTopology().getNodes();
    result.shapes_ = snapshot.getShapes();
    result.children_ = snapshot.getTopology().getChildren();
    result.exclusions_ = snapshot.getTopology().getExclusions();
    result.models_ = snapshot.getModels();
    result.components_ = snapshot.getComposition();

    // Independently enumerate actual native ownership, rather than using the
    // snapshot's pointer-to-index map to obtain the expected node identities.
    std::vector<environment_type::BaseNodeType const*> nativeNodes;
    auto visit = [&](auto&& self, environment_type::BaseNodeType const& node) -> void {
      nativeNodes.push_back(&node);
      for (auto const& child : node.getChildNodes()) { self(self, *child); }
    };
    visit(visit, *environment.getUniverse());
    if (nativeNodes.size() != 6 || result.models_.size() != 5) {
      throw std::logic_error(
          "native atmosphere fixture is not the complete five-layer tree");
    }
    std::array<double, 5> const heightsKm{2., 7., 25., 70., 105.};
    std::array<std::array<double, 3>, 3> const directions{
        {{0., 0., -1.}, {1. / 3., 2. / 3., -2. / 3.}, {-2. / 3., 1. / 3., -2. / 3.}}};
    for (std::size_t layer = 0; layer < heightsKm.size(); ++layer) {
      Point const point(local, 0_m, 0_m,
                        constants::EarthRadius::Mean + heightsKm[layer] * 1_km);
      auto const* node = environment.getUniverse()->getContainingNode(point);
      auto const expectedNode = static_cast<std::int32_t>(5 - layer);
      if (node != nativeNodes[expectedNode]) {
        throw std::logic_error("native atmosphere layer ordering changed");
      }
      auto const& model = node->getModelProperties();
      auto field = model.getMagneticField(point);
      auto const fieldComponents = field.getComponents(common);
      auto const position = point.getCoordinates(common);
      auto const nativeDensity = model.getMassDensity(point) / (1_g / cube(1_cm));
      for (std::size_t variant = 0; variant < directions.size(); ++variant) {
        IntegrationQuery query;
        query.nativeNode_ = expectedNode;
        query.nativeDensityGPerCm3_ = nativeDensity;
        query.chargeNumber_ = variant == 0 ? 1. : (variant == 1 ? -1. : 0.);
        query.particle_.pid_ = variant == 0 ? -13 : 13;
        // The third direction is an artificial zero-charge limit with the
        // same test mass, not a physical neutral-muon process.
        query.particle_.mediumId_ = -99;
        query.particle_.energyGeV_ = 10.;
        query.particle_.weight_ = 2.;
        query.particle_.timeS_ = 1.e-6;
        query.particle_.environmentNode_ = -3;
        query.particle_.historyId_ = 100 + 3 * layer + variant;
        query.particle_.parentHistoryId_ = 17;
        query.particle_.stepId_ = 4;
        query.particle_.generation_ = 3;
        query.particle_.reserved_ = 29;
        for (int axis = 0; axis < 3; ++axis) {
          query.particle_.positionM_[axis] = position[axis] / 1_m;
          query.particle_.direction_[axis] = directions[variant][axis];
          query.nativeFieldT_[axis] = fieldComponents[axis] / 1_T;
        }
        result.integration_.push_back(query);
        result.queries_.push_back(integration_query(query));
      }
    }
    return result;
  }

} // namespace environment_query_test
