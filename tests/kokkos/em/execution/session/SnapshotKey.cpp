/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentSnapshotKey.hpp>
#include "ProductionSnapshotKey.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  namespace current = corsika::kokkos;
  namespace old = corsika::gpu::em::environment_tree;
  unsigned checks{};
  void check(bool value, char const* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
  }

  struct Snapshot {
    std::vector<current::EnvironmentNode> nodes;
    std::vector<current::EnvironmentShape> shapes;
    std::vector<std::int32_t> children, exclusions;
    std::vector<current::EnvironmentModel> models;
    std::vector<current::EnvironmentComponent> composition;
    current::EnvironmentDataView view() const {
      return {nodes.data(),
              shapes.data(),
              children.data(),
              exclusions.data(),
              models.data(),
              composition.data(),
              static_cast<std::uint32_t>(nodes.size()),
              static_cast<std::uint32_t>(children.size()),
              static_cast<std::uint32_t>(exclusions.size()),
              static_cast<std::uint32_t>(models.size()),
              static_cast<std::uint32_t>(composition.size())};
    }
  };

  struct LegacySnapshot {
    std::vector<old::Node> nodes;
    std::vector<std::int32_t> children, exclusions;
    std::vector<old::Model> models;
    std::vector<old::Component> composition;
    std::vector<std::uintptr_t> nativeNodes;
    old::FullView view() const {
      return {{nodes.data(), children.data(), exclusions.data(),
               static_cast<std::uint32_t>(nodes.size())},
              models.data(),
              composition.data(),
              static_cast<unsigned>(models.size()),
              static_cast<unsigned>(composition.size()),
              static_cast<unsigned>(children.size()),
              static_cast<unsigned>(exclusions.size()),
              nativeNodes.data()};
    }
  };

  // Field-format conversion only; the reference key and FNV functions below
  // are byte-preserved production bodies, not rewritten implementations.
  LegacySnapshot legacy(Snapshot const& input) {
    LegacySnapshot out;
    out.children = input.children;
    out.exclusions = input.exclusions;
    for (std::size_t i = 0; i < input.nodes.size(); ++i) {
      auto const& n = input.nodes[i];
      auto const& s = input.shapes[i];
      old::Node node;
      node.parent = n.parentIndex_;
      node.child_begin = n.childOffset_;
      node.child_count = n.childCount_;
      node.excluded_begin = n.excludedOffset_;
      node.excluded_count = n.excludedCount_;
      node.model_id = n.modelIndex_;
      node.radius = s.radiusM_;
      switch (s.kind_) {
        case current::EnvironmentShape::Kind::Universe:
          node.shape = old::Shape::Universe;
          break;
        case current::EnvironmentShape::Kind::Sphere:
          node.shape = old::Shape::Sphere;
          break;
        case current::EnvironmentShape::Kind::Box:
          node.shape = old::Shape::Box;
          break;
        case current::EnvironmentShape::Kind::SeparationPlane:
          node.shape = old::Shape::SeparationPlane;
          break;
        default:
          throw std::runtime_error("Fixture is outside the legacy shape domain");
      }
      for (int a = 0; a < 3; ++a) {
        node.center[a] = s.centerM_[a];
        node.half_extent[a] = s.halfExtentM_[a];
        for (int b = 0; b < 3; ++b) node.axes[a][b] = s.axes_[a][b];
        if (node.shape == old::Shape::SeparationPlane) node.axes[0][a] = s.normal_[a];
      }
      out.nodes.push_back(node);
      out.nativeNodes.push_back(0x1000 + 0x100 * i);
    }
    for (auto const& m : input.models) {
      old::Model model;
      switch (m.densityKind_) {
        case current::EnvironmentModel::Density::Homogeneous:
          model.density_kind = old::Density::Homogeneous;
          break;
        case current::EnvironmentModel::Density::SlidingPlanarExponential:
          model.density_kind = old::Density::SlidingPlanarExponential;
          break;
        case current::EnvironmentModel::Density::FlatExponential:
          model.density_kind = old::Density::FlatExponential;
          break;
        default:
          throw std::runtime_error("Fixture is outside the legacy density domain");
      }
      switch (m.refractivityKind_) {
        case current::EnvironmentModel::Refractivity::Unavailable:
          model.refractivity_kind = old::Refractivity::Unavailable;
          break;
        case current::EnvironmentModel::Refractivity::Uniform:
          model.refractivity_kind = old::Refractivity::Uniform;
          break;
        case current::EnvironmentModel::Refractivity::GladstoneDale:
          model.refractivity_kind = old::Refractivity::GladstoneDale;
          break;
        default:
          throw std::runtime_error("Fixture is outside the legacy refractivity domain");
      }
      model.rho0_g_cm3 = m.referenceDensityGPerCm3_;
      model.lambda_m = m.scaleLengthM_;
      model.inverse_lambda_m = m.inverseScaleLengthPerM_;
      model.reference_height_m = m.referenceHeightM_;
      model.refractivity_value = m.refractivityValue_;
      model.inverse_reference_density_cm3_g = m.inverseReferenceDensityCm3PerG_;
      model.medium_id = m.mediumId_;
      model.composition_begin = m.compositionBegin_;
      model.composition_count = m.compositionCount_;
      for (int a = 0; a < 3; ++a) {
        model.anchor_m[a] = m.anchorM_[a];
        model.axis[a] = m.axis_[a];
        model.magnetic_field_T[a] = m.magneticFieldT_[a];
      }
      out.models.push_back(model);
    }
    for (auto const& c : input.composition)
      out.composition.push_back({c.code_, c.fraction_});
    return out;
  }

  Snapshot fixture(std::string const& medium) {
    Snapshot s;
    s.nodes.resize(4);
    s.shapes.resize(4);
    s.children = {1, 2, 3};
    s.exclusions = {3, 2};
    s.nodes[0].childCount_ = 3;
    s.nodes[0].modelIndex_ = -1;
    s.nodes[0].parentIndex_ = -1;
    for (int i = 1; i < 4; ++i) {
      s.nodes[i].parentIndex_ = 0;
      s.nodes[i].modelIndex_ = 0;
      for (int a = 0; a < 3; ++a) s.shapes[i].centerM_[a] = 100. * i + a;
    }
    s.nodes[2].excludedCount_ = 1;
    s.nodes[3].excludedOffset_ = 1;
    s.nodes[3].excludedCount_ = 1;
    s.shapes[0].kind_ = current::EnvironmentShape::Kind::Universe;
    s.shapes[1].kind_ = current::EnvironmentShape::Kind::Sphere;
    s.shapes[1].radiusM_ = 6371000.;
    s.shapes[2].kind_ = current::EnvironmentShape::Kind::Box;
    for (int a = 0; a < 3; ++a) {
      s.shapes[2].halfExtentM_[a] = a + 1.;
      s.shapes[2].axes_[a][a] = 1.;
    }
    s.shapes[3].kind_ = current::EnvironmentShape::Kind::SeparationPlane;
    s.shapes[3].normal_[0] = .6;
    s.shapes[3].normal_[1] = -.8;
    current::EnvironmentModel m;
    m.densityKind_ = current::EnvironmentModel::Density::Homogeneous;
    m.refractivityKind_ = current::EnvironmentModel::Refractivity::Uniform;
    m.referenceDensityGPerCm3_ =
        medium == "air" ? .001225 : medium == "ice" ? .917 : 2.65;
    m.refractivityValue_ = medium == "air" ? 1.000312 : medium == "ice" ? 1.78 : 1.5;
    m.mediumId_ = medium == "air" ? 1 : medium == "ice" ? 2 : 3;
    m.compositionCount_ = 3;
    m.axis_[2] = 1.;
    m.anchorM_[0] = 17.;
    m.magneticFieldT_[1] = 2.e-5;
    s.models.push_back(m);
    // Deliberately repeated/ordered entries exercise field serialization only.
    s.composition = {{7, .7}, {8, .2}, {8, .1}};
    return s;
  }

  void equivalent(Snapshot const& s, std::string const& label) {
    auto original = legacy(s);
    auto const expected = old::snapshotKey(original.view());
    auto const actual = current::make_environment_snapshot_key(s.view());
    check(actual == expected, "Legacy key sequence changed");
    auto const fingerprint = current::environment_snapshot_fingerprint(s.view());
    check(fingerprint == old::snapshotFingerprint(original.view()),
          "Legacy fingerprint changed");
    auto relocated = s;
    check(relocated.nodes.data() != s.nodes.data(),
          "Relocation did not allocate new storage");
    check(current::make_environment_snapshot_key(relocated.view()) == actual,
          "Host pointers leaked into key");
    for (auto& address : original.nativeNodes) address += 0x400000;
    check(old::snapshotKey(original.view()) == actual,
          "Native identity pointers leaked into oracle key");
    std::cout << "key " << label << " size=" << actual.size()
              << " fingerprint=" << fingerprint << '\n';
  }
} // namespace

int main() {
  try {
    for (auto const* medium : {"air", "ice", "rock"}) {
      auto s = fixture(medium);
      equivalent(s, medium);
      s.models[0].densityKind_ =
          current::EnvironmentModel::Density::SlidingPlanarExponential;
      s.models[0].scaleLengthM_ = -8000.;
      s.models[0].inverseScaleLengthPerM_ = -1. / 8000.;
      s.models[0].referenceHeightM_ = 1200.;
      s.models[0].refractivityKind_ =
          current::EnvironmentModel::Refractivity::GladstoneDale;
      s.models[0].inverseReferenceDensityCm3PerG_ =
          1. / s.models[0].referenceDensityGPerCm3_;
      equivalent(s, std::string(medium) + "-radial");
      s.models[0].densityKind_ = current::EnvironmentModel::Density::FlatExponential;
      equivalent(s, std::string(medium) + "-flat");
    }
    auto s = fixture("air");
    auto const initial = current::make_environment_snapshot_key(s.view());
    auto changed = s;
    changed.shapes[3].normal_[1] += .1;
    check(current::make_environment_snapshot_key(changed.view()) != initial,
          "Plane normal omitted");
    changed = s;
    changed.shapes[3].axes_[0][1] = 99.;
    check(current::make_environment_snapshot_key(changed.view()) == initial,
          "New unused plane axes replaced normal");
    changed = s;
    std::swap(changed.children[0], changed.children[1]);
    check(current::make_environment_snapshot_key(changed.view()) != initial,
          "Child order omitted");
    changed = s;
    std::swap(changed.exclusions[0], changed.exclusions[1]);
    check(current::make_environment_snapshot_key(changed.view()) != initial,
          "Exclusion order omitted");
    changed = s;
    changed.models[0].magneticFieldT_[0] = -0.;
    check(current::environment_snapshot_fingerprint(changed.view()) ==
              current::environment_snapshot_fingerprint(s.view()),
          "Negative-zero canonicalization changed");
    changed = s;
    changed.models[0].densityDeficitGPerCm3_ = .4;
    changed.models[0].refractiveIndexDeficit_ = .5;
    check(current::make_environment_snapshot_key(changed.view()) == initial,
          "Legacy unused firn fields changed key");
    for (int which = 0; which < 3; ++which) {
      auto firn = changed;
      if (which != 1)
        firn.models[0].densityKind_ = current::EnvironmentModel::Density::ExponentialFirn;
      if (which != 0)
        firn.models[0].refractivityKind_ =
            current::EnvironmentModel::Refractivity::ExponentialFirn;
      auto const key = current::make_environment_snapshot_key(firn.view());
      check(key.size() == initial.size() + 2, "Firn extension size changed");
      check(key[key.size() - 2] == .4 && key.back() == .5, "Firn deficit order changed");
      firn.models[0].densityDeficitGPerCm3_ += .01;
      check(current::make_environment_snapshot_key(firn.view()) != key,
            "Density deficit omitted");
      firn = changed;
      firn.models[0].refractivityKind_ =
          current::EnvironmentModel::Refractivity::ExponentialFirn;
      auto const before = current::make_environment_snapshot_key(firn.view());
      firn.models[0].refractiveIndexDeficit_ += .01;
      check(current::make_environment_snapshot_key(firn.view()) != before,
            "Index deficit omitted");
    }
    std::cout << "passed snapshot-key checks=" << checks << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
