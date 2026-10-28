/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/modules/proposal/ContinuousProcess.hpp>
#include <corsika/modules/proposal/InteractionModel.hpp>

#include <vector>

namespace corsika::kokkos::test {
  // No hadronic generator is constructed or called by these compile checks.
  // Concrete module specializations ensure the new members are instantiated,
  // rather than only parsing an uninstantiated class-template declaration.
  class GetterHadronicModel {};

  std::vector<corsika::proposal::NativeInteractionCalculatorView>
  instantiate_interaction_views(
      corsika::proposal::InteractionModel<GetterHadronicModel, GetterHadronicModel> const&
          model) {
    return model.nativeCalculatorViews();
  }

  std::vector<corsika::proposal::NativeContinuousCalculatorView>
  instantiate_continuous_views(corsika::proposal::ContinuousProcess<> const& model) {
    return model.nativeCalculatorViews();
  }
} // namespace corsika::kokkos::test
