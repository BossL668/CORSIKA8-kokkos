/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/setup/SetupStack.hpp>
#include <corsika/stack/TransportIdentityStackExtension.hpp>

namespace corsika::test {
  template <typename TEnvironment>
  struct TransportStack {
    using base_type =
        typename setup::detail::StackGenerator<TEnvironment>::StackWithWeight;
    template <typename T>
    using interface_type =
        CombinedParticleInterface<base_type::template pi_type,
                                  transport::TransportIdentityDataInterface, T>;
    using type = CombinedStack<typename base_type::stack_data_type,
                               transport::TransportIdentityData, interface_type>;

    using history_base_type =
        typename setup::detail::StackGenerator<TEnvironment>::StackWithHistory;
    template <typename T>
    using history_interface_type =
        CombinedParticleInterface<history_base_type::template pi_type,
                                  transport::TransportIdentityDataInterface, T>;
    using history_type =
        CombinedStack<typename history_base_type::stack_data_type,
                      transport::TransportIdentityData, history_interface_type,
                      history::HistorySecondaryProducer>;
  };
} // namespace corsika::test
