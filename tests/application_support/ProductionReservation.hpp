/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/stack/TransportIdentityStackExtension.hpp>
#include <cstdint>
#include <utility>
struct ProductionReservation {
  using value_type = corsika::transport::TransportIdentityData;
  value_type data_;
  template <typename U = value_type>
  auto reserveTransportHistoryIds(std::uint64_t count)
      -> decltype(std::declval<U&>().reserveHistoryIds(count)) {
    return data_.reserveHistoryIds(count);
  }
};
