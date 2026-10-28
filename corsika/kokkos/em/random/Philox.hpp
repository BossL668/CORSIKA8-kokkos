/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  /** Counter/key records for the production Random123 Philox4x32-10 stream.
   * historyId_ and stepId_ fill the counter; seed_, showerId_, processId_ and drawId_
   * are mixed into the key in production order. Queue position is not an input. */
  struct PhiloxCounter {
    std::uint32_t words_[4]{};
  };

  struct PhiloxKey {
    std::uint32_t words_[2]{};
  };

  struct RandomNumberKey {
    std::uint64_t seed_{};
    std::uint64_t showerId_{};
    std::uint64_t historyId_{};
    std::uint64_t stepId_{};
    std::uint32_t processId_{};
    std::uint64_t drawId_{};
  };
  static_assert(std::is_trivially_copyable_v<RandomNumberKey>);
  /** Evaluate the unmodified ten-round counter generator. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline PhiloxCounter philox4x32_10(
      PhiloxCounter const& counter, PhiloxKey const& key);
  /** Return four random words for one complete physical draw identity. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline PhiloxCounter random_words(
      RandomNumberKey const& input);
  /** Convert lane & 3 to an open-interval uniform using (word + 0.5) / 2^32. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline double uniform_open01(
      RandomNumberKey const& input, std::uint32_t lane = 0);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/random/Philox.inl>
