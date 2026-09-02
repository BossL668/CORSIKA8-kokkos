/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/detail/framework/random/random_iterator/detail/Random123/philox.h>

#define CORSIKA_GPU_HOST_DEVICE C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em {

  struct PhiloxCounter {
    std::uint32_t words[4]{};
  };

  struct PhiloxKey {
    std::uint32_t words[2]{};
  };

  struct RandomNumberKey {
    std::uint64_t seed{};
    std::uint64_t shower_id{};
    std::uint64_t history_id{};
    std::uint64_t step_id{};
    std::uint32_t process_id{};
    std::uint64_t draw_id{};
  };

  namespace detail {

    CORSIKA_GPU_HOST_DEVICE inline std::uint64_t splitmix64(std::uint64_t value) {
      value += 0x9e3779b97f4a7c15ULL;
      value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
      value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
      return value ^ (value >> 31U);
    }

  } // namespace detail

  CORSIKA_GPU_HOST_DEVICE inline PhiloxCounter philox4x32_10(
      PhiloxCounter const& counter, PhiloxKey const& key) {
    random_iterator_r123::Philox4x32::ctr_type r123_counter{{
        counter.words[0], counter.words[1], counter.words[2], counter.words[3]}};
    random_iterator_r123::Philox4x32::key_type r123_key{{
        key.words[0], key.words[1]}};
    auto const output = random_iterator_r123::Philox4x32{}(r123_counter, r123_key);
    return {{output.v[0], output.v[1], output.v[2], output.v[3]}};
  }

  /**
   * Map the scheduling-independent random key to a Philox counter and key.
   *
   * history_id and step_id occupy the complete 128-bit counter. The seed, shower, process and
   * draw identifiers are folded into the 64-bit Philox key with SplitMix64. Consequently a
   * history's stream does not depend on its location in a queue or on CUDA launch geometry.
   */
  CORSIKA_GPU_HOST_DEVICE inline PhiloxCounter randomWords(RandomNumberKey const& input) {
    PhiloxCounter const counter{{
        static_cast<std::uint32_t>(input.history_id),
        static_cast<std::uint32_t>(input.history_id >> 32U),
        static_cast<std::uint32_t>(input.step_id),
        static_cast<std::uint32_t>(input.step_id >> 32U),
    }};

    auto key_material = detail::splitmix64(input.seed);
    key_material = detail::splitmix64(key_material ^ input.shower_id);
    key_material =
        detail::splitmix64(key_material ^ static_cast<std::uint64_t>(input.process_id));
    key_material = detail::splitmix64(key_material ^ input.draw_id);
    PhiloxKey const key{{
        static_cast<std::uint32_t>(key_material),
        static_cast<std::uint32_t>(key_material >> 32U),
    }};
    return philox4x32_10(counter, key);
  }

  CORSIKA_GPU_HOST_DEVICE inline double uniformOpen01(RandomNumberKey const& input,
                                                       std::uint32_t lane = 0) {
    auto const words = randomWords(input);
    auto const value = words.words[lane & 3U];
    return (static_cast<double>(value) + 0.5) * (1. / 4294967296.);
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_HOST_DEVICE
