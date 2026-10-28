/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/detail/framework/random/random_iterator/detail/Random123/philox.h>

namespace corsika::kokkos {

  namespace philox_detail {
    CORSIKA_HOST_DEVICE inline std::uint64_t splitmix64(std::uint64_t value) {
      value += 0x9e3779b97f4a7c15ULL;
      value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
      value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
      return value ^ (value >> 31U);
    }
  } // namespace philox_detail

  CORSIKA_HOST_DEVICE inline PhiloxCounter philox4x32_10(PhiloxCounter const& counter,
                                                         PhiloxKey const& key) {
    random_iterator_r123::Philox4x32::ctr_type r123_counter{
        {counter.words_[0], counter.words_[1], counter.words_[2], counter.words_[3]}};
    random_iterator_r123::Philox4x32::key_type r123_key{{key.words_[0], key.words_[1]}};
    auto const output = random_iterator_r123::Philox4x32{}(r123_counter, r123_key);
    return {{output.v[0], output.v[1], output.v[2], output.v[3]}};
  }

  CORSIKA_HOST_DEVICE inline PhiloxCounter random_words(RandomNumberKey const& input) {
    PhiloxCounter const counter{{
        static_cast<std::uint32_t>(input.historyId_),
        static_cast<std::uint32_t>(input.historyId_ >> 32U),
        static_cast<std::uint32_t>(input.stepId_),
        static_cast<std::uint32_t>(input.stepId_ >> 32U),
    }};

    auto key_material = philox_detail::splitmix64(input.seed_);
    key_material = philox_detail::splitmix64(key_material ^ input.showerId_);
    key_material = philox_detail::splitmix64(
        key_material ^ static_cast<std::uint64_t>(input.processId_));
    key_material = philox_detail::splitmix64(key_material ^ input.drawId_);
    PhiloxKey const key{{
        static_cast<std::uint32_t>(key_material),
        static_cast<std::uint32_t>(key_material >> 32U),
    }};
    return philox4x32_10(counter, key);
  }

  CORSIKA_HOST_DEVICE inline double uniform_open01(RandomNumberKey const& input,
                                                   std::uint32_t lane) {
    auto const words = random_words(input);
    auto const value = words.words_[lane & 3U];
    return (static_cast<double>(value) + 0.5) * (1. / 4294967296.);
  }

} // namespace corsika::kokkos
