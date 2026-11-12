/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/framework/process/ProcessSelection.hpp>

namespace {
  template <typename T>
  struct Matches : std::is_same<T, int> {};
  static_assert(corsika::contains_process_v<Matches, int>);
  static_assert(!corsika::contains_process_v<Matches, double>);
} // namespace
