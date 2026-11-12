/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

namespace corsika {
  /** Borrow matching processes from a configured process tree.
   * A match retains its complete scope: traversal stops at matching composites.
   * Neither switch conditions nor physics methods are called. The caller must
   * keep the process tree alive while using the returned references.
   *
   * \tparam TMatch Compile-time predicate on the unqualified process type.
   * \param process An existing process or composition exposing getProcesses().
   */
  template <template <typename> class TMatch, typename TProcess>
  auto matching_processes(TProcess& process);

  /** Borrow exactly one matching process, rejecting missing or ambiguous matches
   * at compile time. This function never chooses a process by its position.
   */
  template <template <typename> class TMatch, typename TProcess>
  decltype(auto) select_process(TProcess& process);

  /** Whether a process tree contains at least one match, without evaluating it. */
  template <template <typename> class TMatch, typename TProcess>
  inline constexpr bool contains_process_v =
      std::tuple_size_v<decltype(
          matching_processes<TMatch>(std::declval<TProcess&>()))> != 0;
} // namespace corsika

#include <corsika/detail/framework/process/ProcessSelection.inl>
