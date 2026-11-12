/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika {
  namespace process_selection_detail {
    template <typename T, typename = void>
    struct HasComponents : std::false_type {};
    template <typename T>
    struct HasComponents<T, std::void_t<decltype(std::declval<T&>().getProcesses())>>
        : std::true_type {};
  } // namespace process_selection_detail

  template <template <typename> class TMatch, typename TProcess>
  auto matching_processes(TProcess& process) {
    if constexpr (TMatch<std::remove_cv_t<TProcess>>::value) {
      return std::tie(process);
    } else if constexpr (process_selection_detail::HasComponents<TProcess>::value) {
      return std::apply(
          [](auto&... components) {
            return std::tuple_cat(matching_processes<TMatch>(components)...);
          },
          process.getProcesses());
    } else {
      return std::tuple<>{};
    }
  }

  template <template <typename> class TMatch, typename TProcess>
  decltype(auto) select_process(TProcess& process) {
    auto matches = matching_processes<TMatch>(process);
    static_assert(std::tuple_size_v<decltype(matches)> == 1,
                  "Execution adapter requires exactly one matching process");
    if constexpr (std::tuple_size_v<decltype(matches)> == 1) return std::get<0>(matches);
  }
} // namespace corsika
