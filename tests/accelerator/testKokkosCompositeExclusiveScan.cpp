/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <corsika/accelerator/em/kokkos/KokkosCompositeExclusiveScan.hpp>

namespace {

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using Memory = typename ExecutionSpace::memory_space;

  template <std::size_t Columns>
  using Counts =
      corsika::accelerator::em::kokkos_detail::CompositeUint32Counts<Columns>;

  template <std::size_t Columns>
  using Value =
      corsika::accelerator::em::kokkos_detail::CompositeScanValue<Columns>;

  template <std::size_t Columns>
  using Workspace = corsika::accelerator::em::kokkos_detail::
      KokkosCompositeExclusiveScanWorkspace<ExecutionSpace, Columns>;

  template <std::size_t Columns>
  std::vector<Counts<Columns>> makeInput(std::size_t const count) {
    std::vector<Counts<Columns>> result(count);
    for (std::size_t source = 0; source < count; ++source) {
      for (std::size_t column = 0; column < Columns; ++column) {
        // Exercise zero columns as well as totals beyond uint32 range.
        result[source].values[column] =
            column == Columns - 1
                ? 0U
                : static_cast<std::uint32_t>(
                      500000U + ((source * 17U + column * 13U) % 1000001U));
      }
    }
    return result;
  }

  template <std::size_t Columns>
  void validate(std::vector<Counts<Columns>> const& input,
                Kokkos::View<Value<Columns>*, Memory> const& prefixes,
                Kokkos::View<Value<Columns>, Memory> const& total,
                ExecutionSpace const& execution) {
    auto host_prefixes = Kokkos::create_mirror_view(prefixes);
    auto host_total = Kokkos::create_mirror_view(total);
    Kokkos::deep_copy(execution, host_prefixes, prefixes);
    Kokkos::deep_copy(execution, host_total, total);
    execution.fence("download composite-scan test output");

    std::array<std::uint64_t, Columns> expected{};
    for (std::size_t source = 0; source < input.size(); ++source) {
      for (std::size_t column = 0; column < Columns; ++column) {
        if (host_prefixes(source).values[column] != expected[column])
          throw std::runtime_error(
              "composite exclusive prefix mismatch at source " +
              std::to_string(source) + ", column " +
              std::to_string(column));
        expected[column] += input[source].values[column];
      }
    }
    for (std::size_t column = 0; column < Columns; ++column) {
      if (host_total().values[column] != expected[column])
        throw std::runtime_error("composite total mismatch at column " +
                                 std::to_string(column));
    }
  }

  template <std::size_t Columns>
  void runCase(std::size_t const count, ExecutionSpace const& execution) {
    namespace detail = corsika::accelerator::em::kokkos_detail;
    auto const host_input_values = makeInput<Columns>(count);
    Kokkos::View<Counts<Columns>*, Memory> input("composite_scan_input",
                                                 count);
    auto host_input = Kokkos::create_mirror_view(input);
    for (std::size_t source = 0; source < count; ++source)
      host_input(source) = host_input_values[source];
    Kokkos::deep_copy(execution, input, host_input);

    Kokkos::View<Value<Columns>*, Memory> automatic_prefixes(
        "automatic_composite_scan_prefixes", count);
    Kokkos::View<Value<Columns>, Memory> automatic_total(
        "automatic_composite_scan_total");
    Workspace<Columns> workspace;
    workspace.exclusiveScan(
        input, count, automatic_prefixes, automatic_total,
        detail::CompositeUint32Loader<Columns>{}, execution);
    validate(host_input_values, automatic_prefixes, automatic_total,
             execution);

    if constexpr (Workspace<Columns>::cubAvailable()) {
      if (count != 0 && !workspace.usedCubForLastScan())
        throw std::runtime_error(
            "automatic Kokkos-CUDA composite scan did not use CUB");
      if (count != 0 &&
          (workspace.cubCapacity() < count ||
           workspace.cubTemporaryBytes() == 0))
        throw std::runtime_error(
            "persistent CUB composite-scan workspace was not allocated");

      if (count > 1) {
        auto const retained_capacity = workspace.cubCapacity();
        auto const retained_bytes = workspace.cubTemporaryBytes();
        auto const smaller_count = count / 2;
        workspace.exclusiveScan(
            input, smaller_count, automatic_prefixes, automatic_total,
            detail::CompositeUint32Loader<Columns>{}, execution);
        if (!workspace.usedCubForLastScan() ||
            workspace.cubCapacity() != retained_capacity ||
            workspace.cubTemporaryBytes() != retained_bytes)
          throw std::runtime_error(
              "smaller repeated scan did not reuse persistent CUB storage");
        validate(std::vector<Counts<Columns>>(host_input_values.begin(),
                                              host_input_values.begin() +
                                                  smaller_count),
                 automatic_prefixes, automatic_total, execution);
      }
    } else if (workspace.usedCubForLastScan() ||
               workspace.cubCapacity() != 0 ||
               workspace.cubTemporaryBytes() != 0) {
      throw std::runtime_error(
          "portable composite scan unexpectedly exposes CUDA state");
    }

    Kokkos::View<Value<Columns>*, Memory> portable_prefixes(
        "portable_composite_scan_prefixes", count);
    Kokkos::View<Value<Columns>, Memory> portable_total(
        "portable_composite_scan_total");
    auto const retained_capacity = workspace.cubCapacity();
    workspace.exclusiveScan(
        input, count, portable_prefixes, portable_total,
        detail::CompositeUint32Loader<Columns>{}, execution,
        detail::CompositeScanImplementation::Portable);
    if (workspace.usedCubForLastScan())
      throw std::runtime_error("forced portable scan reported CUB use");
    if (workspace.cubCapacity() != retained_capacity)
      throw std::runtime_error(
          "forced portable scan changed persistent CUB capacity");
    validate(host_input_values, portable_prefixes, portable_total, execution);
  }

  template <std::size_t Columns>
  void runEmptyCase(ExecutionSpace const& execution) {
    namespace detail = corsika::accelerator::em::kokkos_detail;
    Kokkos::View<Counts<Columns>*, Memory> input("empty_scan_input", 0);
    Kokkos::View<Value<Columns>*, Memory> prefixes("empty_scan_prefixes", 0);
    Kokkos::View<Value<Columns>, Memory> total("empty_scan_total");
    Workspace<Columns> workspace;
    workspace.exclusiveScan(input, 0, prefixes, total,
                            detail::CompositeUint32Loader<Columns>{},
                            execution);
    validate(std::vector<Counts<Columns>>{}, prefixes, total, execution);
  }

} // namespace

int main(int argc, char** argv) {
  try {
    Kokkos::ScopeGuard guard(argc, argv);
    ExecutionSpace execution;
    runCase<7>(8193, execution);
    runCase<8>(10007, execution);
    runCase<7>(19, execution);
    runCase<8>(23, execution);
    runEmptyCase<7>(execution);
    runEmptyCase<8>(execution);
    std::cout << "Kokkos composite exclusive-scan tests passed\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "Kokkos composite exclusive-scan test failed: "
              << error.what() << '\n';
    return 1;
  }
}
