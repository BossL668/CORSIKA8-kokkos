/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <typename TExecutionSpace>
  template <typename TView, typename T>
  void EnvironmentStorage<TExecutionSpace>::upload(TView& out, char const* name,
                                                   T const* data, std::uint32_t count) {
    if (count && !data) {
      throw std::invalid_argument("null full-environment snapshot array");
    }
    out = TView(std::string(name), count);
    auto host = Kokkos::create_mirror_view(out);
    for (std::uint32_t i = 0; i < count; ++i) { host(i) = data[i]; }
    Kokkos::deep_copy(out, host);
  }

  template <typename TExecutionSpace>
  EnvironmentStorage<TExecutionSpace>::EnvironmentStorage(
      EnvironmentDataView const& host) {
    upload(nodes_, "full environment nodes", host.nodes_, host.nodeCount_);
    upload(shapes_, "full environment shapes", host.shapes_, host.nodeCount_);
    upload(children_, "full environment children", host.children_, host.childCount_);
    upload(exclusions_, "full environment exclusions", host.exclusions_,
           host.exclusionCount_);
    upload(models_, "full environment models", host.models_, host.modelCount_);
    upload(composition_, "full environment composition", host.composition_,
           host.componentCount_);
  }

  template <typename TExecutionSpace>
  auto EnvironmentStorage<TExecutionSpace>::getView() const -> view_type {
    return {typename view_type::tree_type(nodes_, shapes_, children_, exclusions_),
            models_, composition_};
  }

  template <typename TExecutionSpace>
  EnvironmentDataView EnvironmentStorage<TExecutionSpace>::getDataView() const {
    return {nodes_.data(),
            shapes_.data(),
            children_.data(),
            exclusions_.data(),
            models_.data(),
            composition_.data(),
            static_cast<std::uint32_t>(nodes_.size()),
            static_cast<std::uint32_t>(children_.size()),
            static_cast<std::uint32_t>(exclusions_.size()),
            static_cast<std::uint32_t>(models_.size()),
            static_cast<std::uint32_t>(composition_.size())};
  }

  template <typename TExecutionSpace>
  std::size_t EnvironmentStorage<TExecutionSpace>::getDeviceBytes() const {
    return nodes_.span() * sizeof(EnvironmentNode) +
           shapes_.span() * sizeof(EnvironmentShape) +
           (children_.span() + exclusions_.span()) * sizeof(std::int32_t) +
           models_.span() * sizeof(EnvironmentModel) +
           composition_.span() * sizeof(EnvironmentComponent);
  }

} // namespace corsika::kokkos
