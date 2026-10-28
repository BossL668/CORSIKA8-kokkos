/*
 * (c) Copyright 2020, 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

// Intentionally no include guard: this single algorithm body is included inside
// the native host and Kokkos host/device next_environment_boundary functions.
// Both supply TParameter, access, current and limit. Separate execution-space
// declarations keep native units and node interfaces out of device compilation,
// without duplicating the algorithm or introducing a run-time dispatch branch.

auto selectedParameter = limit;
auto selectedNode = current;
TParameter const zero{};
auto const own = access.intersect(current);
if (own.hasIntersections() && own.getExit() <= selectedParameter) {
  selectedParameter = own.getExit();
  selectedNode = access.getParent(current);
}
for (unsigned int childIndex = 0; childIndex < access.getChildCount(current);
     ++childIndex) {
  auto const child = access.getChild(current, childIndex);
  auto const intersections = access.intersect(child);
  if (!intersections.hasIntersections()) { continue; }
  auto const entry = intersections.getEntry();
  auto const exit = intersections.getExit();
  if (exit > zero && entry <= selectedParameter && -entry < exit) {
    selectedParameter = entry;
    selectedNode = child;
  }
}
for (unsigned int excludedIndex = 0; excludedIndex < access.getExcludedCount(current);
     ++excludedIndex) {
  auto const excluded = access.getExcluded(current, excludedIndex);
  auto const intersections = access.intersect(excluded);
  if (!intersections.hasIntersections()) { continue; }
  auto const entry = intersections.getEntry();
  if (intersections.getExit() > zero && entry <= selectedParameter) {
    selectedParameter = entry;
    selectedNode = excluded;
  }
}
if (selectedParameter < zero) { selectedParameter = zero; }
return {selectedParameter, selectedNode};
