/* Independent material-interface geometrical-optics radio. SI units throughout.
 * No atmosphere application, transport owner, YAML, or device runtime here. */
#pragma once
#include <corsika/geometry/terrain/FlatTerrainData.hpp>
#include <corsika/geometry/terrain/DemCoverageData.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace corsika::radio::interface {
using Vec3 = terrain::flat::Vec3;
enum class Geometry : std::uint32_t { Uniform, Plane, Mesh };
enum class Algorithm : std::uint32_t { ZHS, CoREAS };
inline char const* algorithmName(Algorithm algorithm) {
  return algorithm==Algorithm::CoREAS?"CoREAS interface endpoints":"ZHS continuous interval moments";
}
enum class PathStatus : std::uint32_t { Missing, Valid, Blocked, Invalid, Unconverged, OutsideCoverage };
// A radial index table is sampled from the caller's actual refractive model.
// Rays retain straight legs; this is NOT a bent-ray atmospheric solution.
struct IndexSample { double height_m{}, index{}; };
struct Medium {
  double refractive_index{1.};
  double attenuation_length_m{HUGE_VAL}; // electric-field amplitude e^(-L/Latt)
  Vec3 center_m{0.,0.,0.};
  double reference_radius_m{};
  std::vector<IndexSample> radial_index;
  // Known layer transitions. Optical integration splits at these spheres;
  // closely spaced table entries can represent both sides of a jump.
  std::vector<double> radial_integration_breaks_m;
};
struct Observer {
  std::string name;
  Vec3 position_m{};
  std::uint32_t region{}; // 0 enclosing, 1 embedded; separate from transport bank IDs
};
struct PropagationConfig {
  terrain::coverage::Data coverage;
  Geometry geometry{Geometry::Uniform};
  Medium media[2];
  Vec3 plane_point_m{};
  Vec3 plane_outward_normal{0.,0.,1.}; // embedded -> enclosing
  std::uint32_t optical_integration_samples{64};
  double visibility_tolerance_m{1.e-7}; // optical endpoint acceptance only
  double shared_edge_tolerance_m{1.e-7};
  bool transmission_bvh{true}; // false retains exhaustive face traversal for validation
};
struct Track {
  Vec3 start_m{}, end_m{};
  double start_time_s{}, end_time_s{};
  double charge_e{}, weight{};
  std::uint64_t history{}, step{};
  std::uint32_t region{}, valid{};
  // Optional accuracy declared by the source transport. The mountain adapter
  // retains its historical 1e-9 chord tolerance; standalone exact fixtures
  // default to input-coordinate/clock ULPs only.
  double relative_chord_tolerance{};
};
struct RayQuery { Vec3 source{}, observer{}; std::uint32_t source_region{}, observer_region{}; };
struct Path {
  PathStatus status{PathStatus::Missing};
  std::uint32_t face{terrain::flat::NoTriangle}, iterations{};
  Vec3 interface_point{}, emit{}, receive_direction{}, normal{};
  double source_index{}, destination_index{}, interface_source_index{}, interface_destination_index{};
  double source_length_m{}, destination_length_m{}, time_s{}, snell_residual{};
  double t_s{1.}, t_p{1.}, attenuation{1.}, jacobian_m2{}, spreading_per_m{};
  // Maps a source transverse vector to receiver components, including 1/R,
  // ray-tube spreading, Fresnel flux normalization and amplitude attenuation.
  double transfer_per_m[9]{};
};
struct Config {
  bool enabled{};
  Algorithm algorithm{Algorithm::ZHS};
  double coreas_cherenkov_threshold{1.e-3};
  PropagationConfig propagation;
  std::vector<Observer> observers;
  double start_time_s{-1.e-6}, sample_rate_Hz{1.e9};
  std::size_t samples{16384};
  unsigned moment_order{12};
  double subdivision_frequency_Hz{1.e9}, fraunhofer_limit{0.025};
  unsigned maximum_subdivision_depth{12};
  // Visibility and finite-face acceptance are discontinuous. Refine mesh
  // sources even when their midpoint has no accepted optical path.
  double mesh_maximum_segment_m{.1};
  std::size_t maximum_device_bytes{256u*1024u*1024u};
  // OpenMP may combine small transport fronts without changing source physics.
  // Zero retains immediate dispatch. CUDA retains its established 8192 buffer.
  std::size_t host_track_buffer_capacity{};
};
struct Statistics {
  std::uint64_t device_tracks{}, cpu_tracks{}, track_observer_pairs{}, leaves{}, paths{};
  std::uint64_t direct_paths{}, transmitted_paths{}, blocked_paths{}, rejected_faces{};
  // wavefronts counts radio kernel launches; CUDA merges transport batches.
  std::uint64_t wavefronts{}, downloads{}, out_of_window{}, errors{};
  std::uint64_t endpoint_contributions{}, regularized_pairs{}, boundary_endpoints{};
  std::uint64_t roundoff_limited_tracks{}, outside_coverage_paths{};
  std::size_t device_bytes{};
};
struct Result {
  Config config;
  Statistics statistics;
  // (order, observer * 6 + region * 3 + component, time cell).
  // Stored area moments in V s^2/m; final E(f) = -i 2 pi f A(f).
  std::vector<double> moments;
  // CoREAS: moments above contain E impulses (V s/m), with no differentiation.
  // This optional second grid contains only the finite-track Cherenkov-limit
  // potential (V s^2/m), differentiated once at final rendering.
  std::vector<double> regularized_moments;
};
// One source stream, two independent emission representations. Shared device
// allocations are counted once here, not by adding the per-result statistics.
struct PairedResult {
  Result coreas, zhs;
  std::size_t device_bytes{};
};
} // namespace corsika::radio::interface
