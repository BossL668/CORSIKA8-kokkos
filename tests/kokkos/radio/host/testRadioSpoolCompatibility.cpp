/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "ProductionRadioTrackSpool.hpp"
#include <corsika/kokkos/radio/RadioTrackSpoolReader.hpp>
#include <corsika/kokkos/radio/RadioTrackSpoolWriter.hpp>
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <iterator>
namespace {
  struct Directory {
    std::filesystem::path path_;
    Directory() {
      char name[] = "/tmp/c8-spool-oracle-XXXXXX";
      auto created = mkdtemp(name);
      if (!created) throw std::runtime_error("test directory");
      path_ = created;
    }
    ~Directory() {
      std::error_code e;
      std::filesystem::remove_all(path_, e);
    }
  };
  auto bytes(std::filesystem::path const& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(f), {});
  }
} // namespace
TEST_CASE("Spool version 1 remains byte-compatible with pinned production",
          "[kokkos][radio][host]") {
  namespace cr = corsika::kokkos::radio;
  Directory tmp;
  auto current = tmp.path_ / "current.bin";
  auto original = tmp.path_ / "original.bin";
  cr::RadioTrackSpoolWriter out(current);
  production_host_radio::RadioTrackSpoolWriter ref(original);
  for (unsigned i = 0; i < 137; ++i) {
    corsika::radio::RadioTrackSegment a{};
    production_host_radio::RadioTrackSegment b{};
    a.pdg_ = b.pdg = i % 2 ? -11 : 11;
    a.weight_ = b.weight = i * 0.5;
    a.startTimeS_ = b.start_time_s = i * 1.e-9;
    a.endTimeS_ = b.end_time_s = (i + 1) * 1.e-9;
    a.startKineticEnergyGeV_ = b.start_kinetic_energy_GeV = 0.1 + i;
    a.endKineticEnergyGeV_ = b.end_kinetic_energy_GeV = 0.05 + i;
    for (unsigned j = 0; j < 3; ++j) {
      a.startPositionM_[j] = b.start_position_m[j] = i + j * 0.5;
      a.endPositionM_[j] = b.end_position_m[j] = i - j * 0.25;
      a.startDirection_[j] = b.start_direction[j] = j == 2 ? -1. : 0.;
      a.endDirection_[j] = b.end_direction[j] = j == 2 ? -1. : 0.;
    }
    out.append(a);
    ref.append(b);
  }
  out.complete();
  ref.complete();
  REQUIRE(bytes(current) == bytes(original));
  REQUIRE(bytes(current).size() == 32 + 137 * 140);
  // Cross-read both directions, then re-encode with the other implementation.
  cr::RadioTrackSpoolReader fromOld(original);
  production_host_radio::RadioTrackSpoolReader fromNew(current);
  cr::RadioTrackSpoolWriter roundNew(tmp.path_ / "new-round.bin");
  production_host_radio::RadioTrackSpoolWriter roundOld(tmp.path_ / "old-round.bin");
  std::vector<corsika::radio::RadioTrackSegment> a;
  std::vector<production_host_radio::RadioTrackSegment> b;
  while (fromOld.readBatch(a, 11))
    for (auto const& x : a) roundNew.append(x);
  while (fromNew.readBatch(b, 19))
    for (auto const& x : b) roundOld.append(x);
  roundNew.complete();
  roundOld.complete();
  REQUIRE(fromOld.finished());
  REQUIRE(fromNew.finished());
  REQUIRE(bytes(current) == bytes(tmp.path_ / "new-round.bin"));
  REQUIRE(bytes(original) == bytes(tmp.path_ / "old-round.bin"));
}
