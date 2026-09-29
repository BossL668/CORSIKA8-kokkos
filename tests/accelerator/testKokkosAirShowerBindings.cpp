#include "../../applications/detail/air_shower_kokkos/KokkosAirShowerBindings.hpp"
#include <array>
#include <stdexcept>
#include <vector>

namespace air = corsika::applications::air_shower;
void require(bool value) { if (!value) throw std::runtime_error("binding regression"); }
struct Model {
  Model() = default;
  Model(Model const&) = delete;
};
struct Counter {
  std::vector<int>& calls;
  int id;
  unsigned count;
  std::vector<int> timings;
  unsigned getCount() { calls.push_back(id); return count; }
  auto const& getTimingSamples() { calls.push_back(id + 10); return timings; }
};
struct Pool {
  struct Stats { int count = 0; };
  std::vector<int>& calls;
  Stats statistics() { calls.push_back(30); return {42}; }
};
struct Cascade {
  std::vector<int> calls;
  void forceInteraction() { calls.push_back(1); }
  void forceDecay() { calls.push_back(2); }
};
int main() {
  std::array<Model, 11> m;
  air::KokkosAirShowerModels models{m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[8],m[9],m[10]};
  require(&models.coreas == &m[0] && &models.zhs == &m[1]);
  require(&models.em_cascade == &m[2] && &models.em_continuous_proposal == &m[3]);
  require(&models.neutrino_primary == &m[4] && &models.hadron_sequence == &m[5]);
  require(&models.decay_sequence == &m[6] && &models.em_continuous == &m[7]);
  require(&models.longitudinal == &m[8] && &models.production == &m[9]);
  require(&models.cut == &m[10]);
  std::vector<int> calls;
  Counter high{calls,2,12,{1,2,3}}, low{calls,1,6,{1}};
  Pool pool{calls};
  Model photo_high, photo_low, fallback;
  air::KokkosAirShowerMonitors monitors{photo_high, photo_low, high, low, &pool, fallback};
  int ph = 7, pl = 3;
  auto before = monitors.capture(ph, pl);
  require(calls == std::vector<int>({1,2,11,12,30}));
  high.count = 100; ph = 20; pl = 40;
  require(before.photo_high == 7 && before.photo_low == 3);
  require(before.high_interactions == 12 && before.low_interactions == 6);
  require(before.high_timings == 3 && before.low_timings == 1 && before.pool.count == 42);
  calls.clear(); monitors.pool = nullptr;
  require(monitors.capture(ph,pl).pool.count == 0);
  require(calls == std::vector<int>({1,2,11,12}));
  for (int flags = 0; flags < 4; ++flags) {
    Cascade underlying;
    int runs = 0;
    auto EAS = air::KokkosCascade{[&](auto& configure) { configure(underlying); ++runs; }};
    if (flags & 1) EAS.forceInteraction();
    if (flags & 2) EAS.forceDecay();
    require(underlying.calls.empty() && runs == 0);
    EAS.run();
    std::vector<int> expected;
    if (flags & 1) expected.push_back(1);
    if (flags & 2) expected.push_back(2);
    require(underlying.calls == expected && runs == 1);
  }
  bool propagated = false;
  auto failing = air::KokkosCascade{[](auto&) { throw std::runtime_error("physical failure"); }};
  try { failing.run(); } catch (std::runtime_error const&) { propagated = true; }
  require(propagated);
}
