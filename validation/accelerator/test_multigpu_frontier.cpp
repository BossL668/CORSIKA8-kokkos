#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include "../../applications/detail/air_shower_multigpu/Frontier.hpp"
#include <corsika/setup/SetupStack.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/IMediumModel.hpp>
#include <corsika/accelerator/em/common/FullEnvironmentTreeBuilder.hpp>
#include <filesystem>
#include <unistd.h>

struct FakeBackend { std::size_t minimumBatchSize() const { return 2; } };
struct FakeRouter {
  bool busy = true;
  int advances = 0;
  template<class... Args> explicit FakeRouter(Args&&...) {}
  bool pending() const { return busy; }
  template<class Stack> std::size_t advanceOneWavefrontAndReturn(Stack&) {
    ++advances; busy = false; return 0;
  }
  void endOfShower() {}
};

int main() {
  using namespace corsika;
  using namespace corsika::applications::multigpu;
  using Env = Environment<IMediumModel>;
  using Stack = setup::HybridStack<Env>;
  Env env;
  auto cs = env.getCoordinateSystem();
  State s{};
  s.pid = 11; s.energy_GeV = 1.; s.weight = 3.;
  s.direction[2] = -1.; s.position_m[2] = 6372000.;
  s.history_id = 8; s.parent_history_id = 1; s.generation = 1;
  Stack first;
  auto particle = gpu::em::router_detail::importParticle(first, s, cs);
  auto step = particle.beginTransportStep();
  if (step != 0 || particle.getStepId() != 1) return 1;
  char name[] = "/tmp/c8-multigpu-frontier-XXXXXX";
  int fd = mkstemp(name);
  if (fd < 0) return 2;
  close(fd);
  try {
    CaptureRouter capture(name, cs, 2.);
    if (!capture.canRoute(particle, step)) return 3;
    capture.stage(particle, particle.getHistoryId(), particle.getParentHistoryId(),
                  particle.getGeneration(), step);
    capture.endOfShower();
    Stack second;
    if (importFrontier(second, cs, name, 2) != 1) return 4;
    auto received = second.last();
    if (received.getHistoryId() != 8 || received.getParentHistoryId() != 1 ||
        received.getGeneration() != 1 || received.getStepId() != 0 ||
        received.getWeight() != 3. || received.beginTransportStep() != 0 ||
        second.reserveTransportHistoryIds(1) != 2 * historyStride) return 5;

    // Input is high-energy-first. Reverse ONLY bounded insertion, retaining
    // weight, parent and zero-based step. No full host-stack materialization.
    auto low = s; low.history_id = 9; low.energy_GeV = .5; low.step_id = 7;
    auto tail = s; tail.history_id = 10; tail.energy_GeV = .25;
    {
      std::ofstream data(name);
      data << "C8_STATIC_FRONTIER_V1\n" << std::setprecision(17);
      write(data, s); write(data, low); write(data, tail);
    }
    FrontierInput source(name);
    Stack bounded;
    source.reserveNamespace(bounded, 3);
    if (source.refill(bounded, cs, 2) != 2 || bounded.getSize() != 2 ||
        bounded.last().getHistoryId() != 8 || bounded.at(0).getHistoryId() != 9 ||
        bounded.at(0).getStepId() != 7 || bounded.last().getWeight() != 3.) return 7;
    bool nonempty_rejected = false;
    try { source.refill(bounded, cs, 2); }
    catch (std::logic_error const&) { nonempty_rejected = true; }
    if (!nonempty_rejected) return 8;
    bounded.clear(); // unit-test disposal only; real cascade consumes the batch
    if (source.refill(bounded, cs, 2) != 1 || source.pending() ||
        source.count() != 3 || source.batches() != 2 || source.peakBatch() != 2 ||
        bounded.last().getHistoryId() != 10) return 9;

    // Duplicate histories across separate batches fail closed.
    {
      std::ofstream data(name);
      data << "C8_STATIC_FRONTIER_V1\n" << std::setprecision(17);
      write(data, s); write(data, s);
    }
    FrontierInput duplicate(name);
    bounded.clear(); duplicate.refill(bounded, cs, 1); bounded.clear();
    bool duplicate_rejected = false;
    try { duplicate.refill(bounded, cs, 1); }
    catch (std::runtime_error const&) { duplicate_rejected = true; }
    if (!duplicate_rejected) return 10;

    // Queued roots keep the shower alive; never refill while native resident
    // work/fallback is pending, and never finalize with unconsumed roots.
    {
      std::ofstream data(name);
      data << "C8_STATIC_FRONTIER_V1\n" << std::setprecision(17); write(data, s);
    }
    FrontierInput delayed(name);
    bounded.clear();
    {
      ScopedFrontierInput scope(&delayed);
      FakeBackend backend; int fallback = 0, output = 0;
      BufferedFrontierRouter<FakeRouter> router(backend, cs, env, fallback, output);
      if (!router.pending() || router.advanceOneWavefrontAndReturn(bounded) != 0 ||
          !bounded.isEmpty() || router.advances != 1 || !router.pending()) return 11;
      bool premature_rejected = false;
      try { router.endOfShower(); }
      catch (std::logic_error const&) { premature_rejected = true; }
      if (!premature_rejected || router.advanceOneWavefrontAndReturn(bounded) != 1 ||
          delayed.pending() || router.pending()) return 12;
    }
    if (activeFrontierInput) return 13;
    // Exact multiples require one EOF probe, not a phantom root or lost batch.
    FrontierInput exact(name); bounded.clear();
    if (exact.refill(bounded, cs, 1) != 1) return 14;
    bounded.clear();
    if (exact.refill(bounded, cs, 1) != 0 || exact.pending() || exact.count() != 1) return 15;
    // V2 keeps the native logical owner even if rounded position is outside it.
    auto inner = env.createNode<Sphere>(Point(cs, 0_m, 0_m, 6372000_m), 1_m);
    auto const* innerNode = inner.get();
    env.getUniverse()->addChild(std::move(inner));
    gpu::em::environment_tree::FullHostTree<Env::BaseNodeType> exported;
    exported.topology = gpu::em::environment_tree::exportEnvironmentTree(
        *env.getUniverse(), cs, [](auto const&, unsigned) { return -1; });
    gpu::em::environment_tree::SnapshotStorage storage(exported);
    auto tree = storage.view();
    auto boundary = s;
    boundary.position_m[2] = std::nextafter(6372001., INFINITY);
    boundary.environment_node = exported.topology.id(innerNode);
    first.clear();
    auto edge = gpu::em::router_detail::importParticle(first, boundary, cs, tree);
    if (env.getUniverse()->getContainingNode(edge.getPosition()) == innerNode) return 16;
    {
      CaptureRouter capture(name, cs, 2., tree);
      capture.stage(edge, edge.getHistoryId(), edge.getParentHistoryId(),
                    edge.getGeneration(), edge.getStepId());
      capture.endOfShower();
    }
    FrontierInput nodeAware(name);
    nodeAware.setEnvironmentTree(tree);
    bounded.clear();
    if (nodeAware.refill(bounded, cs, 2) != 1 || bounded.last().getNode() != innerNode ||
        bounded.last().getPosition().getCoordinates(cs)[2] / 1_m != boundary.position_m[2]) return 17;
    // Both bounded and legacy import preserve the same ID. A different tree
    // must fail before importing any root; old V1 rows must not silently guess.
    Stack legacy;
    if (importFrontier(legacy, cs, name, 4, tree) != 1 || legacy.last().getNode() != innerNode) return 18;
    auto changed = storage;
    changed.nodes[1].radius += 1.;
    bool mismatch = false;
    try { FrontierInput wrong(name); wrong.setEnvironmentTree(changed.view()); }
    catch (std::runtime_error const&) { mismatch = true; }
    if (!mismatch) return 19;
    {
      std::ofstream data(name);
      data << "C8_STATIC_FRONTIER_V1\n" << std::setprecision(17); write(data, s);
    }
    mismatch = false;
    try { FrontierInput old(name); old.setEnvironmentTree(tree); }
    catch (std::runtime_error const&) { mismatch = true; }
    if (!mismatch) return 20;
    s.direction[2] = 0.;
    bool rejected = false;
    try { validate(s); } catch (std::runtime_error const&) { rejected = true; }
    if (!rejected) return 6;
  } catch (...) {
    std::filesystem::remove(name);
    throw;
  }
  std::filesystem::remove(name);
  return 0;
}
