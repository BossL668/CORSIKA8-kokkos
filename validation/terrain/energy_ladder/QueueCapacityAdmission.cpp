// PSR-only configuration admission: exercise the enlarged queue at full capacity.
#include <corsika/modules/transport/kokkos/KokkosInterfaceQueue.hpp>
#include <corsika/modules/transport/kokkos/ExecutionSpace.hpp>
#include <corsika/accelerator/em/common/EmThinning.hpp>
#include <iostream>
#include <stdexcept>

namespace api = corsika::interfaces;
using Space = api::kokkos::ExecutionSpace;
using Queue = api::kokkos::InterfaceQueue<Space>;
using Particle = api::em::EmParticleState;
void require(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  try {
    constexpr std::size_t capacity = 4194304, batch = 4096;
    Kokkos::InitializationSettings settings;
    settings.set_num_threads(256);
    Kokkos::ScopeGuard runtime(settings);
    Space execution;
    require(execution.concurrency() == 256, "Expected 256 OpenMP workers");
    Queue queue(capacity, batch, execution);
    Queue::Particles particles("admission_particles", capacity);
    Kokkos::parallel_for("initialize_admission", Kokkos::RangePolicy<Space>(execution, 0, capacity),
      KOKKOS_LAMBDA(std::size_t i) {
        Particle p;
        p.pid = 11; p.medium_id = 93; p.energy_GeV = 0.02;
        p.direction[0] = 1.; p.history_id = i+1; p.parent_history_id = i+100;
        p.step_id = 5; p.generation = 8; p.weight = 1.;
        particles(i) = p;
      });
    queue.append(particles, capacity, execution);
    require(queue.size() == capacity, "Full queue admission failed");
    Queue::Records records("admission_records", batch);
    Kokkos::parallel_for("admission_continuations", Kokkos::RangePolicy<Space>(execution, 0, batch),
      KOKKOS_LAMBDA(std::size_t i) {
        api::EmStep r;
        r.outcome = api::EmOutcome::Continuation;
        r.end = particles(i); r.end.step_id = 6;
        records(i) = r;
      });
    api::MaterialInterface binding{17, 93, 0, 1};
    auto committed = queue.commitControlled(records, batch, binding, execution);
    require(committed.successors == batch && queue.size() == capacity, "Full queue rotation failed");
    queue.copyPrefix(particles, capacity, execution);
    std::uint64_t errors = 0;
    Kokkos::parallel_reduce("admission_preservation", Kokkos::RangePolicy<Space>(execution, 0, capacity),
      KOKKOS_LAMBDA(std::size_t i, std::uint64_t& count) {
        auto const p = particles(i);
        auto original = (i + batch) % capacity;
        if (p.history_id != original+1 || p.parent_history_id != original+100 ||
            p.step_id != (i >= capacity-batch ? 6 : 5) || p.generation != 8 ||
            p.pid != 11 || p.medium_id != 93 || p.weight != 1. || p.energy_GeV != 0.02)
          ++count;
      }, errors);
    require(errors == 0, "Queue compaction changed particle/RNG identities or weighted energy");
    auto host = Kokkos::create_mirror_view(records);
    host(0) = {}; host(0).outcome = api::EmOutcome::Children; host(0).child_count = 3;
    for (unsigned c = 0; c < 3; ++c) {
      auto& p = host(0).children[c];
      p.pid = 11; p.medium_id = 93; p.energy_GeV = .01;
      p.direction[0] = 1.; p.history_id = capacity+c+1;
    }
    Kokkos::deep_copy(execution, records, host);
    bool rejected = false;
    try { queue.commitControlled(records, 1, binding, execution); }
    catch (std::length_error const&) { rejected = true; }
    require(rejected && queue.size() == capacity, "Overflow must retain the full input queue");
    Queue::Particles first("admission_first", 1);
    queue.copyPrefix(first, 1, execution);
    auto first_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, first);
    require(first_host(0).history_id == batch+1 && first_host(0).step_id == 5,
            "Overflow consumed its input");
    api::em::EmThinningConfig thinning;
    thinning.enabled = 1; thinning.threshold_GeV = 100.; thinning.maximum_weight = .5;
    auto old = api::em::applyEmThinning(thinning, 100., 1., 40., 60., .2, .3);
    require(old.status == api::em::EmThinningStatus::NotApplied, "Old maximum-weight guard differs");
    thinning.maximum_weight = 50.;
    auto current = api::em::applyEmThinning(thinning, 100., 1., 40., 60., .2, .3);
    require(current.status == api::em::EmThinningStatus::Hillas && current.keep_mask == 1 &&
            current.first_weight == 2.5 && current.second_weight == 0., "Requested thinning did not activate");
    std::cout << "{\"passed\":true,\"execution_space\":\"" << Space::name()
      << "\",\"threads\":256,\"capacity\":" << capacity
      << ",\"particle_bytes\":" << sizeof(Particle)
      << ",\"queue_workspace_bytes\":" << Queue::workspaceBytes(capacity,batch)
      << ",\"peak_queue_workspace_bytes\":" << Queue::peakWorkspaceBytes(capacity,batch)
      << ",\"all_particle_states_checked\":" << capacity
      << ",\"overflow_preserves_input\":true,\"thinning_100GeV_Wmax50_activates\":true}\n";
    return 0;
  } catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
}
