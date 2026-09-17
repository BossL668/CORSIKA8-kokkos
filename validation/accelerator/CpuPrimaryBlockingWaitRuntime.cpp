// Standalone CUDA stream/lifecycle tests; no simulation, installed target or table.
#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/accelerator/em/kokkos/CudaCompletionTicket.hpp>
#include <corsika/accelerator/em/kokkos/ResidentExecutionWait.hpp>
#include <corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace acc = corsika::accelerator::em;
namespace detail = acc::kokkos_detail;
namespace {
void require(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}
void cudaCheck(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
template<class F> void rejects(F&& f) {
  bool rejected=false;
  try {f();} catch (std::logic_error const&) {rejected=true;}
  require(rejected,"invalid event/wait transition accepted");
}
struct Stream {
  cudaStream_t value{};
  Stream() {cudaCheck(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
  ~Stream() {if(value){cudaStreamSynchronize(value);cudaStreamDestroy(value);}}
};
struct Payload {
  Kokkos::View<std::uint64_t*,Kokkos::CudaSpace> out;
  std::uint64_t iteration;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    std::uint64_t value=std::uint64_t(i)+17*iteration;
    for(int j=0;j<32;++j) value=value*6364136223846793005ULL+1442695040888963407ULL;
    out(i)=value;
  }
};
std::uint64_t expected(unsigned i,std::uint64_t iteration) {
  std::uint64_t value=i+17*iteration;
  for(int j=0;j<32;++j) value=value*6364136223846793005ULL+1442695040888963407ULL;
  return value;
}
// Outstanding work on another stream must NOT be fenced by the resident waiter.
// A bounded CUDA host callback avoids a busy-wait GPU kernel or occupancy claim.
struct HostGate {
  std::atomic<bool> entered{false}, released{false}, timed_out{false};
};
void CUDART_CB heldHostCallback(void* pointer) {
  auto& gate=*static_cast<HostGate*>(pointer);
  gate.entered.store(true,std::memory_order_release);
  auto const deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
  while(!gate.released.load(std::memory_order_acquire)) {
    if(std::chrono::steady_clock::now()>deadline) {
      gate.timed_out.store(true,std::memory_order_release); break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
struct GateRelease {
  HostGate& gate;
  cudaStream_t stream;
  ~GateRelease() {
    gate.released.store(true,std::memory_order_release);
    cudaStreamSynchronize(stream);
  }
};
using Particle = corsika::gpu::em::EmParticleState;
bool exactDouble(double const a, double const b) {
  // Compare all represented bits, including signed zero. Do not compare object
  // padding: the AoS/SoA round trip only promises the declared particle fields.
  std::uint64_t first{},second{};
  static_assert(sizeof(first)==sizeof(a));
  std::memcpy(&first,&a,sizeof(first));
  std::memcpy(&second,&b,sizeof(second));
  return first==second;
}
void requireExactParticle(Particle const& got, Particle const& expected) {
  require(got.pid==expected.pid,"checkpoint changed pid");
  require(got.medium_id==expected.medium_id,"checkpoint changed medium");
  require(got.generation==expected.generation,"checkpoint changed generation");
  require(got.reserved==expected.reserved,"checkpoint changed reserved flags");
  require(exactDouble(got.energy_GeV,expected.energy_GeV),"checkpoint changed energy");
  for(unsigned axis=0;axis<3;++axis) {
    require(exactDouble(got.position_m[axis],expected.position_m[axis]),
            "checkpoint changed position");
    require(exactDouble(got.direction[axis],expected.direction[axis]),
            "checkpoint changed direction");
  }
  require(exactDouble(got.time_s,expected.time_s),"checkpoint changed time");
  require(exactDouble(got.weight,expected.weight),"checkpoint changed weight");
  require(got.history_id==expected.history_id,"checkpoint changed history identity");
  require(got.parent_history_id==expected.parent_history_id,
          "checkpoint changed parent identity");
  require(got.step_id==expected.step_id,"checkpoint changed step identity");
}
std::vector<Particle> checkpointParticles(std::size_t count, unsigned iteration) {
  std::vector<Particle> result(count);
  constexpr std::int32_t pids[]={22,11,-11,13,-13};
  for(std::size_t i=0;i<count;++i) {
    auto& particle=result[i];
    particle.pid=pids[i%5];
    particle.medium_id=static_cast<std::int32_t>(i%3);
    particle.generation=static_cast<std::uint32_t>(i+iteration);
    particle.reserved=0x5a000000U+static_cast<std::uint32_t>(i);
    particle.energy_GeV=0.125*(i+1)+iteration;
    particle.position_m[0]=-1024.+0.5*i;
    particle.position_m[1]=64.+0.25*iteration;
    particle.position_m[2]=i%2?-0.:0.;
    particle.direction[0]=i%2?-0.:0.;
    particle.direction[1]=0.;
    particle.direction[2]=i%2?-1.:1.;
    particle.time_s=static_cast<double>(i+iteration)*0x1p-30;
    particle.weight=0.5+0.125*(i%31);
    particle.history_id=0x1234567800000000ULL+i+4096ULL*iteration;
    particle.parent_history_id=0x2345678900000000ULL+i/3;
    particle.step_id=0x3456789000000000ULL+i+iteration;
  }
  return result;
}
struct AdvanceCheckpointPayload {
  detail::ParticleSoARawView queue;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    auto particle=queue.load(i);
    // Exact binary operations make the expected GPU work observable without
    // depending on any transport formula, table, random stream or tolerances.
    particle.energy_GeV*=2.;
    particle.position_m[0]+=0.25;
    particle.step_id+=7;
    queue.store(i,particle);
  }
};
void testCheckpointDownloads() {
  Stream owning,unrelated;
  Kokkos::Cuda execution(owning.value),wrong_execution(unrelated.value);
  detail::KokkosWavefrontQueue<Kokkos::Cuda> queue(1024,execution);
  detail::ResidentExecutionWait<Kokkos::Cuda> wait;
  wait.setBlocking(true);
  auto submit=[&](std::vector<Particle> const& input) {
    queue.upload(input,execution);
    Kokkos::parallel_for("advance_checkpoint_payload",
        Kokkos::RangePolicy<Kokkos::Cuda>(execution,0,input.size()),
        AdvanceCheckpointPayload{queue.current().rawDeviceView()});
  };
  auto check=[&](std::vector<Particle> const& got,std::vector<Particle> input) {
    require(got.size()==input.size(),"checkpoint lost/duplicated particles");
    for(std::size_t i=0;i<input.size();++i) {
      input[i].energy_GeV*=2.; input[i].position_m[0]+=0.25; input[i].step_id+=7;
      requireExactParticle(got[i],input[i]);
    }
  };
  require(queue.download(wrong_execution,&wait).empty(),"empty checkpoint returned particles");
  require(wait.blockingCalls()==0,"empty checkpoint submitted a blocking event");
  for(unsigned iteration=1;iteration<=32;++iteration) {
    auto input=checkpointParticles(17+(iteration*31)%1000,iteration);
    submit(input);
    check(queue.download(wrong_execution,&wait),input);
    require(wait.blockingCalls()==iteration,"checkpoint did not use the blocking waiter exactly once");
    // download does not consume the queue; it must leave it synchronized and
    // safe to replace with a differently sized next input, as at a new call.
    queue.clear();
  }
  wait.resetStatistics();
  require(wait.blockingEnabled() && !wait.blockingCalls(),
          "new-shower counter reset changed checkpoint wait selection");
  auto input=checkpointParticles(513,33);
  submit(input);
  check(queue.download(execution,&wait),input);
  require(wait.blockingCalls()==1,"checkpoint event was not reused after counter reset");
  queue.clear();

  // The original and progress-configured paths must retain the original queue
  // fence: forwarding a nonblocking waiter from either resident finale must
  // NOT introduce a cooperative callback/re-entrant scheduling point.
  detail::ResidentExecutionWait<Kokkos::Cuda> progress;
  unsigned progress_calls{};
  progress.setProgress([&] {++progress_calls;throw std::logic_error(
      "checkpoint invoked an original progress callback");return false;});
  input=checkpointParticles(257,34);
  submit(input); check(queue.download(wrong_execution,&progress),input);
  require(progress_calls==0 && progress.blockingCalls()==0,
          "nonblocking checkpoint changed progress/default wait semantics");
  queue.clear(); submit(input); check(queue.download(wrong_execution),input);
  queue.clear(); submit(input); check(queue.download(),input);
  queue.clear();

  // Warm all allocations and this kernel before holding an unrelated stream.
  // A wrong-stream/global fence would hit the bounded gate, failing this test.
  HostGate gate;
  GateRelease release{gate,unrelated.value};
  cudaCheck(cudaLaunchHostFunc(unrelated.value,heldHostCallback,&gate));
  auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
  while(!gate.entered.load(std::memory_order_acquire)) {
    require(std::chrono::steady_clock::now()<deadline,"checkpoint stream callback did not start");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto const before=wait.blockingCalls();
  require(queue.download(wrong_execution,&wait).empty(),"gated empty checkpoint returned data");
  require(wait.blockingCalls()==before,"gated empty checkpoint waited");
  submit(input);
  check(queue.download(wrong_execution,&wait),input);
  require(wait.blockingCalls()==before+1,"gated checkpoint did not use the reusable event");
  require(!gate.timed_out.load(std::memory_order_acquire) &&
          !gate.released.load(std::memory_order_acquire),
          "checkpoint fenced the argument stream instead of its owning stream");
  queue.clear();
}
void testWaits() {
  Stream stream;
  Kokkos::Cuda execution(stream.value);
  Kokkos::View<std::uint64_t*,Kokkos::CudaSpace> device("wait_payload",1024);
  Kokkos::View<std::uint64_t*,Kokkos::CudaHostPinnedSpace> host("wait_payload_host",1024);
  detail::ResidentExecutionWait<Kokkos::Cuda> wait;
  require(!wait.blockingEnabled() && !wait.blockingCalls() &&
          wait.blockingHostSeconds()==0.,"default wait changed");
  auto submit=[&](std::uint64_t iteration) {
    Kokkos::parallel_for("wait_payload",Kokkos::RangePolicy<Kokkos::Cuda>(execution,0,1024),
                         Payload{device,iteration});
    Kokkos::deep_copy(execution,host,device);
  };
  auto check=[&](std::uint64_t iteration) {
    for(unsigned i=0;i<1024;++i) require(host(i)==expected(i,iteration),"stream output mismatch");
  };
  // Default fence, reusable blocking event, and legacy callback/polling all
  // consume the same named kernel/copy sequence and must produce exact bytes.
  for(unsigned mode=0;mode<3;++mode) {
    if(mode==1) {
      wait.setProgress([]{return false;});
      rejects([&]{wait.setBlocking(true);});
      wait.setProgress({}); wait.setBlocking(true);
      rejects([&]{wait.setProgress([]{return false;});});
    } else if(mode==2) {
      wait.setBlocking(false); wait.setProgress([]{return false;});
    }
    wait.resetStatistics();
    for(std::uint64_t iteration=1;iteration<=32;++iteration) {
      submit(iteration); wait.wait(execution,"test resident stream wait"); check(iteration);
    }
    require(wait.blockingCalls()==(mode==1?32:0),"wrong completed blocking wait count");
    if(mode==1) {
      require(wait.blockingEnabled() && wait.blockingHostSeconds()>=0.,"blocking mode lost");
      wait.resetStatistics();
      require(wait.blockingEnabled() && !wait.blockingCalls() &&
              wait.blockingHostSeconds()==0.,"reset destroyed mode or retained counters");
      submit(33); wait.wait(execution,"reuse after counter reset"); check(33);
      require(wait.blockingCalls()==1,"reusable blocking event did not survive reset");
    }
  }
  wait.setProgress({}); wait.setBlocking(true);
  {
    // All allocations precede the unrelated stream gate; only explicit stream
    // submission and event waits occur while that callback is held.
    HostGate gate;
    Stream unrelated;
    GateRelease release{gate,unrelated.value};
    cudaCheck(cudaLaunchHostFunc(unrelated.value,heldHostCallback,&gate));
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(!gate.entered.load(std::memory_order_acquire)) {
      require(std::chrono::steady_clock::now()<deadline,"unrelated stream callback did not start");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    submit(34); wait.wait(execution,"must wait only selected CUDA stream"); check(34);
    require(!gate.timed_out.load(std::memory_order_acquire) &&
            !gate.released.load(std::memory_order_acquire),"resident wait drained an unrelated stream");
  }
  detail::ResidentExecutionWait<Kokkos::OpenMP> cpu_wait;
  require(!cpu_wait.blockingEnabled(),"OpenMP default changed");
  rejects([&]{cpu_wait.setBlocking(true);});
  cpu_wait.setBlocking(false);
  // Raw ticket validation, including reuse on the real non-default stream.
  detail::CudaCompletionTicket normal;
  rejects([&]{normal.consumeBlocking();});
  submit(35); normal.record(execution.cuda_stream());
  rejects([&]{normal.consumeBlocking();});
  require(normal.pending(),"rejected blocking consume discarded ordinary ticket");
  while(!normal.ready()) std::this_thread::yield();
  normal.consume(); check(35);
  detail::CudaCompletionTicket blocking(true);
  rejects([&]{blocking.consumeBlocking();});
  for(std::uint64_t iteration=36;iteration<68;++iteration) {
    submit(iteration); blocking.record(execution.cuda_stream());
    require(blocking.pending(),"blocking ticket not pending after record");
    rejects([&]{blocking.record(execution.cuda_stream());});
    blocking.consumeBlocking(); check(iteration);
    require(!blocking.pending(),"blocking ticket not consumed");
    rejects([&]{blocking.consumeBlocking();});
  }
  require(blocking.drainNoThrow()==cudaSuccess,"idle ticket cleanup failed");
}
} // namespace
int main() {
  try {
    acc::KokkosRuntimeConfig config;
    config.execution_backend="cuda"; config.cooperative_owner=true; config.threads=2;
    {
      acc::KokkosRuntime owner(config);
      testWaits();
      testCheckpointDownloads();
    }
    require(Kokkos::is_finalized(),"runtime did not finalize");
    std::cout<<"{\"complete\":true,\"default_fence_replays\":32,"
      "\"blocking_event_replays\":32,\"callback_polling_replays\":32,"
      "\"raw_blocking_ticket_replays\":32,\"counter_reset_reuse\":true,"
      "\"callback_conflicts_rejected\":true,\"other_stream_not_fenced\":true,"
      "\"checkpoint_download_replays\":32,\"checkpoint_all_particle_fields_exact\":true,"
      "\"checkpoint_empty_no_event\":true,\"checkpoint_counter_reset_reuse\":true,"
      "\"checkpoint_progress_callback_not_invoked\":true,"
      "\"checkpoint_owning_stream_not_argument_stream\":true,"
      "\"scope\":\"real-stream primitive and lifecycle gates, not physics acceptance\"}\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
