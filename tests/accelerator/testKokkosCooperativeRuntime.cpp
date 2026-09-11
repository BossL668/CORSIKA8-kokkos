#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/accelerator/em/kokkos/CudaCompletionTicket.hpp>
#include <corsika/accelerator/em/kokkos/CudaWavefrontControl.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#ifdef C8_COOPERATIVE_CUPTI
#include "CooperativeActivityTrace.hpp"
#endif

namespace em=corsika::accelerator::em;
void require(bool p,char const* what){if(!p)throw std::runtime_error(what);}
void cudaCheck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class Space> struct Arithmetic {
  Kokkos::View<std::uint64_t*,typename Space::memory_space> out;
  int iterations;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    std::uint64_t v=static_cast<std::uint64_t>(i)+1;
    for(int j=0;j<iterations;++j)v=v*6364136223846793005ULL+1442695040888963407ULL;
    out(i)=v;
  }
};

// Representative control shape (counts + errors). This exercises asynchronous
// transfer and reuse, not the EM kernels that will produce the production POD.
struct Control { std::uint64_t counts[9]{}; std::uint64_t interactions{}; std::uint32_t error{}; };
struct ProduceControl {
  Kokkos::View<Control,Kokkos::CudaSpace> output;
  std::uint64_t wave;
  KOKKOS_INLINE_FUNCTION void operator()(int) const {
    Control value{};
    for(unsigned i=0;i<9;++i)value.counts[i]=wave*123+i;
    value.interactions=wave;
    output()=value;
  }
};
template<class F> void rejects(F&& f) {
  bool rejected=false;try{f();}catch(std::logic_error const&){rejected=true;}
  require(rejected,"invalid asynchronous control transition was accepted");
}
void testControlReuse(Kokkos::Cuda const& execution) {
  using Transfer=em::kokkos_detail::CudaWavefrontControl<Control>;
  Transfer transfer;
  auto const address=transfer.pinnedAddress();
  Kokkos::View<Control,Kokkos::CudaSpace> device("probe_front_control");
  rejects([&]{transfer.poll();});rejects([&]{transfer.take();});
  for(std::uint64_t wave=1;wave<=32;++wave){
    Kokkos::parallel_for("produce_control",Kokkos::RangePolicy<Kokkos::Cuda>(execution,0,1),ProduceControl{device,wave});
    transfer.submit(device,execution);
    rejects([&]{transfer.submit(device,execution);});
    rejects([&]{transfer.take();}); // must explicitly observe ready first
    auto const deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(!transfer.poll()){
      require(std::chrono::steady_clock::now()<deadline,"control completion timeout");
      std::this_thread::yield();
    }
    auto const result=transfer.take();
    require(result.interactions==wave&&result.error==0,"control count/error");
    for(unsigned i=0;i<9;++i)require(result.counts[i]==wave*123+i,"control byte transfer");
    require(transfer.pinnedAddress()==address,"pinned control reallocated");
    rejects([&]{transfer.take();});
  }
  // Destroy an in-flight ticket during exceptional cleanup. Its pinned memory
  // and device view must remain alive until the transfer's event completes.
  try {
    Transfer abandoned;
    Kokkos::parallel_for("abandoned_control",Kokkos::RangePolicy<Kokkos::Cuda>(execution,0,1),ProduceControl{device,33});
    abandoned.submit(device,execution);
    throw std::runtime_error("intentional batch cancellation");
  }catch(std::runtime_error const&){}
}
int main(int argc,char** argv) {
  try {
    // Separate invocations test ordinary ownership and explicit sharing.
    bool const single=argc>1&&std::string(argv[1])=="--single";
#ifdef C8_COOPERATIVE_CUPTI
    if (!single) c8_overlap_trace::start();
#endif
    em::KokkosRuntimeConfig owner_config;owner_config.execution_backend="cuda";
    owner_config.cooperative_owner=!single;owner_config.threads=single?1:4;
    {
      em::KokkosRuntime owner(owner_config);
      if(single){
        require(owner.runQueueProbe(1024).roundtrip_exact,"single queue");
        bool rejected=false;try{owner.shareCooperativeLifetime();}catch(std::logic_error const&){rejected=true;}
        require(rejected,"single runtime shared unexpectedly");
      } else {
        em::KokkosRuntimeConfig host_config;host_config.execution_backend="openmp";
        host_config.threads=4;host_config.runtime_lease=owner.shareCooperativeLifetime();
        auto device_config=host_config;device_config.execution_backend="cuda";
        em::KokkosRuntime host(host_config),device(device_config);
        require(host.info().concurrency==4&&host.info().cooperative_runtime,"host configuration");
        require(host.runQueueProbe(1024).roundtrip_exact,"host roundtrip");
        require(device.runQueueProbe(1024).roundtrip_exact,"device roundtrip");
        bool rejected=false;auto bad=host_config;bad.threads=8;
        try{em::KokkosRuntime wrong(bad);}catch(std::invalid_argument const&){rejected=true;}
        require(rejected,"mismatched runtime lease accepted");
        std::thread wrong_thread([&]{bool no=false;try{em::KokkosRuntime wrong(host_config);}catch(std::invalid_argument const&){no=true;}require(no,"cross-thread borrow");});
        wrong_thread.join();
        Kokkos::Cuda gpu;Kokkos::OpenMP cpu;
        testControlReuse(gpu);
        Kokkos::View<std::uint64_t*,Kokkos::CudaSpace> d("overlap_probe",1);
        Kokkos::View<std::uint64_t*,Kokkos::HostSpace> h("host_probe",32768);
        // Warm up both runtimes before any timing observations.
        Kokkos::parallel_for("warm_cuda",Kokkos::RangePolicy<Kokkos::Cuda>(gpu,0,1),Arithmetic<Kokkos::Cuda>{d,1});
        gpu.fence();
        Kokkos::parallel_for("warm_openmp",Kokkos::RangePolicy<Kokkos::OpenMP>(cpu,0,32768),Arithmetic<Kokkos::OpenMP>{h,1});
        em::kokkos_detail::CudaCompletionTicket ticket;
        cudaEvent_t start,end;cudaCheck(cudaEventCreate(&start));cudaCheck(cudaEventCreate(&end));
        auto const origin=std::chrono::steady_clock::now();
        cudaCheck(cudaEventRecord(start,gpu.cuda_stream()));
        Kokkos::parallel_for("cooperative_device_arithmetic",Kokkos::RangePolicy<Kokkos::Cuda>(gpu,0,1),Arithmetic<Kokkos::Cuda>{d,3000000});
        cudaCheck(cudaEventRecord(end,gpu.cuda_stream()));ticket.record(gpu.cuda_stream());
        cudaError_t launch_status;
        while((launch_status=cudaEventQuery(start))==cudaErrorNotReady){
          require(std::chrono::steady_clock::now()-origin<std::chrono::seconds(10),"probe launch timeout");
          std::this_thread::yield();
        }
        cudaCheck(launch_status);
        auto const t0=std::chrono::steady_clock::now();bool const before=!ticket.ready();
#ifdef C8_COOPERATIVE_CUPTI
        auto const host_start_ns=c8_overlap_trace::timestamp();
#endif
        Kokkos::parallel_for("cooperative_host_arithmetic",Kokkos::RangePolicy<Kokkos::OpenMP>(cpu,0,32768),Arithmetic<Kokkos::OpenMP>{h,512});
        auto const t1=std::chrono::steady_clock::now();bool const after=!ticket.ready();
#ifdef C8_COOPERATIVE_CUPTI
        auto const host_end_ns=c8_overlap_trace::timestamp();
#endif
        while(!ticket.ready()){
          require(std::chrono::steady_clock::now()-origin<std::chrono::seconds(10),"probe completion timeout");
          std::this_thread::yield();
        }
        ticket.consume();float gpu_ms;cudaCheck(cudaEventElapsedTime(&gpu_ms,start,end));
#ifdef C8_COOPERATIVE_CUPTI
        c8_overlap_trace::finish(host_start_ns,host_end_ns);
#endif
        auto mirror=Kokkos::create_mirror_view(d);Kokkos::deep_copy(gpu,mirror,d);gpu.fence();
        std::uint64_t expected=1;for(int j=0;j<3000000;++j)expected=expected*6364136223846793005ULL+1442695040888963407ULL;
        require(mirror(0)==expected,"CUDA arithmetic mismatch");
        for(int i=0;i<32768;++i){std::uint64_t v=i+1;for(int j=0;j<512;++j)v=v*6364136223846793005ULL+1442695040888963407ULL;require(h(i)==v,"OpenMP arithmetic mismatch");}
        std::cout<<"{\"host_threads\":4,\"gpu_event_ms\":"<<gpu_ms
                 <<",\"host_start_ms\":"<<std::chrono::duration<double,std::milli>(t0-origin).count()
                 <<",\"host_end_ms\":"<<std::chrono::duration<double,std::milli>(t1-origin).count()
                 <<",\"gpu_pending_before_host\":"<<before<<",\"gpu_pending_after_host\":"<<after
                 <<",\"arithmetic_exact\":true,\"control_reuse_cycles\":32,\"scope\":\"primitive event-bracket probe, not an EM shower benchmark\"}\n";
        cudaCheck(cudaEventDestroy(start));cudaCheck(cudaEventDestroy(end));
      }
    }
    require(Kokkos::is_finalized(),"runtime was not finalized after last lease");
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
