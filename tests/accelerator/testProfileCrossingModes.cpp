// Run the same observer policy on CUDA and OpenMP; no shower/RNG required.
#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileAccumulator.hpp>
#include <iostream>
#include <stdexcept>
#include <limits>
namespace kd=corsika::accelerator::em::kokkos_detail;
namespace phy=corsika::accelerator::em::detail;
namespace em=corsika::gpu::em;
using Mode=corsika::ProfileCrossingMode;
struct Segment { double a,b,w; int pid; };
template<class Ex> struct Count {
  Kokkos::View<Segment*,typename Ex::memory_space> input;
  em::detail::DeviceProfileAccumulator view;
  KOKKOS_FUNCTION void operator()(std::size_t i) const {
    auto const s=input(i);
    phy::accumulateParticleProfile<kd::KokkosProfileAtomicOperations>(view,s.pid,s.a,s.b,s.w);
    // Count mode must not filter the deposit from a backward segment.
    phy::accumulateEnergyProfile<kd::KokkosProfileAtomicOperations>(view,view.energy_loss,12.,3.,1.);
  }
};
template<class Ex> void run() {
  constexpr std::size_t n=10000,bins=32;
  Kokkos::View<Segment*,typename Ex::memory_space> input("segments",n);
  auto host=Kokkos::create_mirror_view(input);
  int pids[]={22,11,-11,13,-13};
  for(std::size_t i=0;i<n;++i)host(i)={double((i*7)%160)*.25-4.,double((i*13)%160)*.25-4.,.25*(1+i%4),pids[i%5]};
  Kokkos::deep_copy(input,host);
  em::GpuEmConfig::ProfileProjection config{};
  config.enabled=config.accumulate_on_device=true;config.output_bin_count=bins;
  config.output_bin_width_g_per_cm2=1.;config.fixed_point_weight_limit=1.e8;
  config.fixed_point_energy_limit_GeV=1.e8;
  for(auto mode:{Mode::Both,Mode::Forward,Mode::OriginalC8}) {
    config.crossing_mode=mode;kd::KokkosProfileAccumulator<Ex> acc;
    acc.initialize(config,Ex{});
    auto view=acc.deviceView();
    if(view.crossing_mode!=mode)throw std::runtime_error("mode not propagated to device");
    Kokkos::parallel_for("profile_modes",Kokkos::RangePolicy<Ex>(0,n),Count<Ex>{input,view});
    Ex{}.fence();auto result=acc.download(Ex{});
    std::vector<double> actual[]={result.photons,result.electrons,result.positrons,result.muons_minus,result.muons_plus};
    double expected[5][bins]{};
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<bins;++j) {
      auto s=host(i);bool hit=mode==Mode::Both ? std::min(s.a,s.b)<j && j<=std::max(s.a,s.b)
          : s.b>s.a && j<=s.b && (mode==Mode::OriginalC8 ? s.a<=j : s.a<j);
      if(hit)expected[i%5][j]+=s.w;
    }
    for(int k=0;k<5;++k)for(std::size_t j=0;j<bins;++j)
      if(std::abs(actual[k][j]-expected[k][j])>1.e-6)throw std::runtime_error("plane oracle mismatch");
    double deposit=0.;for(auto e:result.energy_loss_GeV)deposit+=e;
    if(std::abs(deposit-n)>1.e-6)throw std::runtime_error("count policy changed deposits");
    auto other=config;other.crossing_mode=mode==Mode::Both?Mode::Forward:Mode::Both;
    if(phy::sameProfileConfig(config,other))throw std::runtime_error("cross-mode merge accepted");
    for(double bad:{std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
      if(corsika::detail::longitudinalProfileBins(bad,1.,bins,mode).end ||
         corsika::detail::longitudinalProfileBins(1.,bad,bins,mode).end)
        throw std::runtime_error("nonfinite endpoint accepted");
    std::cout<<Ex::name()<<" mode="<<int(mode)<<" samples="<<n<<" PASS\n";
  }
}
int main(int argc,char** argv) {
  std::string backend=argc>1?argv[1]:"default";
#ifdef KOKKOS_ENABLE_OPENMP
  if(backend=="openmp") {
    Kokkos::OpenMP::impl_initialize(Kokkos::InitializationSettings{}.set_num_threads(4));
    try{run<Kokkos::OpenMP>();}catch(...){Kokkos::OpenMP::impl_finalize();throw;}
    Kokkos::OpenMP::impl_finalize();return 0;
  }
#endif
  Kokkos::ScopeGuard guard(Kokkos::InitializationSettings{}.set_num_threads(4));
  run<Kokkos::DefaultExecutionSpace>();
}
