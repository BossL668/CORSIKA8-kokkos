/* Fixed-ledger tests: use production projection on prescribed charged tracks. */
#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileAccumulator.hpp>
#include <corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp>
#include <iostream>
namespace em=corsika::gpu::em;
namespace kd=corsika::accelerator::em::kokkos_detail;
namespace rd=corsika::accelerator::radio::detail;
namespace rk=corsika::accelerator::radio::kokkos_detail;
namespace ed=corsika::accelerator::em::detail;
void require(bool v,char const* s){if(!v)throw std::runtime_error(s);}
template<class F> void rejects(F f){bool ok=false;try{f();}catch(std::exception const&){ok=true;}require(ok,"invalid ledger accepted");}
#include "CooperativeProjectionFixture.hpp"
void equalWaveforms(corsika::gpu::radio::GpuRadioWaveforms const& a,
                    corsika::gpu::radio::GpuRadioWaveforms const& b) {
  for(auto pair:{std::make_pair(&a.coreas,&b.coreas),std::make_pair(&a.zhs,&b.zhs)}) {
    require(pair.first->size()==pair.second->size(),"waveform observer count");
    for(std::size_t i=0;i<pair.first->size();++i){
      auto const& x=(*pair.first)[i];auto const& y=(*pair.second)[i];
      require(x.x==y.x && x.y==y.y && x.z==y.z,"fixed decode changes waveform");
    }
  }
}
template<class Ex> void verify(std::size_t n) {
  auto t=tracks(n);auto middle=t.begin()+n/2;
  Accumulators<Ex> reference, a, b, floating;
  reference.accumulate(t);floating.accumulate(t);
  a.accumulate({t.begin(),middle});b.accumulate({middle,t.end()});
  auto all=reference.radio.downloadFixed("event",ed::CooperativeEndpoint::Cuda,reference.ex);
  auto nonzero=[](auto const& v){return std::any_of(v.begin(),v.end(),[](auto x){return x!=0;});};
  require(nonzero(all.coreas) && nonzero(all.zhs),"radio fixture must exercise nonzero signals");
  auto one=a.radio.downloadFixed("event",ed::CooperativeEndpoint::Cuda,a.ex);
  auto two=b.radio.downloadFixed("event",ed::CooperativeEndpoint::OpenMP,b.ex);
  rd::CooperativeRadioMerge merge("event",radioConfig());
  rejects([&]{merge.take();});
  merge.commit(one);rejects([&]{merge.commit(one);});
  auto bad=two;bad.config.zhs_observers[0].sample_rate_Hz*=2.;
  rejects([&]{merge.commit(bad);});
  bad=two;bad.shower_identity="other";rejects([&]{merge.commit(bad);});
  merge.commit(two);auto sum=merge.take();rejects([&]{merge.take();});
  require(sum.coreas==all.coreas && sum.zhs==all.zhs,"split radio integers differ");
  require(sum.counters.valid_tracks==all.counters.valid_tracks &&
          sum.counters.coreas_contributions==all.counters.coreas_contributions &&
          sum.counters.zhs_contributions==all.counters.zhs_contributions &&
          sum.counters.zhs_subtracks==all.counters.zhs_subtracks,"radio counters differ");
  equalWaveforms(rd::decodeFixedRadio(sum),floating.radio.download(floating.ex));
  rejects([&]{a.radio.downloadFixed("event",ed::CooperativeEndpoint::Cuda,a.ex);});
  rejects([&]{a.accumulate(t);});
  auto pa=a.profile.downloadFixed("event",ed::CooperativeEndpoint::Cuda,a.ex);
  auto pb=b.profile.downloadFixed("event",ed::CooperativeEndpoint::OpenMP,b.ex);
  auto pr=reference.profile.downloadFixed("event",ed::CooperativeEndpoint::Cuda,reference.ex);
  ed::CooperativeProfileMerge pm("event",profileConfig());pm.commit(pa);pm.commit(pb);
  auto ps=pm.take();
  require(ps.histograms==pr.histograms,"split profile integers differ");
  auto decoded=ed::decodeFixedProfile(ps);auto old=floating.profile.download(floating.ex);
  require(decoded.electrons==old.electrons && decoded.positrons==old.positrons &&
          decoded.energy_loss_GeV==old.energy_loss_GeV &&
          decoded.weighted_deposited_energy_GeV==old.weighted_deposited_energy_GeV &&
          decoded.steps==old.steps,"profile decode differs");
  // Transaction rollback if only the SECOND (ZHS) array overflows.
  auto huge=one;huge.zhs[0]=std::numeric_limits<std::int64_t>::max();
  auto positive=two;positive.zhs[0]=1;
  rd::CooperativeRadioMerge transaction("event",radioConfig());transaction.commit(huge);
  rejects([&]{transaction.commit(positive);});
  positive.zhs[0]=0;transaction.commit(positive);
  auto recovered=transaction.take();
  require(recovered.zhs[0]==std::numeric_limits<std::int64_t>::max(),"partial failed commit");
  auto negative=one;negative.coreas[0]=std::numeric_limits<std::int64_t>::min();
  auto decrement=two;decrement.coreas[0]=-1;
  rd::CooperativeRadioMerge underflow("event",radioConfig());underflow.commit(negative);
  rejects([&]{underflow.commit(decrement);});
  decrement.coreas[0]=0;underflow.commit(decrement);
  require(underflow.take().coreas[0]==std::numeric_limits<std::int64_t>::min(),"negative overflow rollback");
  auto counter_overflow=two;
  counter_overflow.counters.valid_tracks=std::numeric_limits<unsigned long long>::max();
  rd::CooperativeRadioMerge counter_check("event",radioConfig());counter_check.commit(one);
  rejects([&]{counter_check.commit(counter_overflow);});
  counter_check.commit(two);require(counter_check.take().coreas==sum.coreas,"counter rollback");
  auto max_a=one,max_b=two;
  max_a.counters.maximum_segment_length_m=1.e308;
  max_b.counters.maximum_segment_length_m=1.e308;
  rd::CooperativeRadioMerge maximum_check("event",radioConfig());
  maximum_check.commit(max_a);maximum_check.commit(max_b);
  require(maximum_check.take().counters.maximum_segment_length_m==1.e308,
          "diagnostic maxima were incorrectly added");
  auto different_scale=pb;different_scale.energy_units*=2.;
  ed::CooperativeProfileMerge scale_check("event",profileConfig());scale_check.commit(pa);
  rejects([&]{scale_check.commit(different_scale);});
  scale_check.commit(pb);require(scale_check.take().histograms==ps.histograms,"scale rollback");
  auto profile_bad=pb;profile_bad.counters.steps=std::numeric_limits<unsigned long long>::max();
  ed::CooperativeProfileMerge pt("event",profileConfig());pt.commit(pa);
  if(pa.counters.steps) rejects([&]{pt.commit(profile_bad);});
  pt.commit(pb);require(pt.take().histograms==ps.histograms,"profile failed commit was partial");
  auto bytes=a.radio.deviceBytes()+a.profile.deviceBytes();
  for(int repeat=0;repeat<32;++repeat){
    a.radio.reset(a.ex);a.profile.reset(1.e6,1.e6,a.ex);
    a.accumulate({t.begin(),middle});
    auto ra=a.radio.downloadFixed("repeat",ed::CooperativeEndpoint::Cuda,a.ex);
    auto rp=a.profile.downloadFixed("repeat",ed::CooperativeEndpoint::Cuda,a.ex);
    require(ra.coreas==one.coreas && ra.zhs==one.zhs && rp.histograms==pa.histograms,"reset leaked prior shower");
    require(a.radio.deviceBytes()+a.profile.deviceBytes()==bytes,"accumulator memory grew");
  }
  std::cout<<"endpoint="<<Ex::name()<<" tracks="<<n
           <<" split_integer_equal=true legacy_float_equal=true reuse=32 bytes="<<bytes<<"\n"<<std::flush;
}
int main() {
  try {
    corsika::accelerator::em::KokkosRuntimeConfig cfg;
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
    cfg.execution_backend="cuda";cfg.threads=4;cfg.cooperative_owner=true;
#elif defined(KOKKOS_ENABLE_CUDA)
    cfg.execution_backend="cuda";
#else
    cfg.execution_backend="openmp";cfg.threads=4;
#endif
    corsika::accelerator::em::KokkosRuntime runtime(cfg);
#if defined(KOKKOS_ENABLE_CUDA)
    for(auto n:{257u,16385u})verify<Kokkos::Cuda>(n);
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
    for(auto n:{257u,16385u})verify<Kokkos::OpenMP>(n);
#endif
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<"\n";return 1;}
}
