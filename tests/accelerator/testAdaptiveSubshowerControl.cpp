#include <corsika/accelerator/em/detail/AdaptiveSubshowerControl.hpp>
#include <iostream>
#include <stdexcept>
using corsika::accelerator::em::detail::AdaptiveSubshowerControl;
void require(bool value,char const* message) {
  if(!value) throw std::runtime_error(message);
}
int main() {
  try {
    for(unsigned fast:{0U,1U}) {
      AdaptiveSubshowerControl c;
      require(c.waves(0,1,65536)==64 && c.waves(1,1,65536)==8,
              "cold probes are not bounded useful batches");
      for(unsigned k:{0U,1U}) for(unsigned e:{0U,1U}) {
        std::size_t requested=1;
        for(int i=0;i<12;++i) {
          c.observe(e,k,4096,requested,requested,4096*requested,
                    (e==fast?.01:5.)*requested);
          requested=c.waves(e,k,4096);
        }
      }
      require(c.unitCost(fast,1)*100<c.unitCost(1-fast,1),
              "throughput estimate biased by endpoint identity");
      require(c.waves(fast,1,4096)==1024,"fast endpoint restricted by its identity");
      require(c.waves(1-fast,1,4096)<64,"slow endpoint lease not adapted");
      require(c.waves(fast,0,4096)<=16,"photon history lease limit broken");
      auto before=c.sample(fast,1).observations;
      c.observe(fast,1,4096,0,8,4096,10.);
      require(c.sample(fast,1).observations==before,"allocation refusal polluted estimate");
      require(c.waves(fast,1,100000000)<c.waves(fast,1,4096),
              "lease prediction ignores input size");
    }
    AdaptiveSubshowerControl service;
    require(service.inputLimit(1,1,65536,20.)==2048,"cold CPU quantum fills full arena");
    service.observe(1,1,2048,8,8,2048*8,20.,2048,20.);
    auto rate=service.sample(1,1).records_per_ms;
    for(int i=0;i<100;++i)service.observe(1,1,16,1,1,16,1000.,2048,20.);
    require(service.inputLimit(1,1,65536,20.)==2048,"sparse tails shrank useful input width");
    require(service.sample(1,1).records_per_ms==rate,"sparse fixed overhead polluted full-batch throughput");
    require(service.sample(1,1).full_observations==1,"tails counted as full-batch calibration");
    // Shrink wave count first; width changes only after two oversize FULL
    // one-wave calls. A single outlier and a long multiwave call are not enough.
    service.observe(1,1,2048,8,8,2048*8,1000.,2048,20.);
    require(service.inputLimit(1,1,65536,20.)==2048,"multiwave cost shrank width before wave lease");
    for(int i=0;i<2;++i)service.observe(1,1,2048,1,1,2048,100.,2048,20.);
    auto budget=service.inputLimit(1,1,65536,20.);
    require(budget==1024,"oversized full single-wave input not halved");
    require(service.waves(1,1,budget,20.)<=8,"CPU time budget ignored by wave lease");
    require(service.inputLimit(1,1,128,5.)==128,"explicit small arena cap exceeded");
    service.observeGpuDuration(20.);
    require(service.hostTargetMs(true)==20.,"fast GPU forces sub-amortization microjobs");
    for(int i=0;i<20;++i)service.observeGpuDuration(1000.);
    require(service.hostTargetMs(true)==100.,"slow GPU does not relax host quantum");
    require(service.hostTargetMs(false)==200.,"idle GPU unnecessarily limits CPU work");
    for(int i=0;i<2;++i)service.observe(1,1,budget,8,8,budget*8,10.,budget,200.);
    require(service.inputLimit(1,1,65536,200.)==2*budget,"full CPU batch cannot recover after two useful samples");
    AdaptiveSubshowerControl server;server.setHostInitialBatch(16384);
    require(server.inputLimit(1,1,65536,100.)==16384,"server cold width ignored");
    for(int i=0;i<2;++i)server.observe(1,1,16384,8,8,131072,10.,16384,100.);
    require(server.inputLimit(1,1,65536,100.)==32768,"fast server width permanently capped");
    require(server.inputLimit(1,1,4096,100.)==4096,"explicit arena cap exceeded");
    AdaptiveSubshowerControl arena;
    // Real GPU photon runs often never fill a 400k arena. A useful 16k batch
    // must still calibrate, and later one-particle tails must not overwrite it.
    arena.observe(0,0,16384,4,16,65536,20.,457784,200.);
    require(arena.sample(0,0).full_observations==1,"arena mistaken for useful calibration width");
    auto calibrated_rate=arena.sample(0,0).records_per_ms;
    for(int i=0;i<1000;++i)arena.observe(0,0,1,1,16,1,50.,457784,200.);
    require(arena.sample(0,0).records_per_ms==calibrated_rate,"small tails polluted GPU throughput");
    // Swap measured device speeds: useful quantum must swap as well. An
    // independent CUDA driver must not reduce the host lease to its former
    // synchronous receipt-poll interval.
    for(unsigned fast:{0U,1U}) {
      AdaptiveSubshowerControl symmetric;
      for(unsigned e:{0U,1U})
        symmetric.observe(e,1,4096,8,8,32768,e==fast?10.:100.,65536,200.);
      require(symmetric.targetMs(fast,1,false)>10*symmetric.targetMs(1-fast,1,false),
              "work quantum still biased toward GPU");
      for(unsigned e:{0U,1U})
        require(symmetric.targetMs(e,1,true)==symmetric.targetMs(e,1,false),
                "in-flight flag fragments independent useful work");
      require(symmetric.targetMs(fast,1,true)>500.,"fast endpoint cannot amortize work");
    }
    AdaptiveSubshowerControl independent;
    // A laptop-like 60:40 measured work ratio gives a 160 ms CPU quantum,
    // not 10 ms after observing a short CUDA call. This checks the causal
    // controller change, not a wall-clock speed-up claim.
    independent.observe(0,1,4096,8,8,32768,20.,4096,200.);
    independent.observe(1,1,4096,8,8,32768,30.,4096,200.);
    independent.observeGpuDuration(4.);
    require(std::abs(independent.targetMs(1,1,true)-160.)<1.e-10,
            "autonomous driver still imposes the synchronous 10 ms CPU cap");
    require(independent.waves(1,1,4096,independent.targetMs(1,1,true))==16,
            "independent CPU wave lease did not grow within hysteresis");
    AdaptiveSubshowerControl short_species;short_species.setHostInitialBatch(16384);
    for(int i=0;i<2;++i)short_species.observe(1,0,16384,2,16,32768,2.,16384,100.);
    require(short_species.inputLimit(1,0,65536,100.)==32768,
            "early species completion prevents useful full-width CPU growth");
    std::cout<<"PASS: symmetric fast-CPU/fast-GPU estimates, cold probes, native wave ceilings, invalid observations\n";
  } catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
}
