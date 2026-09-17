#include <corsika/modules/transport/kokkos/KokkosInterfaceQueue.hpp>
#include <corsika/modules/transport/kokkos/ExecutionSpace.hpp>
#include <iostream>
#include <vector>

namespace api=corsika::interfaces;
using Space=api::kokkos::ExecutionSpace;
using Queue=api::kokkos::InterfaceQueue<Space>;
using Particle=api::em::EmParticleState;
void check(bool ok,char const* why){if(!ok)throw std::runtime_error(why);}
template<class F> void reject(F&& f) {
  bool failed=false;try{f();}catch(std::exception const&){failed=true;}
  check(failed,"invalid queue operation accepted");
}
Particle particle(std::uint64_t h) {
  Particle p;p.pid=h%3==0?22:h%3==1?11:-11;p.history_id=h;
  p.parent_history_id=h+10000;p.medium_id=h%2?17:93;p.generation=8;
  p.energy_GeV=.02;p.direction[0]=1.;p.time_s=1.e-9;p.weight=2.;p.step_id=5;
  return p;
}
int main() {
  try {
    Kokkos::ScopeGuard runtime;
    Space execution;
    Queue queue(8,3,execution);
    Queue::Particles buffer("test_interface_staging",8);
    auto host=Kokkos::create_mirror_view(buffer);
    auto inject=[&](std::vector<std::uint64_t> ids) {
      for(std::size_t i=0;i<ids.size();++i)host(i)=particle(ids[i]);
      Kokkos::deep_copy(execution,buffer,host);
      queue.append(buffer,ids.size(),execution);execution.fence();
    };
    auto expect=[&](std::vector<std::uint64_t> ids) {
      check(queue.size()==ids.size(),"wrong FIFO size");
      queue.copyPrefix(buffer,ids.size(),execution);
      Kokkos::deep_copy(execution,host,buffer);execution.fence();
      for(std::size_t i=0;i<ids.size();++i) {
        auto p=particle(ids[i]);
        check(host(i).history_id==p.history_id&&host(i).parent_history_id==p.parent_history_id&&
              host(i).medium_id==p.medium_id&&host(i).pid==p.pid&&host(i).step_id==p.step_id&&
              host(i).weight==p.weight&&host(i).time_s==p.time_s,
              "FIFO identity/material/species was reordered or corrupted");
      }
      check(queue.capacity()==8,"resident allocation grew");
    };
    Queue::Records records("test_interface_records",3);
    auto r=Kokkos::create_mirror_view(records);
    auto commit=[&](std::size_t count) {
      Kokkos::deep_copy(execution,records,r);
      auto n=queue.commit(records,count,execution);execution.fence();return n;
    };
    inject({10,20,30});
    r(0)={};r(0).outcome=api::EmOutcome::Continuation;r(0).end=particle(10);
    r(1)={};r(1).outcome=api::EmOutcome::Children;r(1).child_count=3;
    for(int c=0;c<3;++c)r(1).children[c]=particle(201+c);
    r(1).children[2].weight=0.;
    r(2)={};r(2).outcome=api::EmOutcome::Fallback;
    check(commit(3)==3,"zero-weight/fallback admission differs");expect({10,201,202});
    inject({99});expect({10,201,202,99});
    r(1).child_count=2;r(1).children[0]=particle(2011);r(1).children[1]=particle(2012);
    check(commit(2)==3,"mixed prefix successor count");expect({202,99,10,2011,2012});
    r(0)={};r(0).outcome=api::EmOutcome::Children;r(0).child_count=3;
    for(int c=0;c<3;++c)r(0).children[c]=particle(301+c);
    check(commit(1)==3,"three-child append");expect({99,10,2011,2012,301,302,303});
    reject([&]{commit(1);}); // Seven - one + three exceeds eight.
    expect({99,10,2011,2012,301,302,303});
    reject([&]{inject({88,89});});expect({99,10,2011,2012,301,302,303});
    r(0).outcome=api::EmOutcome::Cut;check(commit(1)==0,"cut generated a survivor");
    std::vector<std::uint64_t> order{10,2011,2012,301,302,303};expect(order);
    // Repeated FIFO rotations force many device compactions with a bounded
    // allocation, and must preserve the entire particle identity each time.
    for(int round=0;round<200;++round) {
      r(0)={};r(0).outcome=api::EmOutcome::Continuation;r(0).end=particle(order.front());
      auto first=order.front();order.erase(order.begin());order.push_back(first);
      check(commit(1)==1,"continuation lost");expect(order);
    }
    while(queue.size()) {
      r(0)={};r(0).outcome=api::EmOutcome::Escape;commit(1);
    }
    expect({});inject({777});expect({777});
    // The multi-wavefront path validates on device and downloads one POD.
    api::MaterialInterface binding{17,93,0,1};
    auto controlled=[&](std::size_t count) {
      Kokkos::deep_copy(execution,records,r);
      auto result=queue.commitControlled(records,count,binding,execution);
      execution.fence();return result;
    };
    inject({888,889,890,891,892,893,894});
    r(0)={};r(0).outcome=api::EmOutcome::Children;r(0).child_count=3;
    for(int c=0;c<3;++c)r(0).children[c]=particle(901+c);
    reject([&]{controlled(1);});expect({777,888,889,890,891,892,893,894});
    r(0).children[0].medium_id=-9;
    reject([&]{controlled(1);});expect({777,888,889,890,891,892,893,894});
    r(0)={};r(0).outcome=api::EmOutcome::Fallback;
    auto control=controlled(1);
    check(control.fallbacks==1&&control.successors==0&&!control.error,"device fallback control differs");
    expect({888,889,890,891,892,893,894});
    std::cout<<"resident queue PASS execution="<<Space::name()
             <<" stable rotations=200 capacity=8 overflow retains input\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
