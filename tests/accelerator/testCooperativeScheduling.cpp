#include <corsika/accelerator/em/detail/CooperativeScheduling.hpp>
#include <corsika/accelerator/em/detail/CheckedFixedAccumulatorMerge.hpp>
#include <iostream>
#include <functional>
#include <vector>
using namespace corsika::accelerator::em::detail;
void require(bool p) { if (!p) throw std::runtime_error("cooperative test failure"); }
void rejects(std::function<void()> f) { bool threw=false;try { f(); }catch(std::exception const&){threw=true;}require(threw); }
int main() {
  CooperativeBatchState s;
  rejects([&]{s.submit();}); s.stage(1); rejects([&]{s.stage(2);}); require(s.submit()==1);
  s.stage(2); rejects([&]{s.submit();}); rejects([&]{s.commit(1);});
  s.waiting();s.controlReady(false);s.waiting();s.controlReady(true);
  rejects([&]{s.commit(2);});s.commit(1);rejects([&]{s.commit(1);});
  require(s.submit()==2);s.waiting();s.controlReady(true);s.commit(2);
  require(!s.pending());rejects([&]{s.stage(2);});
  s.fail();rejects([&]{s.stage(3);});rejects([&]{s.submit();});
  CooperativeLoadBalancer b;
  auto choice=b.choose(8192,false,false,8192,2048);
  require(choice.endpoint==CooperativeEndpoint::Cuda&&choice.particles==8192);
  choice=b.choose(1000,true,false,8192,2048);
  require(choice.endpoint==CooperativeEndpoint::OpenMP&&choice.particles==256);
  require(b.choose(1000,true,true,8192,2048).particles==0);
  b.observe(CooperativeEndpoint::OpenMP,2048,1.);require(b.cpuBatch()==2048);
  require(b.choose(1000,true,false,8192,100).particles==100);
  rejects([&]{b.observe(CooperativeEndpoint::Cuda,0,1.);});
  rejects([&]{b.observe(CooperativeEndpoint::Cuda,1,0.);});
  rejects([&]{b.observe(CooperativeEndpoint::Cuda,2048,std::numeric_limits<double>::denorm_min());});
  rejects([&]{b.observe(static_cast<CooperativeEndpoint>(2),1,1.);});
  auto r=checkedCooperativeHistoryRange(123,5);require(r.first==123&&r.limit==128);
  rejects([&]{checkedCooperativeHistoryRange(UINT64_MAX,1);});
  rejects([&]{checkedCooperativeHistoryRange(0,1);});
  CooperativeHistoryAllocator ids(123);
  require(ids.reserve(5).limit == 128);
  require(ids.reserve(7).first == 128);
  auto const saved_next = ids.next();
  rejects([&]{ids.reserve(UINT64_MAX);});require(ids.next()==saved_next);
  FixedAccumulatorLayout layout{"test-grid",1e-12,3};
  std::vector<std::int64_t> a{1,INT64_MAX,-4}, c{-1,0,7};
  mergeFixedAccumulators(a,layout,c,layout);require(a==std::vector<std::int64_t>({0,INT64_MAX,3}));
  auto saved=a;c={1,1,1};rejects([&]{mergeFixedAccumulators(a,layout,c,layout);});require(a==saved);
  a={0,INT64_MIN,0};saved=a;c={1,-1,1};rejects([&]{mergeFixedAccumulators(a,layout,c,layout);});require(a==saved);
  auto other=layout;other.identity="different-time-grid";rejects([&]{mergeFixedAccumulators(a,layout,c,other);});
  other=layout;other.units_per_integer*=2;rejects([&]{mergeFixedAccumulators(a,layout,c,other);});
  std::cout<<"PASS: bounded state, exactly-once commits, load balancing, ranges, fixed merge\n";
}
