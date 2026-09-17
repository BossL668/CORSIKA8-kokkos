// PSR-only analytic CPU/device geometry and resident terminal-inventory tests.
#include <corsika/geometry/terrain/KokkosDemCoverage.hpp>
#include <corsika/modules/transport/kokkos/KokkosInterfaceQueue.hpp>
#include <corsika/modules/transport/kokkos/ExecutionSpace.hpp>
#include <corsika/modules/radio/interface/KokkosPropagation.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
namespace cov=corsika::terrain::coverage;
namespace flat=corsika::terrain::flat;
namespace api=corsika::interfaces;
using Space=api::kokkos::ExecutionSpace;
corsika::terrain::FlatTerrainData loadRadioBvhProbeMesh(char const*);
void check(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
corsika::terrain::FlatTerrainData square() {
  corsika::terrain::FlatTerrainData m;m.vertices={{-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0}};
  m.indexed_triangles={{{0,1,2},{0,0,1}},{{0,2,3},{0,0,1}}};return m;
}
// Independent long-double Cartesian plane oracle: no BVH/production roots.
double oracle(flat::QuadraticPath const& p) {
  long double best=HUGE_VALL;
  long double o[2]{p.start.origin.x,p.start.origin.y},d[2]{p.start.direction.x,p.start.direction.y},q[2]{p.quadratic.x,p.quadratic.y};
  for(int axis=0;axis<2;++axis)for(int side:{-1,1}) {
    long double a=q[axis],b=d[axis],c=o[axis]-side,r[2];int n=0;
    if(a==0.){if(b!=0.)r[n++]=-c/b;}
    else {long double delta=b*b-4*a*c;if(delta>=0.){r[n++]=(-b-sqrtl(delta))/(2*a);r[n++]=(-b+sqrtl(delta))/(2*a);}}
    for(int k=0;k<n;++k) {
      auto s=r[k],other=o[1-axis]+s*d[1-axis]+s*s*q[1-axis];
      if(s>=0.&&s<=p.maximum_length_m&&side*(b+2*a*s)>1.e-12&&fabsl(other)<=1.+1.e-12)best=std::min(best,std::max(s,1.e-9L));
    }
  }
  return static_cast<double>(best);
}
int main(int argc,char** argv) {
  try {
    Kokkos::InitializationSettings settings;settings.set_num_threads(256);
    Kokkos::ScopeGuard runtime(settings);Space execution;
    auto data=cov::fromHeightfield(square());cov::KokkosData<Space> device(data);
    std::mt19937_64 rng(39101);std::uniform_real_distribution<double> u(-.95,.95),angle(-3.14159,3.14159);
    std::vector<flat::QuadraticPath> paths;
    for(int i=0;i<100000;++i){double a=angle(rng),dx=cos(a),dy=sin(a),q=i%3?u(rng)*.02:0.;
      paths.push_back({{{u(rng),u(rng),i%2?1.e6:-1000.},{dx,dy,0.},true},{-dy*q,dx*q,0.},10.});}
    paths.push_back({{{1,0,1.e7},{1,0,0},true},{0,0,0},HUGE_VAL});
    paths.push_back({{{1,0,0},{-1,0,0},true},{0,0,0},HUGE_VAL});
    paths.push_back({{{0,0,0},{1/std::sqrt(2.),1/std::sqrt(2.),0},true},{0,0,0},10});
    paths.push_back({{{0,0,0},{0,0,1},true},{0,0,0},HUGE_VAL});
    // Exact tangency at y=1: it touches and returns, and must not exit there.
    paths.push_back({{{0,0,0},{0,1,0},true},{0,-.25,0},3});
    Kokkos::View<flat::QuadraticPath*,Space::memory_space> input("coverage_test_paths",paths.size());
    Kokkos::View<cov::Hit*,Space::memory_space> output("coverage_test_hits",paths.size());
    auto host=Kokkos::create_mirror_view(input);for(std::size_t i=0;i<paths.size();++i)host(i)=paths[i];Kokkos::deep_copy(input,host);
    auto view=device.view();Kokkos::parallel_for("dem_boundary_analytic_queries",Kokkos::RangePolicy<Space>(execution,0,paths.size()),KOKKOS_LAMBDA(std::size_t i){output(i)=cov::nextExit(view,input(i));});
    auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
    double maximum=0.;
    for(std::size_t i=0;i<paths.size();++i) {
      auto h=cov::nextExit(data.view(),paths[i]);double expected=oracle(paths[i]);
      check(h.found()==std::isfinite(expected),"CPU analytic hit classification");
      check(h.edge==result(i).edge,"CPU/device edge ownership mismatch");
      if(h.found()) {
        maximum=std::max(maximum,std::abs(h.distance-expected));
        check(std::abs(h.distance-expected)<1.e-10,"CPU analytic distance error");
        check(std::abs(h.distance-result(i).distance)<1.e-11,"CPU/device distance error");
      }
    }
    auto l=square();l.vertices={{0,0,0},{4,0,0},{4,1,0},{1,1,0},{1,4,0},{0,4,0}};
    l.indexed_triangles={{{0,1,3},{0,0,1}},{{1,2,3},{0,0,1}},{{0,3,5},{0,0,1}},{{3,4,5},{0,0,1}}};
    auto nonconvex=cov::fromHeightfield(l);
    check(cov::contains(nonconvex.view(),{.5,3,50000}),"high atmosphere footprint");
    check(!cov::contains(nonconvex.view(),{3,3,0}),"concave notch admission");
    check(!cov::segmentInside(nonconvex.view(),{.5,3,0},{3,.5,0}),"leave/reenter optical leg accepted");
    check(cov::segmentInside(nonconvex.view(),{.5,.5,0},{3,.5,0}),"interior optical leg rejected");
    // Mixed FIFO: terminal exits retain all state, and have no successors.
    using Queue=api::kokkos::InterfaceQueue<Space>;Queue queue(16,8,execution);
    Queue::Particles particles("boundary_queue_input",8);auto ph=Kokkos::create_mirror_view(particles);
    Queue::Records records("boundary_queue_records",8);auto rh=Kokkos::create_mirror_view(records);
    long double initial=0.,escaped=0.,retained=0.,removed=0.;
    for(unsigned i=0;i<8;++i) {
      api::em::EmParticleState p;p.pid=i%3==0?22:i%3==1?11:-11;p.medium_id=i%2;
      p.history_id=100+i;p.parent_history_id=7;p.direction[0]=1.;p.energy_GeV=10+i;p.weight=1+3*i;
      ph(i)=p;initial+=p.weight*p.energy_GeV;rh(i)={};rh(i).start=p;rh(i).end=p;
      rh(i).end.energy_GeV-=.1;rh(i).end.step_id=1;rh(i).end.position_m[0]=1.;rh(i).has_track=1;rh(i).distance_m=1.;
      rh(i).outcome=i%2?api::EmOutcome::Continuation:api::EmOutcome::DomainEscape;rh(i).domain_edge=1;
      removed+=p.weight*(p.energy_GeV-rh(i).end.energy_GeV);
      (i%2?retained:escaped)+=rh(i).end.weight*rh(i).end.energy_GeV;
    }
    Kokkos::deep_copy(particles,ph);queue.append(particles,8,execution);Kokkos::deep_copy(records,rh);
    auto control=queue.commitControlled(records,8,api::MaterialInterface{0,1,0,1},execution);
    check(!control.error&&queue.size()==4&&control.successors==4,"terminal queue lost/duplicated particles");
    queue.copyPrefix(particles,4,execution);Kokkos::deep_copy(ph,particles);
    for(unsigned i=0;i<4;++i)check(ph(i).history_id==101+2*i&&ph(i).parent_history_id==7,"survivor identity/order corrupted");
    check(std::abs(initial-escaped-retained-removed)<1.e-12,"weighted queue energy closure");
    // Real production DEM perimeter and station admission; export exactly the
    // same constructed contour used by transport for the diagnostic figure.
    std::size_t real_edges=0;
    if(argc==3) {
      auto m=loadRadioBvhProbeMesh(argv[1]);
      auto real=cov::fromHeightfield(m);real_edges=real.edges.size();check(real_edges==2016,"real DEM perimeter edge count");
      for(auto p:{flat::Vec3{2516.535431729868,4294.03815963489,500.52949201660203},
                  flat::Vec3{397.88140553660566,-2028.7543838949,-145.70048570153665},
                  flat::Vec3{1677.6818310890271,-2030.0016098328992,-114.16758852890496},
                  flat::Vec3{-1160.5292565842756,537.1068996418721,-100.32429011659633}})
        check(cov::contains(real.view(),p),"production source/station outside coverage");
      std::ofstream f(argv[2]);f<<std::setprecision(17)<<"edge,x0_m,y0_m,x1_m,y1_m\n";
      for(std::size_t i=0;i<real.edges.size();++i){auto e=real.edges[i];f<<i<<','<<e.a.x<<','<<e.a.y<<','<<e.b.x<<','<<e.b.y<<'\n';}
    }
    std::cout<<std::setprecision(17)<<"{\"passed\":true,\"execution\":\""<<Space::name()<<"\",\"concurrency\":"<<execution.concurrency()
      <<",\"analytic_cases\":"<<paths.size()<<",\"maximum_distance_error_m\":"<<maximum<<",\"queue_input\":8,\"queue_exits\":4,\"queue_retained\":4,\"real_edges\":"<<real_edges<<"}\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
