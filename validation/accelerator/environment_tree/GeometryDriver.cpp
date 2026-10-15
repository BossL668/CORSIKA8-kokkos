#include "GeometryDriver.hpp"
#include <corsika/accelerator/em/kokkos/KokkosEnvironmentTree.hpp>
#include <corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp>
#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>
#include <corsika/accelerator/em/detail/LeptonTransportStep.hpp>
#include <chrono>
using Execution=Kokkos::DefaultExecutionSpace;
namespace et=corsika::gpu::em::environment_tree;
void initializeDevice(int argc,char** argv) { Kokkos::initialize(argc,argv); }
void finalizeDevice() { Kokkos::finalize(); }
char const* deviceName() { return Execution::name(); }
DeviceResult runDeviceGeometry(std::vector<et::Node> const& nodes,
    std::vector<et::NodeId> const& children,std::vector<et::NodeId> const& exclusions,
    std::vector<Query> const& queries) {
  using Clock=std::chrono::steady_clock;
  auto ms=[](Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();};
  DeviceResult result;
  struct Host {std::vector<et::Node> const& nodes;std::vector<et::NodeId> const& children;std::vector<et::NodeId> const& exclusions;};
  auto start=Clock::now();
  corsika::accelerator::em::kokkos_detail::EnvironmentTreeOwner<Execution> owner(Host{nodes,children,exclusions});
  auto tree=owner.view();
  Kokkos::View<Query*,Execution> input("tree inputs",queries.size());
  Kokkos::View<Result*,Execution> output("tree outputs",queries.size());
  auto hi=Kokkos::create_mirror_view(input);
  for (unsigned i=0;i<queries.size();++i) hi(i)=queries[i];
  Kokkos::deep_copy(input,hi);
  Execution().fence();result.upload_ms=ms(start);
  auto launch=[&]() {
    Kokkos::parallel_for("native environment tree search",Kokkos::RangePolicy<Execution>(0,queries.size()),
      KOKKOS_LAMBDA(int i) {
        auto q=input(i);et::StraightAccess access{};
        static_cast<et::TreeView&>(access)=tree;
        for (int a=0;a<3;++a) {access.position[a]=q.p[a];access.velocity[a]=q.v[a];}
        auto r=corsika::geometry_detail::nextEnvironmentBoundary(access,q.node,q.limit);
        output(i)={r.distance,r.node,et::locate(tree,q.p)};
      });
    Execution().fence();
  };
  launch();
  for (int rep=0;rep<7;++rep) { start=Clock::now();launch();result.kernel_ms.push_back(ms(start)); }
  start=Clock::now();
  auto ho=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),output);
  result.results.resize(queries.size());
  for (unsigned i=0;i<queries.size();++i) result.results[i]=ho(i);
  result.download_ms=ms(start);
  return result;
}

std::vector<ModelResult> runDeviceModels(std::vector<et::Model> const& models,
                                       std::vector<ModelQuery> const& queries,et::FullView full) {
  corsika::accelerator::em::kokkos_detail::FullEnvironmentOwner<Execution> owner(full);
  auto full_device=owner.view();
  Kokkos::View<et::Model*,Execution> m("node models",models.size());
  Kokkos::View<ModelQuery*,Execution> q("model queries",queries.size());
  Kokkos::View<ModelResult*,Execution> r("model outputs",queries.size());
  auto hm=Kokkos::create_mirror_view(m);auto hq=Kokkos::create_mirror_view(q);
  for (unsigned i=0;i<models.size();++i) hm(i)=models[i];
  for (unsigned i=0;i<queries.size();++i) hq(i)=queries[i];
  Kokkos::deep_copy(m,hm);Kokkos::deep_copy(q,hq);
  Kokkos::parallel_for("exported C8 environment models",Kokkos::RangePolicy<Execution>(0,queries.size()),
    KOKKOS_LAMBDA(int i) {
      auto const query=q(i);auto const model=full_device.model_count?
          full_device.models[query.id]:m(query.id);
      r(i)={et::density(model,query.p),et::refractiveIndex(model,query.p),
            et::grammage(model,query.p,query.direction,query.length),
            et::arclength(model,query.p,query.direction,query.grammage)};
    });
  auto hr=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),r);
  std::vector<ModelResult> result(queries.size());
  for (unsigned i=0;i<queries.size();++i) result[i]=hr(i);
  return result;
}

bool testNodeQueue() {
  namespace kd=corsika::accelerator::em::kokkos_detail;
  using Particle=corsika::gpu::em::EmParticleState;
  constexpr unsigned count=257;
  kd::ParticleSoA<typename Execution::memory_space> queue("logical node test",count);
  Kokkos::View<Particle*,Execution> output("logical node roundtrip",count);
  auto raw=queue.rawDeviceView();
  Kokkos::parallel_for("store logical node",Kokkos::RangePolicy<Execution>(0,count),KOKKOS_LAMBDA(int i) {
    Particle p;p.pid=11;p.medium_id=17;p.environment_node=i-3;
    p.history_id=i+1000;p.energy_GeV=i+1.;p.reserved=0x1234;
    queue.store(i,p);auto r=raw.load(i);raw.store(i,r);output(i)=queue.load(i);
  });
  auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),output);
  for (unsigned i=0;i<count;++i) {
    auto const p=host(i);
    if (p.environment_node!=static_cast<int>(i)-3 || p.medium_id!=17 ||
        p.energy_GeV!=i+1. || p.history_id!=i+1000 || p.reserved!=0x1234) return false;
  }
  return queue.deviceBytes()>=queue.deviceBytesForCapacity(count);
}

std::vector<PhotonSequenceResult> runPhotonSequences(et::FullView tree,
    std::vector<corsika::gpu::em::EmParticleState> const& particles) {
  namespace em=corsika::gpu::em;
  corsika::accelerator::em::kokkos_detail::FullEnvironmentOwner<Execution> owner(tree);
  em::EnvironmentSnapshot environment;environment.geometry=em::EnvironmentGeometry::NodeTree;
  environment.node_tree=owner.view();
  Kokkos::View<em::EmParticleState*,Execution> input("photon boundary inputs",particles.size());
  Kokkos::View<PhotonSequenceResult*,Execution> output("photon boundary sequences",particles.size());
  auto host=Kokkos::create_mirror_view(input);
  for (unsigned i=0;i<particles.size();++i) host(i)=particles[i];
  Kokkos::deep_copy(input,host);
  Kokkos::parallel_for("photon full-tree transport",Kokkos::RangePolicy<Execution>(0,particles.size()),
    KOKKOS_LAMBDA(int i) {
      em::EmInteractionRecord interaction;interaction.particle=input(i);
      interaction.status=em::EmInteractionStatus::NoDiscreteInteraction;
      em::ExternalTransportBoundary external;external.disable_observation=true;
      for (unsigned step=0;step<16;++step) {
        auto r=corsika::accelerator::em::detail::transportPhoton(environment,interaction,external);
        if (r.fallback_flag) {output(i).fallback=1;break;}
        output(i).records[step]=r.record;output(i).count=step+1;
        if (r.record.limit!=em::PhotonTransportLimit::LayerBoundary) break;
        interaction.particle=r.record.end;
      }
    });
  auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),output);
  std::vector<PhotonSequenceResult> out(particles.size());
  for (unsigned i=0;i<particles.size();++i) out[i]=result(i);
  return out;
}

std::vector<LeptonSequenceResult> runLeptonSequences(et::FullView tree,
    std::vector<corsika::gpu::em::EmParticleState> const& particles) {
  namespace em=corsika::gpu::em;
  corsika::accelerator::em::kokkos_detail::FullEnvironmentOwner<Execution> owner(tree);
  em::EnvironmentSnapshot environment;environment.geometry=em::EnvironmentGeometry::NodeTree;
  environment.node_tree=owner.view();
  // Deliberately wrong global field: only the current node's field is valid.
  environment.magnetic_field_T[0]=123.;
  Kokkos::View<em::EmParticleState*,Execution> input("lepton boundary inputs",particles.size());
  Kokkos::View<LeptonSequenceResult*,Execution> output("lepton boundary sequences",particles.size());
  auto host=Kokkos::create_mirror_view(input);
  for (unsigned i=0;i<particles.size();++i) host(i)=particles[i];
  Kokkos::deep_copy(input,host);
  Kokkos::parallel_for("lepton full-tree transport",Kokkos::RangePolicy<Execution>(0,particles.size()),
    KOKKOS_LAMBDA(int i) {
      em::EmInteractionRecord interaction;interaction.particle=input(i);
      interaction.status=em::EmInteractionStatus::NoDiscreteInteraction;
      // A synthetic linear range isolates transport/node semantics. It is NOT
      // a replacement or an accuracy oracle for the physical PROPOSAL tables.
      double mass=em::transportMassGeV(interaction.particle.pid)*1000.;
      em::tables::NativeUtilityColumn utility{};utility.pdg_id=interaction.particle.pid;
      utility.particle_mass_MeV=mass;utility.lower_energy_limit_MeV=mass;
      utility.spline.axis={em::tables::NativeAxisType::Linear,2,mass,1.e6,1.e6-mass};
      utility.spline.coefficient_count=2;
      double values[]{0.,(1.e6-mass)*1000.};
      double derivatives[]{values[1],values[1]};
      em::tables::NativePhysicsView table{};table.em_transport_cut_MeV=.5;table.muon_transport_cut_MeV=.5;
      table.proposal_native.utility_columns=&utility;table.proposal_native.utility_column_count=1;
      table.proposal_native.cubic_values=values;table.proposal_native.cubic_node_derivatives=derivatives;
      table.proposal_native.cubic_coefficient_count=2;
      em::ExternalTransportBoundary external;external.disable_observation=true;
      for (unsigned step=0;step<32;++step) {
        auto& record=output(i).records[step];
        auto status=corsika::accelerator::em::detail::transportLepton(
            table,false,environment,interaction,record,output(i).failure,external);
        if (status) {output(i).fallback=status;break;}
        output(i).count=step+1;
        if (record.limit!=em::LeptonTransportLimit::LayerBoundary &&
            record.limit!=em::LeptonTransportLimit::MagneticStep &&
            record.limit!=em::LeptonTransportLimit::ContinuousStep) break;
        interaction.particle=record.end;
      }
    });
  auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),output);
  std::vector<LeptonSequenceResult> out(particles.size());
  for (unsigned i=0;i<particles.size();++i) out[i]=result(i);
  return out;
}

CurvedDeviceResult runDeviceCurved(std::vector<et::Node> const& nodes,
    std::vector<et::NodeId> const& children,std::vector<et::NodeId> const& exclusions,
    std::vector<CurvedQuery> const& queries) {
  struct Host {std::vector<et::Node> const& nodes;std::vector<et::NodeId> const& children;std::vector<et::NodeId> const& exclusions;};
  corsika::accelerator::em::kokkos_detail::EnvironmentTreeOwner<Execution> owner(Host{nodes,children,exclusions});
  auto tree=owner.view();
  Kokkos::View<CurvedQuery*,Execution> input("curved inputs",queries.size());
  Kokkos::View<CurvedResult*,Execution> output("curved outputs",queries.size());
  auto hi=Kokkos::create_mirror_view(input);
  for (unsigned i=0;i<queries.size();++i) hi(i)=queries[i];
  Kokkos::deep_copy(input,hi);
  auto launch=[&]() {
    Kokkos::parallel_for("native curved environment search",Kokkos::RangePolicy<Execution>(0,queries.size()),
      KOKKOS_LAMBDA(int i) {
        auto q=input(i);auto access=q.track;
        static_cast<et::TreeView&>(access)=tree;
        auto r=corsika::geometry_detail::nextEnvironmentBoundary(access,q.node,q.limit);
        output(i)={{r.distance,r.node,et::locate(tree,access.position)},access.unsupported_shape,access.incomplete_roots};
      });
    Execution().fence();
  };
  CurvedDeviceResult result;launch();
  for (int rep=0;rep<7;++rep) {
    auto t=std::chrono::steady_clock::now();launch();
    result.kernel_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
  }
  auto ho=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),output);
  for (unsigned i=0;i<queries.size();++i) result.results.push_back(ho(i));
  return result;
}
