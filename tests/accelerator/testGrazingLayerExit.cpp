// Regression for the Fe event 497, history 269889964 grazing the 10 km shell.
// The linear synthetic range isolates geometry; this is not a PROPOSAL oracle.
#include <corsika/accelerator/em/detail/LeptonTransportStep.hpp>
#include <cstdio>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
using namespace corsika::gpu::em;

C8_ACCELERATOR_INLINE_FUNCTION unsigned checkGrazingLayerExit() {
  unsigned errors=0;
  for(int variant=0;variant<18;++variant) {
    EmParticleState p{}; p.pid=variant%2 ? -11 : 11;
    p.energy_GeV=.0011781468091319534; p.weight=1.;
    double x[]={10922.399827625619,11062.164778047485,6380981.063260761};
    double u[]={.9848172495015158,.17358291920439897,-.0019887797133937932};
    for(int j=0;j<3;++j){p.position_m[j]=x[j];p.direction[j]=u[j];}
    // Probe neighbouring representable positions on both sides, not only
    // the decimal state which happens to round onto the shell on the host.
    int offset=variant/2-4;
    for(int j=0;j<(offset<0?-offset:offset);++j)
      p.position_m[2]=::nextafter(p.position_m[2],offset<0?0.:1.e8);
    EnvironmentSnapshot env{};
    env.number_of_layers=2;env.observation_radius_m=6371000.;
    env.observation_plane_normal[2]=1.;
    for(int j=0;j<2;++j){auto& l=env.atmosphere_layers[j];
      l.inner_radius_m=j?6381000.:6371000.;l.outer_radius_m=j?6391000.:6381000.;
      l.density_model=DensityModel::Homogeneous;l.density_parameter_a=.001;}
    env.magnetic_field_T[0]=27.232962e-6;
    env.magnetic_field_T[2]=-49.332426e-6;
    tables::NativeUtilityColumn utility{};utility.pdg_id=p.pid;
    utility.particle_mass_MeV=.5109989;utility.lower_energy_limit_MeV=.5109989;
    utility.spline.axis={tables::NativeAxisType::Linear,2,.5109989,10.,9.4890011};
    utility.spline.coefficient_count=2;
    double values[]={0.,9.4890011},derivatives[]={9.4890011,9.4890011};
    tables::NativePhysicsView view{};view.em_transport_cut_MeV=.5;
    view.proposal_native.utility_columns=&utility;view.proposal_native.utility_column_count=1;
    view.proposal_native.cubic_values=values;view.proposal_native.cubic_node_derivatives=derivatives;
    view.proposal_native.cubic_coefficient_count=2;
    EmInteractionRecord input{};input.particle=p;input.status=EmInteractionStatus::NoDiscreteInteraction;
    LeptonTransportRecord record{};ProposalFallbackEvent fallback{};
    auto status=corsika::accelerator::em::detail::transportLepton(view,false,env,input,record,fallback);
    bool good=status==0 && record.distance_m>0. && record.end.energy_GeV<p.energy_GeV &&
        record.end.time_s>p.time_s && record.end.step_id==p.step_id+1;
    good=good && (record.limit!=LeptonTransportLimit::LayerBoundary ||
                 record.end_layer_index!=record.start_layer_index);
    if(!good){++errors;printf("grazing variant=%d status=%u reason=%u distance=%.17g\n",
        variant,status,unsigned(fallback.reason),record.distance_m);}
  }
  // The generic primitive keeps first-root semantics, while layer callers
  // can explicitly require an inward or outward crossing (also for B=0).
  EmParticleState p{};p.pid=11;p.energy_GeV=1.;p.direction[0]=1.;
  p.position_m[0]=-2.;double zero[]={0.,0.,0.};
  auto first=intersectUniformMagneticSphere(p,.0005109989,-1.,zero,zero,1.,10.);
  auto entry=intersectUniformMagneticSphere(p,.0005109989,-1.,zero,zero,1.,10.,.2,-1);
  auto exit=intersectUniformMagneticSphere(p,.0005109989,-1.,zero,zero,1.,10.,.2,+1);
  errors+=first.distance_m!=1. || entry.distance_m!=1. || exit.distance_m!=3.;
  return errors;
}
#if defined(__CUDACC__)
__global__ void testDevice(unsigned* errors){*errors=checkGrazingLayerExit();}
#endif
int main(){
  unsigned errors=checkGrazingLayerExit();
  printf("host grazing_layer_exit errors=%u\n",errors);
#if defined(__CUDACC__)
  unsigned* device=nullptr;unsigned device_errors=0;
  if(cudaMalloc(&device,sizeof(unsigned))!=cudaSuccess)return 2;
  testDevice<<<1,1>>>(device);
  if(cudaDeviceSynchronize()!=cudaSuccess ||
     cudaMemcpy(&device_errors,device,sizeof(unsigned),cudaMemcpyDeviceToHost)!=cudaSuccess)return 2;
  cudaFree(device);printf("device grazing_layer_exit errors=%u\n",device_errors);errors+=device_errors;
#endif
  return errors?1:0;
}
