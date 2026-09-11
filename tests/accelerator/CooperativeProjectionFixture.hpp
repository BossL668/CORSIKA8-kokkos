#pragma once
// Prescribed-track fixtures shared by the two isolated cooperative tests.
corsika::gpu::radio::GpuRadioConfig radioConfig() {
  corsika::gpu::radio::GpuRadioConfig c;
  c.enabled=true;c.deterministic=true;c.track_diagnostics=true;
  c.propagation.homogeneous_refractive_index=1.0003;
  for(int i=0;i<3;++i) {
    corsika::gpu::radio::RadioObserverSnapshot o;
    o.position_m[0]=15.+i*5.;o.position_m[1]=10.;o.position_m[2]=-10.;
    o.start_time_s=40.e-9+i*1.e-9;o.sample_rate_Hz=1.e9;
    o.number_of_bins=513+i*2;o.duration_s=(o.number_of_bins-1)/o.sample_rate_Hz;
    c.coreas_observers.push_back(o);c.zhs_observers.push_back(o);
  }
  return c;
}
em::GpuEmConfig::ProfileProjection profileConfig() {
  em::GpuEmConfig::ProfileProjection p;
  p.enabled=true;p.accumulate_on_device=true;p.axis_direction[0]=1.;
  p.axis_step_length_m=1.;p.axis_grammage_g_per_cm2={0.,1.,2.,3.,4.,5.};
  p.output_bin_count=16;p.output_bin_width_g_per_cm2=.5;
  p.fixed_point_weight_limit=1.e6;p.fixed_point_energy_limit_GeV=1.e6;
  return p;
}
std::vector<em::LeptonTransportRecord> tracks(std::size_t n) {
  std::vector<em::LeptonTransportRecord> t(n);
  for(std::size_t i=0;i<n;++i){
    auto& r=t[i];r.start.pid=(i%3==0?-11:11);r.start.weight=1.;
    r.start.energy_GeV=1.;r.start.direction[0]=1.;r.start.history_id=i+1;
    r.start.position_m[0]=.05*(i%7);r.start.position_m[1]=.01*(i%5);
    r.start.time_s=(i%21)*.1e-9;
    r.end=r.start;r.end.energy_GeV=.999;r.end.position_m[0]+=1.;
    r.end.time_s+=1./(299792458.*std::sqrt(1.-em::ElectronMassGeV*em::ElectronMassGeV));
    r.end.step_id++;r.input_index=i;r.distance_m=1.;
    r.traversed_grammage_g_per_cm2=.1;r.continuous_deposited_energy_GeV=.001;
    r.start_density_g_per_cm3=.001;r.end_density_g_per_cm3=.001;
  }
  return t;
}
template<class Ex> auto uploadTracks(std::vector<em::LeptonTransportRecord> const& t,Ex const& ex) {
  Kokkos::View<em::LeptonTransportRecord*,typename Ex::memory_space> v("merge_tracks",t.size());
  auto h=Kokkos::create_mirror(v);
  for(std::size_t i=0;i<t.size();++i)h(i)=t[i];
  Kokkos::deep_copy(ex,v,h);ex.fence("stage merge tracks");return v;
}
template<class Ex> struct Accumulators {
  Ex ex{};
  rk::KokkosRadioAccumulator<Ex> radio;
  kd::KokkosProfileAccumulator<Ex> profile;
  Kokkos::View<double*,typename Ex::memory_space> depth{"merge_depth",6};
  em::detail::DeviceProfileProjection projection{};
  Accumulators() {
    radio.initialize(radioConfig(),{},ex);
    auto pc=profileConfig();profile.initialize(pc,ex);
    auto h=Kokkos::create_mirror(depth);
    for(int i=0;i<6;++i)h(i)=i;
    Kokkos::deep_copy(ex,depth,h);ex.fence("stage profile axis");
    projection.axis_direction[0]=1.;projection.axis_step_length_m=1.;
    projection.axis_grammage_g_per_cm2=depth.data();projection.axis_support_count=6;
  }
  void accumulate(std::vector<em::LeptonTransportRecord> const& t) {
    auto v=uploadTracks(t,ex);
    radio.accumulateLeptonTracks(v,t.size(),ex);
    profile.accumulateLepton(projection,v,t.size(),{},0,ex);
    ex.fence("prescribed tracks completed");
  }
};

