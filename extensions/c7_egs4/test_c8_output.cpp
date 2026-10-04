#include "Egs4C8Session.hpp"
#include <corsika/accelerator/em/common/CorsikaOutputSink.hpp>
#include <corsika/accelerator/em/common/EnvironmentSnapshotBuilder.hpp>
#include <corsika/accelerator/radio/common/RadioSnapshotBuilder.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/modules/ObservationPlane.hpp>
#include <corsika/modules/writers/EnergyLossWriter.hpp>
#include <corsika/modules/writers/LongitudinalWriter.hpp>
#include <corsika/modules/writers/ProductionWriter.hpp>
#include <corsika/modules/radio/CoREAS.hpp>
#include <corsika/modules/radio/ZHS.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <corsika/modules/radio/propagators/TabulatedFlatAtmospherePropagator.hpp>
#include <boost/filesystem.hpp>
#include <parquet/file_reader.h>
#include <parquet/column_reader.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

namespace {
using namespace corsika;
namespace eg=c7_egs4;
namespace app=c7_egs4::application;
void check(bool b,char const* message){if(!b)throw std::runtime_error(message);}
struct UnusedWriter {};
struct Result {
  std::uint64_t steps{},radio_tracks{},observations{},children{};
  double deposited_GeV{},observed_available_GeV{},discarded_GeV{},initial_GeV{};
  std::vector<double> coreas,zhs,profile,deposition;
};
template<class Reader,class Scalar>
void appendColumn(std::shared_ptr<parquet::ColumnReader> const& source,std::vector<double>& values) {
  auto column=std::static_pointer_cast<Reader>(source);
  while(column->HasNext()) {
    Scalar batch[1024];std::int64_t count=0;
    auto rows=column->ReadBatch(1024,nullptr,nullptr,batch,&count);
    check(rows==count&&count>0,"Unexpected null/truncated output column");
    for(std::int64_t i=0;i<count;++i){check(std::isfinite(batch[i]),"Non-finite serialized output");values.push_back(batch[i]);}
  }
}
std::vector<double> readColumn(boost::filesystem::path const& path,char const* name) {
  auto file=parquet::ParquetFileReader::OpenFile(path.string(),false);auto metadata=file->metadata();
  int index=-1;
  for(int i=0;i<metadata->num_columns();++i)if(metadata->schema()->Column(i)->name()==name)index=i;
  check(index>=0,"Missing output column");
  auto type=metadata->schema()->Column(index)->physical_type();
  check(type==parquet::Type::DOUBLE||type==parquet::Type::FLOAT,"Non-floating output column");
  std::vector<double> values;
  for(int group=0;group<metadata->num_row_groups();++group) {
    auto column=file->RowGroup(group)->Column(index);
    if(type==parquet::Type::DOUBLE)appendColumn<parquet::DoubleReader,double>(column,values);
    else appendColumn<parquet::FloatReader,float>(column,values);
  }
  check(values.size()==std::size_t(metadata->num_rows())&&!values.empty(),"Parquet row-count mismatch");return values;
}
std::vector<double> readRadio(boost::filesystem::path const& path) {
  std::vector<double> values;
  for(auto const* component:{"Ex","Ey","Ez"}) {
    auto data=readColumn(path,component);check(data.size()==60000,"Wrong three-antenna radio output length");
    values.insert(values.end(),data.begin(),data.end());
  }
  check(std::inner_product(values.begin(),values.end(),values.begin(),0.)>0.,"Serialized radio field is zero");return values;
}
std::vector<double> waveform(ObserverCollection<TimeDomainObserver> const& detector) {
  std::vector<double> result;double norm=0.;
  for(auto const& observer:detector.getObservers())for(auto const* component:
      {&observer.getWaveformX(),&observer.getWaveformY(),&observer.getWaveformZ()})
    for(double value:*component) {
      check(std::isfinite(value),"Non-finite radio waveform");result.push_back(value);norm+=value*value;
    }
  check(norm>0.,"Radio output is empty/zero");return result;
}
Result run(app::Session& session,bool host_reference,boost::filesystem::path const& root,bool device_radio=false,bool device_profile=false) {
  using MediumInterface=IRefractiveIndexModel<IMediumModel>;
  Environment<MediumInterface> environment;auto cs=environment.getCoordinateSystem();
  auto R=constants::EarthRadius::Mean;Point earth{cs,0_m,0_m,0_m},sea{cs,0_m,0_m,R};
  create_5layer_atmosphere<MediumInterface,GladstoneDaleRefractiveIndex>(
      environment,AtmosphereId::LinsleyUSStd,earth,1.0003,sea);
  auto snapshot=gpu::em::makeCorsika7AtmosphereSnapshot(AtmosphereId::LinsleyUSStd,
      {0.,0.,0.},3,(R+1100_m)/1_m,{2.e-5,1.e-5,-4.e-5},.2);
  // The actual C8 builder, not a hand-populated test density schema.
  check(snapshot.number_of_layers==5,"Actual C8 atmosphere import failed");
  Point top{cs,0_m,0_m,R+112799_m},ground{cs,0_m,0_m,R+1100_m};
  ShowerAxis axis(top,ground-top,environment,false,3000);
  EnergyLossWriter energy(axis,1_g/square(1_cm),0_g/square(1_cm));
  LongitudinalWriter profile(axis,1_g/square(1_cm),ProfileCrossingMode::Both);
  ObservationPlane<void> observation(Plane{ground,DirectionVector{cs,{0.,0.,1.}}},DirectionVector{cs,{1.,0.,0.}});
  ObserverCollection<TimeDomainObserver> coreas_detector,zhs_detector;
  for(double x:{50.,100.,200.}) {
    TimeDomainObserver observer("antenna_"+std::to_string(int(x)),Point{cs,x*1_m,0_m,R+1100_m},
        cs,0_ns,100000_ns,2.e8_Hz,0_ns);
    coreas_detector.addObserver(observer);zhs_detector.addObserver(observer);
  }
  auto propagator=make_tabulated_flat_atmosphere_radio_propagator(environment,top,ground,100_m);
  CoREAS<decltype(coreas_detector),decltype(propagator)> coreas(coreas_detector,propagator);
  ZHS<decltype(zhs_detector),decltype(propagator)> zhs(zhs_detector,propagator);
  for(auto const* name:{"energy","profile","particles","CoREAS","ZHS"})boost::filesystem::create_directories(root/name);
  energy.startOfLibrary(root/"energy");profile.startOfLibrary(root/"profile");observation.startOfLibrary(root/"particles");
  coreas.startOfLibrary(root/"CoREAS");zhs.startOfLibrary(root/"ZHS");
  energy.startOfShower(0);profile.startOfShower(0);observation.startOfShower(0);
  coreas.startOfShower(0);zhs.startOfShower(0);
  ProductionWriter production(axis,1_g/square(1_cm));UnusedWriter interaction;
  gpu::em::CorsikaOutputSink sink(cs,energy,profile,production,observation,interaction,coreas,zhs,true,device_radio);
  double mass=get_mass(Code::Electron)/1_MeV;
  app::Configuration configuration;configuration.environment=snapshot;configuration.earth_radius_m=R/1_m;
  configuration.electron_mass_MeV=mass;configuration.seed=731;configuration.shower_id=129;
  configuration.reuse_queue_workspace=device_profile;
  if(device_profile) {
    auto& p=configuration.profile;p.enabled=p.accumulate_on_device=true;
    p.crossing_mode=profile.getCrossingMode();p.output_bin_count=energy.GetNBins();
    p.output_bin_width_g_per_cm2=1.;p.energy_loss_threshold_g_per_cm2=0.; // test zero threshold + endpoint deposit
    p.fixed_point_weight_limit=1.e6;p.fixed_point_energy_limit_GeV=10.;
    auto start=axis.getStart().getCoordinates(cs);auto direction=axis.getDirection().getComponents(cs);
    for(int j=0;j<3;++j){p.axis_start_position_m[j]=start[j]/1_m;p.axis_direction[j]=direction[j].magnitude();}
    p.axis_step_length_m=axis.getSteplength()/1_m;
    for(auto x:axis.getGrammageSupport())p.axis_grammage_g_per_cm2.push_back(x/(1_g/square(1_cm)));
  }
  if(device_radio)configuration.radio=gpu::radio::makeGpuRadioConfig(environment,top,ground,100_m,coreas_detector,zhs_detector);
  std::vector<gpu::em::EmParticleState> input;
  Result result;
  for(int pdg:{11,-11,22})for(double height:{1100.01,4000.1}) {
    gpu::em::EmParticleState p;p.pid=pdg;p.energy_GeV=.1;p.medium_id=3;p.weight=1.25;p.history_id=input.size()+1;
    p.position_m[0]=10.;p.position_m[1]=20.;p.position_m[2]=R/1_m+height;
    p.direction[0]=.1;p.direction[1]=.05;p.direction[2]=-std::sqrt(.9875);input.push_back(p);
    result.initial_GeV+=p.weight*(p.energy_GeV+(pdg==11?-mass:pdg==-11?mass:0.)*.001);
  }
  configuration.first_child_id=input.size()+1;
  app::OutputCallbacks callbacks;
  unsigned profile_callbacks=0;
  callbacks.profile=[&](auto const& r){++profile_callbacks;sink.onGpuProfile(r);};
  callbacks.step=[&](auto const& r){check(!device_profile,"Resident profile called CPU step consumer");sink.onStep(r);};callbacks.radio=[&](auto const& r){
    check(!device_radio,"Device mode incorrectly called scalar radio");sink.onRadioTrack(r);};
  unsigned waveform_callbacks=0;
  callbacks.radio_waveforms=[&](auto const& waveforms,std::uint64_t tracks){
    ++waveform_callbacks;sink.onGpuRadioWaveforms(waveforms,tracks);};
  callbacks.observation=[&](auto const& r){sink.onObservation(r);auto p=r.particle;
    result.observed_available_GeV+=p.weight*(p.energy_GeV+(p.pid==11?-mass:p.pid==-11?mass:0.)*.001);};
  callbacks.discarded=[&](double energy,unsigned){result.discarded_GeV+=energy;};
  auto native=session.run(configuration,input,callbacks,host_reference);result.children=native.children;
  if(device_profile) {
    check(profile_callbacks==1&&native.profile_steps==native.steps&&native.profile_steps>0,
      "Resident profile was missing/duplicated");
    check(!native.profile_fixed_point_overflows&&!native.profile_invalid_records&&
      native.profile_execution_space==native.execution_space,"Invalid device profile execution");
    check(native.host_output_records<native.steps,"Profile mode still downloaded all output records");
    std::cout<<"resident_profile_steps="<<native.profile_steps<<" host_records="<<native.host_output_records<<'\n';
  }
  if(device_radio) {
    check(waveform_callbacks==1&&native.radio.lepton_tracks==native.radio_tracks&&native.radio_tracks>0,
      "Radio segments/one-shot waveform export not exercised");
    check(native.radio.coreas_contributions&&native.radio.zhs_contributions&&!native.radio.fixed_point_overflows,
      "GPU radio is empty or overflowed");
    check(native.radio_execution_space==native.execution_space,"Radio did not use native execution space");
    if(native.execution_space=="Cuda")check(native.radio_kernel_ms>0.,"Missing CUDA event execution evidence");
    std::cout<<"radio_execution="<<native.radio_execution_space<<" radio_cuda_ms="<<native.radio_kernel_ms
      <<" radio_pairs="<<native.radio.track_observer_pairs<<" coreas_contributions="<<native.radio.coreas_contributions<<'\n';
  }
  // Check raw accumulators before endOfShower resets them. ZHS is still
  // vector potential here; the field comparison below reads the final file.
  waveform(coreas_detector);waveform(zhs_detector);
  auto stats=sink.statistics();result.steps=stats.steps;result.radio_tracks=stats.radio_tracks;result.observations=stats.observations;
  result.deposited_GeV=energy.getEnergyLost()/1_GeV;
  check(result.steps&&result.radio_tracks&&result.observations&&result.children,"Missing real C8 output path coverage");
  check(std::abs(result.deposited_GeV-stats.weighted_deposited_energy_GeV)<1.e-10,"C8 writer lost or duplicated deposition");
  check(std::abs(result.initial_GeV-result.deposited_GeV-result.observed_available_GeV-result.discarded_GeV)<1.e-9,"Real-output energy ledger mismatch");
  check(observation.statistics().particles==stats.observations,"Observation writer mismatch");
  energy.endOfShower(0);profile.endOfShower(0);observation.endOfShower(0);coreas.endOfShower(0);zhs.endOfShower(0);
  energy.endOfLibrary();profile.endOfLibrary();observation.endOfLibrary();coreas.endOfLibrary();zhs.endOfLibrary();
  std::size_t parquet_count=0;
  for(boost::filesystem::recursive_directory_iterator i(root),end;i!=end;++i)
    if(boost::filesystem::is_regular_file(i->path())&&i->path().extension()==".parquet") {
      check(boost::filesystem::file_size(i->path())>0,"Empty C8 parquet output");++parquet_count;
    }
  check(parquet_count>=5,"C8 output files not written");
  result.deposition=readColumn(root/"energy"/"dEdX.parquet","total");
  double storage_rounding=std::abs(std::accumulate(result.deposition.begin(),result.deposition.end(),0.)-result.deposited_GeV);
  // The standard EnergyLossWriter serializes non-negative bins as float32;
  // the independent double-precision, in-memory ledger above is unchanged.
  check(storage_rounding<=std::numeric_limits<float>::epsilon()*result.deposited_GeV,
    "Serialized deposition differs from energy ledger");
  for(auto const* name:{"photon","electron","positron"}) {
    auto data=readColumn(root/"profile"/"profile.parquet",name);
    check(data.size()==profile.getNBins()&&std::accumulate(data.begin(),data.end(),0.)>0.,"Empty EM profile component");
    result.profile.insert(result.profile.end(),data.begin(),data.end());
  }
  auto observed=readColumn(root/"particles"/"particles.parquet","kinetic_energy");
  check(observed.size()==result.observations,"Serialized observation count mismatch");
  result.coreas=readRadio(root/"CoREAS"/"observers.parquet");
  result.zhs=readRadio(root/"ZHS"/"observers.parquet");
  std::cout<<"execution_space="<<native.execution_space<<" waves="<<native.waves<<" children="<<result.children<<" steps="<<result.steps
    <<" radio_tracks="<<result.radio_tracks<<" observations="<<result.observations<<" deposited_GeV="<<result.deposited_GeV
    <<" observed_available_GeV="<<result.observed_available_GeV<<" discarded_GeV="<<result.discarded_GeV
    <<" output="<<root<<" parquet_files="<<parquet_count<<" deposition_storage_rounding_GeV="<<storage_rounding<<'\n';return result;
}
double relativeL2(std::vector<double> const& a,std::vector<double> const& b) {
  check(a.size()==b.size(),"Radio size mismatch");double d=0.,n=0.;
  for(std::size_t i=0;i<a.size();++i){d+=(a[i]-b[i])*(a[i]-b[i]);n+=a[i]*a[i];}
  return std::sqrt(d/n);
}
}
int main(int argc,char** argv) {
  try {
    if(argc!=2)return 2;corsika::logging::set_level(corsika::logging::level::err);
    char directory[]="/tmp/c7-egs4-c8-output-XXXXXX";auto* path=mkdtemp(directory);check(path,"Cannot create owned output directory");
    app::Session session(argv[1]);std::cout<<std::setprecision(17);
    auto host=run(session,true,boost::filesystem::path(path)/"host");
    auto device=run(session,false,boost::filesystem::path(path)/"device");
    auto device_radio=run(session,false,boost::filesystem::path(path)/"device_radio",true);
    auto resident=run(session,false,boost::filesystem::path(path)/"resident",true,true);
    check(resident.steps==device_radio.steps&&resident.children==device_radio.children&&
      resident.observations==device_radio.observations&&resident.coreas==device_radio.coreas&&resident.zhs==device_radio.zhs,
      "Resident profile/workspace changed transport or radio");
    std::cout<<"resident_profile_relative_L2="<<relativeL2(device_radio.profile,resident.profile)
      <<" resident_deposition_relative_L2="<<relativeL2(device_radio.deposition,resident.deposition)<<'\n';
    check(relativeL2(device_radio.profile,resident.profile)<1.e-10&&relativeL2(device_radio.deposition,resident.deposition)<1.e-8,
      "Resident profile/deposition disagrees with CPU writer");
    check(device_radio.profile==device.profile&&device_radio.deposition==device.deposition&&
      device_radio.steps==device.steps&&device_radio.radio_tracks==device.radio_tracks,"GPU radio changed transported shower");
    double same_c=relativeL2(device.coreas,device_radio.coreas),same_z=relativeL2(device.zhs,device_radio.zhs);
    std::cout<<"same_track_coreas_device_radio_relative_L2="<<same_c<<"\nsame_track_zhs_device_radio_relative_L2="<<same_z<<'\n';
    check(same_c<1.e-5&&same_z<1.e-5,"GPU projection differs from scalar same-track reference");
    check(host.steps==device.steps&&host.children==device.children&&host.observations==device.observations,"Real-output control mismatch");
    double c=relativeL2(host.coreas,device.coreas),z=relativeL2(host.zhs,device.zhs);
    double p=relativeL2(host.profile,device.profile),d=relativeL2(host.deposition,device.deposition);
    std::cout<<"coreas_host_device_relative_L2="<<c<<"\nzhs_host_device_relative_L2="<<z
      <<"\nprofile_host_device_relative_L2="<<p<<"\ndeposition_host_device_relative_L2="<<d<<'\n';
    check(c<1.e-5&&z<1.e-5&&p<1.e-10&&d<1.e-8,"Real C8 output host/device mismatch");
    std::cout<<"scope=native C++ EGS4 EM cascades through actual C8 atmosphere/profile/deposition/observation/CoREAS/ZHS; no thinning, hadronic shower, rare-channel integration or speed claim\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
