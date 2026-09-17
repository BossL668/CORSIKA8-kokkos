#include <corsika/modules/radio/interface/Output.hpp>
#include <unsupported/Eigen/FFT>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
namespace corsika::radio::interface {
namespace {
void jsonString(std::ostream& stream,std::string const& value){
  constexpr char hex[]="0123456789abcdef";stream<<'"';
  for(unsigned char ch:value){
    if(ch=='"'||ch=='\\')stream<<'\\'<<char(ch);
    else if(ch<32)stream<<"\\u00"<<hex[ch>>4]<<hex[ch&15];
    else stream<<char(ch);
  }stream<<'"';
}
std::complex<double> momentFactor(std::size_t k,std::size_t n,unsigned order){
  // complex pow(0,0) can be NaN in the host standard library. ZHS's final
  // derivative hid this at DC; an endpoint E impulse has a physical DC area.
  if(order==0)return {1.,0.};
  if(k==0)return {};
  return std::pow(std::complex<double>(0.,-2*3.141592653589793238462643383279502884*k/n),order);
}
}
Rendered render(Result const& result){
  auto const& c=result.config;std::size_t n=c.samples,channels=c.observers.size()*6,bins=n/2+1;
  bool coreas=c.algorithm==Algorithm::CoREAS;
  if(n<64||n>(1u<<20)||n%2||channels==0||channels>1536||c.moment_order>20||
     !std::isfinite(c.sample_rate_Hz)||c.sample_rate_Hz<=0.||
     result.moments.size()!=(c.moment_order+1)*channels*n||
     (coreas&&result.regularized_moments.size()!=result.moments.size())||
     (!coreas&&!result.regularized_moments.empty()))
    throw std::invalid_argument("invalid interface radio moment result");
  Rendered out;out.field_spectrum.resize(channels*bins);out.field.resize(channels*n);
  Eigen::FFT<double> fft;std::vector<double> time(n);std::vector<std::complex<double>> frequency(n),potential(bins),regularized(bins),full(n);
  constexpr double pi=3.141592653589793238462643383279502884;
  for(std::size_t channel=0;channel<channels;++channel){
    std::fill(potential.begin(),potential.end(),std::complex<double>{});
    std::fill(regularized.begin(),regularized.end(),std::complex<double>{});
    for(unsigned order=0;order<=c.moment_order;++order){
      std::copy_n(result.moments.data()+(order*channels+channel)*n,n,time.data());
      fft.fwd(frequency,time);
      for(std::size_t k=0;k<bins;++k)potential[k]+=frequency[k]*momentFactor(k,n,order);
      if(coreas){
        std::copy_n(result.regularized_moments.data()+(order*channels+channel)*n,n,time.data());
        fft.fwd(frequency,time);
        for(std::size_t k=0;k<bins;++k)regularized[k]+=frequency[k]*momentFactor(k,n,order);
      }
    }
    for(std::size_t k=0;k<bins;++k){
      auto derivative=std::complex<double>(0.,-2*pi*k*c.sample_rate_Hz/n);
      auto e=coreas?potential[k]+regularized[k]*derivative:potential[k]*derivative;
      if(k==0&&!coreas)e=0.;
      out.field_spectrum[channel*bins+k]=e;
      auto sampled=k==n/2?std::complex<double>(e.real(),0.):e;
      full[k]=sampled*c.sample_rate_Hz;
      if(k>0&&k<n/2)full[n-k]=std::conj(full[k]);
    }
    fft.inv(time,full);
    for(std::size_t k=0;k<n;++k){if(!std::isfinite(time[k]))throw std::runtime_error("nonfinite rendered interface field");out.field[channel*n+k]=time[k];}
  }
  return out;
}
void writeResult(Result const& r,std::string const& directory){
  namespace fs=std::filesystem;auto const& c=r.config;
  auto output=fs::path(directory);if(fs::exists(output))throw std::runtime_error("refuse to overwrite interface radio output");
  auto rendered=render(r);fs::create_directories(output);
  std::ofstream raw(output/"moments.bin",std::ios::binary);raw.write(reinterpret_cast<char const*>(r.moments.data()),r.moments.size()*sizeof(double));
  if(c.algorithm==Algorithm::CoREAS){
    std::ofstream regularized(output/"regularized_moments.bin",std::ios::binary);
    regularized.write(reinterpret_cast<char const*>(r.regularized_moments.data()),r.regularized_moments.size()*sizeof(double));
    if(!regularized)throw std::runtime_error("interface CoREAS regularized moment write failed");
  }
  std::ofstream config(output/"config.json");config<<std::setprecision(17);
  config<<"{\"algorithm\":\""<<algorithmName(c.algorithm)<<"\",\"samples\":"<<c.samples
    <<",\"sample_rate_Hz\":"<<c.sample_rate_Hz<<",\"start_time_s\":"<<c.start_time_s<<",\"moment_order\":"<<c.moment_order
    <<",\"subdivision_frequency_Hz\":"<<c.subdivision_frequency_Hz<<",\"fraunhofer_limit\":"<<c.fraunhofer_limit
    <<",\"maximum_subdivision_depth\":"<<c.maximum_subdivision_depth<<",\"geometry\":"<<int(c.propagation.geometry)
    <<",\"mesh_maximum_segment_m\":"<<c.mesh_maximum_segment_m
    <<",\"source_causality_policy\":\"declared-transport-and-input-ulp-beta-v2\""
    <<",\"transmission_candidate_search\":\""<<(c.propagation.transmission_bvh?"conservative-snell-bvh-v1":"exhaustive")<<"\""
    <<",\"dem_coverage_boundary\":"<<(c.propagation.coverage.view().enabled()?"true":"false")
    <<",\"outside_coverage_paths\":"<<r.statistics.outside_coverage_paths
    <<",\"optical_integration_samples\":"<<c.propagation.optical_integration_samples
    <<",\"visibility_tolerance_m\":"<<c.propagation.visibility_tolerance_m
    <<",\"shared_edge_tolerance_m\":"<<c.propagation.shared_edge_tolerance_m
    <<",\"maximum_device_bytes\":"<<c.maximum_device_bytes
    <<",\"plane_point_m\":["<<c.propagation.plane_point_m.x<<','<<c.propagation.plane_point_m.y<<','<<c.propagation.plane_point_m.z<<']'
    <<",\"plane_outward_normal\":["<<c.propagation.plane_outward_normal.x<<','<<c.propagation.plane_outward_normal.y<<','<<c.propagation.plane_outward_normal.z<<']'
    <<",\"scope\":\"straight optical legs, at most one transmitted interface; no reflected/diffracted rays\",\"observers\":[";
  for(std::size_t i=0;i<c.observers.size();++i){auto o=c.observers[i];if(i)config<<',';
    config<<"{\"name\":";jsonString(config,o.name);config<<",\"position_m\":["<<o.position_m.x<<','<<o.position_m.y<<','<<o.position_m.z<<"],\"region\":"<<o.region<<'}';}
  config<<"],\"media\":[";
  for(int i=0;i<2;++i){auto const& m=c.propagation.media[i];if(i)config<<',';
    config<<"{\"index\":"<<m.refractive_index<<",\"attenuation_length_m\":";
    if(std::isfinite(m.attenuation_length_m))config<<m.attenuation_length_m;else config<<"null";
    config<<",\"center_m\":["<<m.center_m.x<<','<<m.center_m.y<<','<<m.center_m.z<<"],\"reference_radius_m\":"<<m.reference_radius_m<<",\"radial_index\":[";
    for(std::size_t j=0;j<m.radial_index.size();++j){if(j)config<<',';config<<'['<<m.radial_index[j].height_m<<','<<m.radial_index[j].index<<']';}
    config<<"],\"radial_integration_breaks_m\":[";
    for(std::size_t j=0;j<m.radial_integration_breaks_m.size();++j){if(j)config<<',';config<<m.radial_integration_breaks_m[j];}config<<"]}";
  }config<<']';
  if(c.algorithm==Algorithm::CoREAS)config<<",\"moment_quantity\":\"electric_field_impulse\",\"moment_units\":\"V s/m\",\"regularized_moment_units\":\"V s^2/m\",\"coreas_cherenkov_threshold\":"<<c.coreas_cherenkov_threshold;
  config<<"}\n";
  std::ofstream field(output/"field.csv"),spectrum(output/"spectrum.csv");field<<std::setprecision(17);spectrum<<std::setprecision(17);
  field<<"observer,time_s,Ex_V_m,Ey_V_m,Ez_V_m,Ex_outside,Ey_outside,Ez_outside,Ex_inside,Ey_inside,Ez_inside\n";
  spectrum<<"observer,frequency_Hz,Ex_real,Ex_imag,Ey_real,Ey_imag,Ez_real,Ez_imag\n";
  std::size_t n=c.samples,bins=n/2+1;
  for(std::size_t o=0;o<c.observers.size();++o){
    for(std::size_t k=0;k<n;++k){field<<o<<','<<c.start_time_s+k/c.sample_rate_Hz;
      for(int a=0;a<3;++a)field<<','<<rendered.field[(o*6+a)*n+k]+rendered.field[(o*6+3+a)*n+k];
      for(int a=0;a<6;++a){field<<','<<rendered.field[(o*6+a)*n+k];}
      field<<'\n';}
    for(std::size_t k=0;k<bins;++k){spectrum<<o<<','<<k*c.sample_rate_Hz/n;
      for(int a=0;a<3;++a){auto e=rendered.field_spectrum[(o*6+a)*bins+k]+rendered.field_spectrum[(o*6+3+a)*bins+k];
        // Include the declared absolute start time in the continuous transform.
        e*=std::exp(std::complex<double>(0.,-2*3.14159265358979323846*k*c.sample_rate_Hz/n*c.start_time_s));spectrum<<','<<e.real()<<','<<e.imag();}spectrum<<'\n';}
  }
  if(!raw||!config||!field||!spectrum)throw std::runtime_error("interface radio output write failed");
}
} // namespace corsika::radio::interface
