// Calls the production beta5 optical functions directly; no reimplementation.
#include <corsika/modules/radio/interface/Propagation.hpp>
#include <fstream>
#include <iomanip>
#include <filesystem>
namespace ri=corsika::radio::interface;
namespace d=ri::detail;
int main(int argc,char** argv){
  if(argc!=2)return 2;std::filesystem::path root=argv[1];std::filesystem::create_directories(root);
  std::ofstream out(root/"beta5_plane.csv");out<<std::setprecision(17);
  out<<"n1,n2,angle_deg,transmitted,interface_x,observer_x,sin_emit,sin_receive,time_s,t_s,t_p,flux_s,flux_p,jacobian_m2,spreading_per_m,transfer_s,transfer_p\n";
  constexpr double pi=3.14159265358979323846;
  for(int medium=0;medium<2;++medium){
    double n1=medium?1.0003:2.,n2=medium?2.:1.;
    for(int i=0;i<=320;++i){
      double angle=i*.25,theta=angle*pi/180.;auto f=d::fresnel(n1,n2,std::cos(theta));
      if(!f.transmitted||f.cos_transmitted<1.e-8){
        out<<n1<<','<<n2<<','<<angle<<",0,0,0,0,0,0,0,0,0,0,0,0,0,0\n";continue;
      }
      d::PropagationView v;v.geometry=ri::Geometry::Plane;v.media[1].index=n1;v.media[0].index=n2;
      double lateral=100*std::tan(theta)+300*(n1/n2*std::sin(theta))/f.cos_transmitted;
      ri::RayQuery query{{0,0,-100},{lateral,0,300},1,0};auto ray=d::transmittedPath(v,query,0);
      if(ray.status!=ri::PathStatus::Valid)return 3;
      double flux=std::sqrt(n2*ray.receive_direction.z/(n1*ray.emit.z));
      auto s=d::applyTransfer(ray,{0,1,0});auto pol=d::applyTransfer(ray,{ray.emit.z,0,-ray.emit.x});
      out<<n1<<','<<n2<<','<<angle<<",1,"<<ray.interface_point.x<<','<<lateral<<','
         <<ray.emit.x<<','<<ray.receive_direction.x<<','<<ray.time_s<<','<<ray.t_s<<','<<ray.t_p<<','
         <<flux*ray.t_s<<','<<flux*ray.t_p<<','<<ray.jacobian_m2<<','<<ray.spreading_per_m<<','
         <<d::norm(s)<<','<<d::norm(pol)<<'\n';
    }
  }
  std::ofstream gradient(root/"beta5_gradient.csv");gradient<<std::setprecision(17)<<"z_m,x_m,time_s\n";
  ri::IndexSample table[]{{-1,2.0002},{1100,1.78}};
  d::PropagationView v;v.media[0].center={0,0,-1.e10};v.media[0].radius=1.e10;
  v.media[0].samples=table;v.media[0].count=2;v.integration_samples=64;
  for(int i=1;i<=100;++i){double z=i*10.;auto ray=d::directPath(v,{{0,0,0},{.6*z,0,z},0,0});
    if(ray.status!=ri::PathStatus::Valid)return 4;
    gradient<<z<<','<<.6*z<<','<<ray.time_s<<'\n';}
  std::ofstream attenuation(root/"beta5_attenuation.csv");
  attenuation<<std::setprecision(17)<<"distance_m,amplitude_per_m,time_s\n";
  d::PropagationView uniform;uniform.media[0].index=1.33;uniform.media[0].attenuation_length_m=100.;
  for(int i=1;i<=200;++i){double r=i*5.;auto ray=d::directPath(uniform,{{0,0,0},{r,0,0},0,0});
    attenuation<<r<<','<<d::norm(d::applyTransfer(ray,{0,1,0}))<<','<<ray.time_s<<'\n';}
}
