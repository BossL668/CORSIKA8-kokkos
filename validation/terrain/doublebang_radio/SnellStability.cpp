#include <corsika/modules/radio/interface/Propagation.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace ri=corsika::radio::interface;
namespace d=ri::detail;
int main(int argc,char** argv){
  if(argc!=2)return 2;
  std::ofstream out(argv[1]);out<<std::setprecision(17);
  out<<"offset,depth,reverse,status,snell_residual,source_length_m,destination_length_m,time_s,emit_x,emit_y,emit_z,receive_x,receive_y,receive_z,actual_depth_m\n";
  for(double offset:{0.,2500.})for(double depth:{1.e-6,1.e-4,.01})for(unsigned reverse:{0,1}){
    d::PropagationView v;v.geometry=ri::Geometry::Plane;v.plane_point={offset,offset,offset};
    v.media[0].index=1.000327;v.media[1].index=2.;
    ri::Vec3 s{offset,offset,offset-depth},o{offset+6000.,offset+3000.,offset+2000.};
    auto p=d::transmittedPath(v,reverse?ri::RayQuery{o,s,0,1}:ri::RayQuery{s,o,1,0},0);
    out<<offset<<','<<depth<<','<<reverse<<','<<unsigned(p.status)<<','<<p.snell_residual<<','<<p.source_length_m<<','<<p.destination_length_m<<','<<p.time_s
       <<','<<p.emit.x<<','<<p.emit.y<<','<<p.emit.z<<','<<p.receive_direction.x<<','<<p.receive_direction.y<<','<<p.receive_direction.z<<','<<(offset-s.z)<<'\n';
  }
}

