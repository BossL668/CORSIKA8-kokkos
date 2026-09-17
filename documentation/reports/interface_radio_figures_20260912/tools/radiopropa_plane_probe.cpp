// Independent, unmodified RadioPropa Discontinuity, sampled with ray bundles.
#include <radiopropa/Candidate.h>
#include <radiopropa/Geometry.h>
#include <radiopropa/module/Discontinuity.h>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <cmath>
using namespace radiopropa;
struct Ray{bool valid{};Vector3d direction,amplitude;};
Ray refract(Vector3d direction,double n1,double n2){
  Candidate candidate;Vector3d point(0,0,0);double distance=1.e-8;
  candidate.previous.setPosition(point-direction*distance);
  candidate.current.setPosition(point+direction*distance);
  candidate.previous.setDirection(direction);candidate.current.setDirection(direction);
  candidate.previous.setAmplitude(Vector3d(0,1,0));candidate.current.setAmplitude(Vector3d(0,1,0));
  Discontinuity boundary(new Plane(Vector3d(0,0,0),Vector3d(0,0,1)),n1,n2);
  boundary.process(&candidate);
  if(candidate.secondaries.empty())return {};
  auto const& state=candidate.secondaries[0]->current;
  return {true,state.getDirection(),state.getAmplitude()};
}
Vector3d onScreen(Vector3d direction,double n1,double n2,Vector3d center,Vector3d normal){
  auto r=refract(direction,n1,n2);
  if(!r.valid)throw std::runtime_error("ray bundle crosses critical angle");
  Vector3d hit(100*direction.x/direction.z,100*direction.y/direction.z,0);
  return hit+r.direction*((center-hit).dot(normal)/r.direction.dot(normal));
}
int main(int argc,char** argv){
  if(argc!=2)return 2;std::filesystem::path root=argv[1];std::filesystem::create_directories(root);
  std::ofstream out(root/"radiopropa_plane.csv");out<<std::setprecision(17);
  out<<"n1,n2,angle_deg,transmitted,sin_receive,amplitude_s,observer_x\n";
  constexpr double pi=3.14159265358979323846;
  for(int medium=0;medium<2;++medium){
    double n1=medium?1.0003:2.,n2=medium?2.:1.;
    for(int i=0;i<=320;++i){double angle=i*.25,theta=angle*pi/180.;
      auto r=refract(Vector3d(std::sin(theta),0,std::cos(theta)),n1,n2);
      out<<n1<<','<<n2<<','<<angle<<','<<r.valid<<','<<(r.valid?r.direction.x:0)<<','
         <<(r.valid?r.amplitude.getR():0)<<','<<(r.valid?100*std::tan(theta)+300*r.direction.x/r.direction.z:0)<<'\n';
    }
  }
  std::ofstream beam(root/"radiopropa_beam.csv");beam<<std::setprecision(17)<<"n1,n2,angle_deg,h_rad,jacobian_m2\n";
  for(int medium=0;medium<2;++medium){
    double n1=medium?1.0003:2.,n2=medium?2.:1.;
    for(double angle:{0.,10.,20.,29.}){double theta=angle*pi/180.;
      Vector3d direction(std::sin(theta),0,std::cos(theta)),u(std::cos(theta),0,-std::sin(theta)),v(0,1,0);
      auto ray=refract(direction,n1,n2);Vector3d center(100*std::tan(theta)+300*ray.direction.x/ray.direction.z,0,300);
      for(double h:{1.e-3,1.e-4,1.e-5,1.e-6}){
        auto du=(onScreen((direction+u*h).getUnitVector(),n1,n2,center,ray.direction)-onScreen((direction-u*h).getUnitVector(),n1,n2,center,ray.direction))/(2*h);
        auto dv=(onScreen((direction+v*h).getUnitVector(),n1,n2,center,ray.direction)-onScreen((direction-v*h).getUnitVector(),n1,n2,center,ray.direction))/(2*h);
        beam<<n1<<','<<n2<<','<<angle<<','<<h<<','<<du.cross(dv).getR()<<'\n';
      }
    }
  }
}
