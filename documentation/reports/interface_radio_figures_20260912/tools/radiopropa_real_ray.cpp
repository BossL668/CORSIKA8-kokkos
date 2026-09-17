// Independent public RadioPropa propagation; no beta5 headers or optical functions.
#include <radiopropa/Candidate.h>
#include <radiopropa/Geometry.h>
#include <radiopropa/ScalarField.h>
#include <radiopropa/module/Discontinuity.h>
#include <radiopropa/module/PropagationCK.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
using namespace radiopropa;namespace fs=std::filesystem;
constexpr double C=299792458.;
struct AirTable: ScalarField{
 Vector3d center;double radius,frozen;bool constant;std::vector<std::pair<double,double>> table;
 AirTable(fs::path path,bool fixed,double n):frozen(n),constant(fixed){
  std::ifstream in(path);double index;unsigned count,breaks;in>>index>>center.x>>center.y>>center.z>>radius>>count>>breaks;
  table.resize(count);for(auto& pair:table)in>>pair.first>>pair.second;
 }
 std::pair<double,double> valueSlope(Vector3d const& p)const{
  double h=(p-center).getR()-radius;
  if(h<table.front().first||h>table.back().first)throw std::runtime_error("outside actual index table");
  auto it=std::upper_bound(table.begin(),table.end(),std::make_pair(h,HUGE_VAL));
  if(it==table.end())--it;auto a=*(it-1),b=*it;double slope=(b.second-a.second)/(b.first-a.first);
  return {a.second+slope*(h-a.first),slope};
 }
 double getValue(Vector3d const& p)const override{return constant?frozen:valueSlope(p).first;}
 Vector3d getGradient(Vector3d const& p)const override{return constant?Vector3d(0,0,0):(p-center).getUnitVector()*valueSlope(p).second;}
};
struct Input{std::string name;int rock;Vector3d source,observer,point,normal,launch;double n1,n2;};
struct Sample{Vector3d point;double time;};
struct Result{Vector3d end,point,launch,exit;double time,rockLength;int steps;std::vector<Sample> path;};
Result trace(Input const& x,AirTable* air,Vector3d launch,double maxStep,bool record){
 Result r{};r.launch=launch;r.point=x.source;Vector3d direction=launch;
 r.path.push_back({x.source,0});
 if(x.rock){
  r.rockLength=(x.point-x.source).dot(x.normal)/launch.dot(x.normal);
  if(!(r.rockLength>0))throw std::runtime_error("no forward plane hit");
  r.point=x.source+launch*r.rockLength;r.time=r.rockLength*x.n1/C;
  Candidate boundaryRay;
  boundaryRay.previous.setPosition(r.point-launch*1.e-7);boundaryRay.current.setPosition(r.point+launch*1.e-7);
  boundaryRay.previous.setDirection(launch);boundaryRay.current.setDirection(launch);
  auto polarization=launch.cross(x.normal).getUnitVector();
  boundaryRay.previous.setAmplitude(polarization);boundaryRay.current.setAmplitude(polarization);
  Discontinuity boundary(new Plane(r.point,x.normal),x.n1,air->getValue(r.point));boundary.process(&boundaryRay);
  if(boundaryRay.secondaries.empty())throw std::runtime_error("TIR");
  direction=boundaryRay.secondaries[0]->current.getDirection();
  r.path.push_back({r.point,r.time});
 }
 r.exit=direction;Vector3d screen=(x.observer-x.point).getUnitVector();
 ParticleState state(0,250.e6,r.point,direction,Vector3d(0,1,0));Candidate ray(state);ray.setNextStep(maxStep);
 // The pinned ParticleState::setDirection uses acos(old.dot(new)) and skips
 // an update if this rounds to zero. Weak atmospheric gradients therefore
 // need steps above its angular rounding floor. Use explicit fixed steps;
 // verify 40/20/10 m against an independent continuous ODE reference.
 PropagationCK propagate(air,1.e-12,maxStep,maxStep);
 double startTime=r.time;
 for(int i=0;i<1000000;++i){
  double oldTime=ray.getPropagationTime();auto before=ray.current.getPosition();
  propagate.process(&ray);auto after=ray.current.getPosition();++r.steps;
  if((after-x.observer).dot(screen)>=0){
   double f=(x.observer-before).dot(screen)/(after-before).dot(screen);
   r.end=before+(after-before)*f;
   r.time=startTime+oldTime+f*(ray.getPropagationTime()-oldTime);
   r.path.push_back({r.end,r.time});return r;
  }
  if(record)r.path.push_back({after,startTime+ray.getPropagationTime()});
 }
 throw std::runtime_error("ray failed to reach observer screen");
}
int main(int argc,char** argv){
 if(argc!=2)return 2;fs::path root=argv[1];std::ifstream in(root/"selected_ray.txt");unsigned count;in>>count;
 std::ofstream summary(root/"radiopropa_summary.csv");summary<<std::setprecision(17);
 summary<<"case,mode,max_step_m,miss_m,flight_s,steps,iterations,px,py,pz,ex,ey,ez,rx,ry,rz,endx,endy,endz\n";
 for(unsigned i=0;i<count;++i){
  Input x;in>>x.name>>x.rock;
  for(auto ptr:{&x.source,&x.observer,&x.point,&x.normal,&x.launch})in>>ptr->x>>ptr->y>>ptr->z;in>>x.n1>>x.n2;
  auto w=x.launch,trial=std::abs(w.z)<.9?Vector3d(0,0,1):Vector3d(0,1,0);
  auto u=w.cross(trial).getUnitVector(),v=w.cross(u).getUnitVector();
  for(std::string mode:{"frozen_forward","native_forward","native_shoot"}){
   for(double maxStep:{40.,20.,10.}){
    ref_ptr<AirTable> air=new AirTable(root/"media.txt",mode=="frozen_forward",x.n2);
    double a=0,b=0;int iterations=0;Result r;
    auto direction=[&](double av,double bv){return (w+u*av+v*bv).getUnitVector();};
    if(mode=="native_shoot"){
     for(int iteration=0;iteration<10;++iteration){
      r=trace(x,air.get(),direction(a,b),maxStep,false);auto error=r.end-x.observer;
      std::cerr<<x.name<<' '<<mode<<' '<<maxStep<<" iteration "<<iteration<<" miss "<<std::setprecision(12)<<error.getR()<<'\n';
      if(error.getR()<2.e-8)break;
      double h=1.e-6;
      auto da=(trace(x,air.get(),direction(a+h,b),maxStep,false).end-trace(x,air.get(),direction(a-h,b),maxStep,false).end)/(2*h);
      auto db=(trace(x,air.get(),direction(a,b+h),maxStep,false).end-trace(x,air.get(),direction(a,b-h),maxStep,false).end)/(2*h);
      double ja=da.dot(u),jb=db.dot(u),jc=da.dot(v),jd=db.dot(v),e=error.dot(u),f=error.dot(v),det=ja*jd-jb*jc;
      if(std::abs(det)<1.e-15)throw std::runtime_error("singular shooting Jacobian");
      a-=(jd*e-jb*f)/det;b-=(-jc*e+ja*f)/det;++iterations;
     }
    }
    r=trace(x,air.get(),direction(a,b),maxStep,true);double miss=(r.end-x.observer).getR();
    if(mode=="native_shoot"&&miss>2.e-7){summary.flush();throw std::runtime_error("shooting did not converge");}
    summary<<x.name<<','<<mode<<','<<maxStep<<','<<miss<<','<<r.time<<','<<r.steps<<','<<iterations;
    for(auto q:{r.point,r.launch,r.exit,r.end})summary<<','<<q.x<<','<<q.y<<','<<q.z;summary<<'\n';
    if(maxStep==10.){
     std::ofstream points(root/(x.name+"_"+mode+".csv"));points<<std::setprecision(17)<<"x,y,z,time_s\n";
     for(auto s:r.path)points<<s.point.x<<','<<s.point.y<<','<<s.point.z<<','<<s.time<<'\n';
    }
   }
  }
  std::cout<<"RadioPropa complete "<<x.name<<'\n';
 }
}
