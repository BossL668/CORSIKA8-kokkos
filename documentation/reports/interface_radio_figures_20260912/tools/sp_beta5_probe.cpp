// Actual beta5 Fresnel and vector-transfer functions, called without Kokkos runtime.
#include <corsika/modules/radio/interface/Propagation.hpp>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <iostream>
using namespace corsika::radio::interface;
using namespace corsika::radio::interface::detail;
int main(int argc,char** argv){
 if(argc!=2)return 2;std::filesystem::path root=argv[1];std::ifstream in(root/"sp_input.txt");
 Path p;for(auto v:{&p.emit,&p.receive_direction,&p.normal})in>>v->x>>v->y>>v->z;
 double n1,n2,g,t0,t1,q;Vec3 a,b;in>>n1>>n2>>g>>a.x>>a.y>>a.z>>b.x>>b.y>>b.z>>t0>>t1>>q;
 if(!in)return 3;constexpr double pi=3.14159265358979323846;
 double ci=dot(p.emit,p.normal),ct=dot(p.receive_direction,p.normal),angle=std::acos(ci)*180/pi;
 double critical=std::asin(n2/n1)*180/pi,brewster=std::atan(n2/n1)*180/pi;
 auto source=scale(sub(b,a),q/(light_speed*(t1-t0)));
 source=sub(source,scale(p.emit,dot(source,p.emit)));source=unit(source);
 auto s=unit(cross(p.emit,p.normal)),ip=unit(cross(s,p.emit)),tp=unit(cross(s,p.receive_direction));
 auto f=fresnel(n1,n2,ci);double common=g*std::sqrt(n2*ct/(n1*ci));
 transfer(p,common*f.t_s,common*f.t_p);
 auto out=applyTransfer(p,source),outS=applyTransfer(p,s),outP=applyTransfer(p,ip);
 std::ofstream selected(root/"sp_selected.csv");selected<<std::setprecision(17);
 selected<<"n1,n2,incidence_deg,transmission_deg,brewster_deg,critical_deg,t_s,t_p,T_s,T_p,common_per_m,in_s,in_p,out_s,out_p,m_ss,m_sp,m_ps,m_pp,longitudinal_per_m,s_x,s_y,s_z,pi_x,pi_y,pi_z,pt_x,pt_y,pt_z\n";
 selected<<n1<<','<<n2<<','<<angle<<','<<std::acos(ct)*180/pi<<','<<brewster<<','<<critical<<','<<f.t_s<<','<<f.t_p<<','<<n2*ct/(n1*ci)*f.t_s*f.t_s<<','<<n2*ct/(n1*ci)*f.t_p*f.t_p<<','<<common<<','<<dot(source,s)<<','<<dot(source,ip)<<','<<dot(out,s)<<','<<dot(out,tp)<<','<<dot(outS,s)<<','<<dot(outP,s)<<','<<dot(outS,tp)<<','<<dot(outP,tp)<<','<<dot(out,p.receive_direction);
 for(auto v:{s,ip,tp})selected<<','<<v.x<<','<<v.y<<','<<v.z;selected<<'\n';
 std::ofstream scan(root/"sp_fresnel_scan.csv");scan<<std::setprecision(17)<<"angle_deg,transmitted,t_s,t_p,T_s,T_p\n";
 auto row=[&](double deg){
  auto x=fresnel(n1,n2,std::cos(deg*pi/180));
  scan<<deg<<','<<x.transmitted;
  if(x.transmitted){double flux=n2*x.cos_transmitted/(n1*std::cos(deg*pi/180));scan<<','<<x.t_s<<','<<x.t_p<<','<<flux*x.t_s*x.t_s<<','<<flux*x.t_p*x.t_p;}
  else scan<<",nan,nan,0,0";
  scan<<'\n';
 };
 for(int j=0;j<=680;++j)row(j*.05);
 row(angle);row(brewster);row(critical-1.e-7);
 std::cout<<"PASS actual beta5 s/p probe: "<<angle<<" deg, t_s "<<f.t_s<<", t_p "<<f.t_p<<'\n';
}

