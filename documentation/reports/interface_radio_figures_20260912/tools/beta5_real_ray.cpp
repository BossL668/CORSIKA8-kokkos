// Actual DEM and recorded source replay. The optical functions are beta5 production code.
#include <corsika/modules/radio/interface/Propagation.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <functional>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <iostream>
using namespace corsika::radio::interface;
using namespace corsika::radio::interface::detail;
namespace fs=std::filesystem;
double axis(Vec3 v,int k){return k==0?v.x:k==1?v.y:v.z;}
int main(int argc,char** argv){
 if(argc!=3&&argc!=4)return 2;fs::path root=argv[1];std::ifstream mesh(argv[2],std::ios::binary);
 std::string line;unsigned nv=0,nf=0;
 while(std::getline(mesh,line)){std::istringstream s(line);std::string a,b;s>>a>>b;
  if(a=="element"&&b=="vertex")s>>nv;if(a=="element"&&b=="face")s>>nf;if(a=="end_header")break;}
 if(!nv||!nf)throw std::runtime_error("invalid PLY header");
 corsika::terrain::FlatTerrainData terrain;terrain.vertices.resize(nv);
 for(auto& v:terrain.vertices){mesh.read(reinterpret_cast<char*>(&v.x),8);mesh.read(reinterpret_cast<char*>(&v.y),8);mesh.read(reinterpret_cast<char*>(&v.z),8);}
 terrain.indexed_triangles.resize(nf);
 for(auto& t:terrain.indexed_triangles){unsigned char count;mesh.read(reinterpret_cast<char*>(&count),1);if(count!=3)throw std::runtime_error("nontriangle");
  mesh.read(reinterpret_cast<char*>(t.vertex),12);t.normal=unit(cross(sub(terrain.vertices[t.vertex[1]],terrain.vertices[t.vertex[0]]),sub(terrain.vertices[t.vertex[2]],terrain.vertices[t.vertex[0]])));}
 if(!mesh)throw std::runtime_error("truncated mesh");
 terrain.indices.resize(nf);std::iota(terrain.indices.begin(),terrain.indices.end(),0);
 std::function<void(unsigned,unsigned)> build=[&](unsigned first,unsigned last){
  unsigned id=terrain.nodes.size();Vec3 lo{HUGE_VAL,HUGE_VAL,HUGE_VAL},hi{-HUGE_VAL,-HUGE_VAL,-HUGE_VAL};
  for(unsigned j=first;j<last;++j)for(auto k:terrain.indexed_triangles[terrain.indices[j]].vertex){auto v=terrain.vertices[k];
   lo={std::min(lo.x,v.x),std::min(lo.y,v.y),std::min(lo.z,v.z)};hi={std::max(hi.x,v.x),std::max(hi.y,v.y),std::max(hi.z,v.z)};}
  terrain.nodes.push_back({lo,hi,0,first,0});
  if(last-first<=8)terrain.nodes[id].count=last-first;
  else{auto d=sub(hi,lo);int k=d.x>d.y?(d.x>d.z?0:2):(d.y>d.z?1:2);unsigned mid=(first+last)/2;
   auto center=[&](unsigned t){auto f=terrain.indexed_triangles[t];return axis(add(add(terrain.vertices[f.vertex[0]],terrain.vertices[f.vertex[1]]),terrain.vertices[f.vertex[2]]),k);};
   std::nth_element(terrain.indices.begin()+first,terrain.indices.begin()+mid,terrain.indices.begin()+last,[&](unsigned a,unsigned b){return center(a)<center(b);});
   build(first,mid);build(mid,last);}
  terrain.nodes[id].skip=terrain.nodes.size();
 };build(0,nf);
 PropagationView view;view.geometry=Geometry::Mesh;view.mesh=terrain.view();view.faces=nf;
 std::ifstream medium(root/"media.txt");std::vector<IndexSample> table[2];std::vector<double> breaks[2];
 for(int i=0;i<2;++i){auto& m=view.media[i];medium>>m.index>>m.center.x>>m.center.y>>m.center.z>>m.radius>>m.count>>m.break_count;
  table[i].resize(m.count);breaks[i].resize(m.break_count);for(auto& s:table[i])medium>>s.height_m>>s.index;for(auto& b:breaks[i])medium>>b;
  m.samples=table[i].data();m.breaks=breaks[i].data();}
 if(argc==4){
  std::ifstream checks(root/"path_check_inputs.txt");unsigned nc;checks>>nc;
  std::ofstream report(root/"terrain_path_checks.csv");report<<std::setprecision(17)<<"case,reference,segments,blocked,inside_selected_face,endpoint_miss_m\n";
  for(unsigned i=0;i<nc;++i){std::string name;unsigned region,faceId;Vec3 source,observer;
   checks>>name>>region>>faceId>>source.x>>source.y>>source.z>>observer.x>>observer.y>>observer.z;
   for(std::string reference:{"native_shoot","scipy_reference"}){
    std::ifstream path(root/(name+"_"+reference+".csv"));std::getline(path,line);std::vector<Vec3> points{source};
    while(std::getline(path,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);Vec3 x;row>>x.x>>x.y>>x.z;if(!row)throw std::runtime_error("bad path CSV");if(norm(sub(x,points.back()))>1.e-12)points.push_back(x);}
    if(points.size()<2)throw std::runtime_error("empty path CSV");
    unsigned blocked=0;for(unsigned j=1;j<points.size();++j)if(!segmentClear(view,points[j-1],points[j]))++blocked;
    bool inside=!region||containsTriangle(view,faceId,points[1]);double miss=norm(sub(points.back(),observer));
    report<<name<<','<<reference<<','<<points.size()-1<<','<<blocked<<','<<inside<<','<<miss<<'\n';
    if(blocked||!inside||miss>2.e-7)throw std::runtime_error("external path failed real-terrain validation");
   }
  }std::cout<<"PASS external rays clear the actual DEM mesh\n";return 0;
 }
 std::ifstream obs(root/"observers.txt");unsigned no;obs>>no;std::vector<Observer> observers(no);
 for(auto& o:observers)obs>>o.name>>o.position_m.x>>o.position_m.y>>o.position_m.z>>o.region;
 std::ifstream tracks(root/"tracks.txt");unsigned nt;tracks>>nt;
 std::ofstream out(root/"beta5_real_paths.csv");out<<std::setprecision(17);
 out<<"step,history,pdg,region,observer,face,sx,sy,sz,ox,oy,oz,px,py,pz,nx,ny,nz,ex,ey,ez,rx,ry,rz,n1,n2,rock_m,air_m,flight_s,emission_s,arrival_s,snell_residual,t_s,t_p,jacobian_m2,spreading_per_m,leaf_criterion,leaf_length_m\n";
 for(unsigned j=0;j<nt;++j){unsigned step,history;int pdg;double weight,t0,t1;Vec3 a,b;
  unsigned region;tracks>>step>>pdg>>weight>>a.x>>a.y>>a.z>>b.x>>b.y>>b.z>>t0>>t1>>history>>region;
  Vec3 source=scale(add(a,b),.5),delta=sub(b,a);double length=norm(delta);unsigned nvalid=0;
  for(auto const& o:observers)for(unsigned faceId=0;faceId<(region?nf:1);++faceId){
   auto path=region?transmittedPath(view,{source,o.position_m,region,o.region},faceId):directPath(view,{source,o.position_m,region,o.region});
   if(path.status!=PathStatus::Valid)continue;
   double distance=path.source_length_m+path.destination_length_m,cosine=dot(unit(delta),path.emit);
   double criterion=std::max(length/distance,std::max(0.,1-cosine*cosine)*length*length/distance*2.e9/light_speed*2*3.141592653589793);
   if(length>.1||criterion>.025)continue;
   if(!region&&directShadowBoundary(view,a,b,o.position_m,0,1)>0)continue;
   if(!region){path.interface_point=source;path.destination_length_m=path.source_length_m;path.source_length_m=0;}
   out<<step<<','<<history<<','<<pdg<<','<<region<<','<<o.name<<','<<faceId;
   for(auto v:{source,o.position_m,path.interface_point,path.normal,path.emit,path.receive_direction})out<<','<<v.x<<','<<v.y<<','<<v.z;
   out<<','<<path.interface_source_index<<','<<path.interface_destination_index<<','<<path.source_length_m<<','<<path.destination_length_m<<','<<path.time_s<<','<<.5*(t0+t1)<<','<<.5*(t0+t1)+path.time_s<<','<<path.snell_residual<<','<<path.t_s<<','<<path.t_p<<','<<path.jacobian_m2<<','<<path.spreading_per_m<<','<<criterion<<','<<length<<'\n';
   ++nvalid;
  }std::cout<<"track "<<step<<" valid optical branches "<<nvalid<<'\n';
 }
}
