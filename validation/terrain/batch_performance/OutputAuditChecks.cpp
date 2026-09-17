// Output-only concurrency: exact bytes, interface ambiguity and failure order.
#include <applications/detail/mountain/TerrainShowerOutput.hpp>
#include <applications/detail/mountain/TerrainScene.hpp>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <random>
using namespace corsika;
using namespace corsika::units::si;
using Output=corsika::applications::terrain::ShowerOutput;
using Record=corsika::terrain::TerrainEmStep;
void require(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
std::string bytes(std::filesystem::path const& file){
  std::ifstream stream(file,std::ios::binary);
  return {std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
}
void geometryReference(terrain::TriangularMesh const& mesh,terrain::Environment const& env){
  corsika::applications::terrain::OutputMeshAudit audit(&mesh);auto cs=mesh.getCoordinateSystem();
  require(audit.enabled(),"flat audit inactive");std::uint64_t queries=0;
  auto check=[&](Point const& p){
    require(audit.contains(p)==mesh.contains(p),"flat audit differs from CPU mesh containment");
    auto const& root=std::as_const(*env.getUniverse());
    require(audit.containingNode(root,p)==root.getContainingNode(p),"flat audit differs from CPU volume tree");++queries;
  };
  auto lo=mesh.getBounds().getMin().getCoordinates(cs),hi=mesh.getBounds().getMax().getCoordinates(cs);
  std::mt19937_64 random(761231);std::uniform_real_distribution<double> uniform(-.01,1.01);
  for(unsigned i=0;i<10000;++i)check(Point(cs,
    lo.getX()+uniform(random)*(hi.getX()-lo.getX()),lo.getY()+uniform(random)*(hi.getY()-lo.getY()),lo.getZ()+uniform(random)*(hi.getZ()-lo.getZ())));
  auto stride=std::max<std::size_t>(1,mesh.getTriangleCount()/1000);
  for(std::size_t i=0;i<mesh.getTriangleCount();i+=stride){
    auto const& tri=mesh.getTriangle(i);auto center=tri.computeCentroid(mesh.getVertices());
    auto vertex=mesh.getVertex(tri.getVertexIndices()[0]);
    for(double distance:{-1.e-5,-1.e-6,-1.e-8,-1.e-9,0.,1.e-9,1.e-8,1.e-6,1.e-5}){
      check(center+tri.getNormal()*(distance*1_m));check(vertex+tri.getNormal()*(distance*1_m));
    }
  }
  auto translated=make_translation(cs,{.25_m,-.75_m,.125_m});
  check(Point(translated,0_m,0_m,0_m));
  std::cout<<"PASS CPU mesh/tree oracle "<<queries<<" points; random volume, faces, vertices, nanometre offsets and transformed frame\n";
}
int main(int argc,char** argv){try{
  if(argc==2){
    terrain::Environment env;corsika::applications::terrain::Scene scene(argv[1]);auto mesh=scene.build(env);
    geometryReference(*mesh.mesh,env);return 0;
  }
  terrain::Environment env;auto cs=env.getCoordinateSystem();
  terrain::TerrainAtmosphereConfig cfg{2802.};terrain::buildAtmosphere(env,cfg);
  terrain::MeshData data;
  data.vertices={{{-1.,-1.,-1.}},{{1.,-1.,-1.}},{{1.,1.,-1.}},{{-1.,1.,-1.}},
                 {{-1.,-1.,1.}},{{1.,-1.,1.}},{{1.,1.,1.}},{{-1.,1.,1.}}};
  data.faces={{{0,2,1}},{{0,3,2}},{{4,5,6}},{{4,6,7}},{{0,1,5}},{{0,5,4}},
              {{1,2,6}},{{1,6,5}},{{2,3,7}},{{2,7,6}},{{3,0,4}},{{3,4,7}}};
  auto mesh=terrain::embedTerrain(env,cfg,terrain::ClosedMesh(terrain::MeshLoader::toPoints(data,cs,1_m),data.faces),Point(cs,0_m,0_m,0_m));
  geometryReference(*mesh.mesh,env);
  std::vector<Record> original(513);
  for(std::size_t i=0;i<original.size();++i){
    auto& r=original[i];r.has_track=i%10!=0;r.start.pid=22;r.start.weight=1.;
    r.start.history_id=i+1;r.start.medium_id=i%2;r.start.energy_GeV=1.;r.start.direction[0]=1.;
    r.start.position_m[0]=i%2?.1:2.;r.end=r.start;r.end.energy_GeV=.999;
    r.path_midpoint_m[0]=r.start.position_m[0];r.deposited_GeV=.001;
    r.outcome=terrain::TerrainEmOutcome::Continuation;
  }
  // Exactly on the interface, intentionally use the opposite logical owner.
  auto* node=env.getUniverse()->getContainingNode(Point(cs,1_m,0_m,0_m));
  original[7].path_midpoint_m[0]=1.;
  original[7].start.medium_id=!bool(dynamic_cast<terrain::TriangularMesh const*>(&node->getVolume()));
  for(auto const& scenario:{"valid","mismatch","missing_mesh","identity_first"}){
    auto records=original;
    if(std::string(scenario)!="valid"){
      records[301].start.medium_id=0;records[301].path_midpoint_m[0]=.1;
    }
    if(std::string(scenario)=="identity_first"){
      records[11].crossed_material=true;records[11].end.history_id=9999;
    }
    std::string firstError;YAML::Node firstSummary;
    for(unsigned threads:{1u,32u,256u}){
      auto folder=std::filesystem::path(std::string(scenario)+std::to_string(threads));
      require(std::filesystem::create_directory(folder),"test output already exists");
      Output output(env,std::string(scenario)=="missing_mesh"?nullptr:mesh.mesh,10000);
      output.device_output_threads=threads;output.ledger.enabled=true;
      output.reference_device_geometry=threads==1;
      output.startOfLibrary(boost::filesystem::path(folder.string()));output.startOfShower(0);
      std::string error;
      try{
        auto guard=output.prepareDeviceBatch(records);
        for(auto const& r:records)if(r.has_track)output.recordDeviceStep(r);
      }catch(std::runtime_error const& e){error=e.what();}
      // A failed ordered callback must not leave a pointer to its old batch.
      std::vector<Record> empty;
      {auto guard=output.prepareDeviceBatch(empty);}
      output.endOfShower(0);output.endOfLibrary();auto summary=output.getSummary();
      require((std::string(scenario)=="valid")==error.empty(),"expected audit failure missing");
      if(threads==1){firstError=error;firstSummary=summary;}
      else{
        require(error==firstError,"parallel audit changed first failure");
        for(auto const& name:{"tracks.csv","deposits.csv","domain_exits.csv","window_survivors.csv"})
          require(bytes(folder/name)==bytes(std::filesystem::path(std::string(scenario)+"1")/name),"parallel audit changed CSV bytes");
        for(auto const& name:{"steps","rock_steps","air_steps","surface_ambiguities","material_mismatches","deposited_energy_GeV"})
          require(summary[name].as<std::string>()==firstSummary[name].as<std::string>(),"parallel audit changed accounting");
        require(summary["device_geometry_audit"]["prepared_records"].as<unsigned>()>0,"parallel path not exercised");
      }
    }
    std::cout<<"PASS "<<scenario<<" serial/32/256 threads; exact output and first failure\n";
  }
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
