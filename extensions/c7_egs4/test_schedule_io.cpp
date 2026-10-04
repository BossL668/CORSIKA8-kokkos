#include "Egs4MultiGpuIO.hpp"
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <fstream>
#include <iostream>
#include <cstdlib>
namespace c7_egs4::multigpu {void checkInterrupt(){}}
namespace {
using namespace c7_egs4::multigpu;
void check(bool x,char const* m){if(!x)throw std::runtime_error(m);}
template<class T>T take(arrow::Result<T> v){check(v.ok(),"Arrow operation failed");return std::move(v).ValueOrDie();}
void ok(arrow::Status const& s){if(!s.ok())throw std::runtime_error(s.ToString());}
template<class F>void rejects(F f){bool rejected=false;try{f();}catch(std::exception const&){rejected=true;}check(rejected,"Expected input rejection");}
void fixture(fs::path const& root,double field,double shift=0.,bool empty_zhs=false) {
  for(auto const* name:{"profile/profile.parquet","energyloss/dEdX.parquet","CoREAS/observers.parquet","ZHS/observers.parquet","particles/particles.parquet"}) {
    auto path=root/name;fs::create_directories(path.parent_path());
    auto group=fs::path(name).begin()->string();bool radio=group=="CoREAS"||group=="ZHS";
    std::ofstream(path.parent_path()/"config.yaml")<<(group=="ZHS"&&empty_zhs?"algorithm: ZHS\nobservers: {}\n":"grid: fixed\n");
    arrow::DoubleBuilder x,v;arrow::Int32Builder event;
    if(!(group=="ZHS"&&empty_zhs)) {
      ok(x.AppendValues({shift,1.+shift}));ok(v.AppendValues({field,-field}));ok(event.AppendValues({0,0}));
    }
    auto data=arrow::Table::Make(arrow::schema({arrow::field("shower",arrow::int32()),arrow::field(radio?"Time":"X",arrow::float64()),
      arrow::field("value",arrow::float64())}),{take(event.Finish()),take(x.Finish()),take(v.Finish())});
    auto stream=take(arrow::io::FileOutputStream::Open(path.string()));ok(parquet::arrow::WriteTable(*data,arrow::default_memory_pool(),stream,10));ok(stream->Close());
  }
}
}
int main(){try {
  char path[]="/tmp/egs4-schedule-io-XXXXXX";check(mkdtemp(path)!=nullptr,"Temporary directory failed");fs::path root(path);
  fixture(root/"a",1.);fixture(root/"b",-1.);fixture(root/"bad_grid",1.,.001);
  validateOutputs(root/"a");merge({root/"a",root/"b"},root/"sum");validateOutputs(root/"sum");
  auto stream=take(arrow::io::ReadableFile::Open((root/"sum/CoREAS/observers.parquet").string()));
  std::unique_ptr<parquet::arrow::FileReader> reader;ok(parquet::arrow::OpenFile(stream,arrow::default_memory_pool(),&reader));
  std::shared_ptr<arrow::Table> result;ok(reader->ReadTable(&result));
  auto values=std::static_pointer_cast<arrow::DoubleArray>(result->GetColumnByName("value")->chunk(0));
  check(values->Value(0)==0.&&values->Value(1)==0.,"Merged power instead of signed electric field");
  rejects([&]{merge({root/"a",root/"a"},root/"duplicate");});
  rejects([&]{merge({root/"a",root/"bad_grid"},root/"wrong_grid");});
  check(!fs::exists(root/"wrong_grid/MERGE.json"),"Bad merge published success");
  auto frontier=root/"roots.txt";
  std::string row="22 1 0 0 0 1 2 0 0 0 2000 0 0 -1\n";
  {std::ofstream f(frontier);f<<"C8_STATIC_FRONTIER_V1\n"<<row<<row;}
  rejects([&]{partition(frontier,{root/"duplicate_roots.txt"});});
  {std::ofstream f(frontier);f<<"C8_STATIC_FRONTIER_V1\n22 1 0 0 1 1 2 0 0 0 2000 0 0 -1\n";}
  rejects([&]{partition(frontier,{root/"continued_root.txt"});});
  {std::ofstream f(frontier);f<<"C8_STATIC_FRONTIER_V1\n"<<row;}
  auto split=partition(frontier,{root/"s1.txt",root/"s2.txt",root/"s3.txt",root/"s4.txt"});
  check(split.at("roots")==1&&split.at("weighted_energy_GeV")==2.,"Partition lost weight");
  check(split.at("counts")==Json::array({1,0,0,0}),"Empty shards failed");
  {std::ofstream f(frontier);f<<"C8_STATIC_FRONTIER_V1\n13 7 0 0 0 10 2 0 0 0 2000 0 0 -1\n";}
  rejects([&]{partition(frontier,{root/"muon_forbidden.txt"});});
  auto muons=partition(frontier,{root/"muon_enabled.txt"},true);
  check(muons.at("roots")==1&&muons.at("weighted_energy_GeV")==20.,"Muon partition lost identity/energy");
  {std::ofstream f(frontier);f<<"C8_STATIC_FRONTIER_V1\n13 7 0 0 1 10 2 0 0 0 2000 0 0 -1\n";}
  rejects([&]{partition(frontier,{root/"muon_continuation.txt"},true);});
  fixture(root/"coreas_only_a",1.,0.,true);fixture(root/"coreas_only_b",-1.,0.,true);
  validateOutputs(root/"coreas_only_a");
  merge({root/"coreas_only_a",root/"coreas_only_b"},root/"coreas_only_sum");validateOutputs(root/"coreas_only_sum");
  std::ofstream(root/"coreas_only_a/ZHS/config.yaml")<<"algorithm: ZHS\nobservers: {antenna: {}}\n";
  rejects([&]{validateOutputs(root/"coreas_only_a");});
  std::cout<<"PASS signed-field cancellation, grid/duplicate rejection, fresh-root/weight conservation; retained "<<root<<'\n';
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
