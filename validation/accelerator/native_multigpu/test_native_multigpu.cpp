#include "NativeCoordinator.hpp"
#include <arrow/api.h>
#include <algorithm>
#include <iterator>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <thread>
#include <chrono>
#include <sys/wait.h>
#include <unistd.h>
using namespace c8::multigpu;
static void check(bool ok,std::string const& msg) { if(!ok) throw std::runtime_error(msg); }
static void ok(arrow::Status const& s) { check(s.ok(),s.ToString()); }
template<class T> T take(arrow::Result<T> r) { ok(r.status()); return std::move(r).ValueOrDie(); }
static void text(fs::path const& p,std::string const& s) { std::ofstream out(p); out<<s; check(bool(out),"fixture write failed"); }
static Json json(fs::path const& p) { std::ifstream in(p); return Json::parse(in); }
static void writeTable(fs::path const& p,std::shared_ptr<arrow::Table> const& t) {
  fs::create_directories(p.parent_path());
  auto out=take(arrow::io::FileOutputStream::Open(p.string()));
  ok(parquet::arrow::WriteTable(*t,arrow::default_memory_pool(),out,2)); ok(out->Close());
}
static std::shared_ptr<arrow::Table> read(fs::path const& p) {
  auto in=take(arrow::io::ReadableFile::Open(p.string()));
  std::unique_ptr<parquet::arrow::FileReader> r; ok(parquet::arrow::OpenFile(in,arrow::default_memory_pool(),&r));
  std::shared_ptr<arrow::Table> t; ok(r->ReadTable(&t)); return t;
}
static std::shared_ptr<arrow::Array> doubles(std::vector<double> const& v) {
  arrow::DoubleBuilder b; ok(b.AppendValues(v)); return take(b.Finish());
}
static void fixture(fs::path const& p,double value,double timeOffset=0) {
  for(auto name:{"profile/profile.parquet","production_profile/profile.parquet","energyloss/dEdX.parquet",
                "CoREAS/observers.parquet","ZHS/observers.parquet"}) {
    bool radio=std::string(name).find("observers")!=std::string::npos;
    auto t=arrow::Table::Make(arrow::schema({arrow::field("shower",arrow::float64()),
      arrow::field(radio?"Time":"X",arrow::float64()),arrow::field(radio?"Ex":"total",arrow::float64())}),
      {doubles({0,0,0}),doubles({timeOffset,timeOffset+1,timeOffset+2}),doubles({value,-value,value})});
    writeTable(p/name,t); text((p/name).parent_path()/"config.yaml","type: fixture\n");
  }
  for(auto name:{"particles/particles.parquet","interactions/interactions.parquet"}) {
    writeTable(p/name,arrow::Table::Make(arrow::schema({arrow::field("pdg",arrow::float64())}),{doubles({11})}));
    text((p/name).parent_path()/"config.yaml","type: record\n");
  }
  fs::create_directories(p/"primary"); text(p/"primary/config.yaml","energy: 10\n"); text(p/"primary/summary.yaml","pdg: 22\n");
}
static int worker(int argc,char** argv) {
  std::string out,frontierIn,frontierOut,emBackend,stepfc; unsigned id=0;
  for(int i=1;i<argc;++i) {
    std::string a(argv[i]);
    if(a=="-f") out=argv[++i];
    else if(a=="--frontier-in") frontierIn=argv[++i];
    else if(a=="--frontier-out") frontierOut=argv[++i];
    else if(a=="--frontier-worker-id") id=std::stoul(argv[++i]);
    else if(a=="--em-backend") emBackend=argv[++i];
    else if(a=="--egs4-stepfc") stepfc=argv[++i];
  }
  if(std::getenv("C8_TEST_EXPECT_EGS4"))
    check(emBackend=="kokkos-egs4" && std::stod(stepfc)==.0625,
      "Both prefix and workers must receive EGS4 and its step factor");
  else
    check(emBackend==(frontierIn.empty()?"proposal":"kokkos-proposal"),
      "PROPOSAL prefix must stay scalar; workers must receive the renamed Kokkos backend");
  if(auto fail=std::getenv("C8_TEST_FAIL_WORKER")) {
    if(id==unsigned(std::stoul(fail))) return 42;
    if(id) std::this_thread::sleep_for(std::chrono::seconds(10));
  }
  if(std::getenv("C8_TEST_SLEEP") && id) std::this_thread::sleep_for(std::chrono::seconds(10));
  if(!frontierOut.empty()) text(frontierOut,
    "C8_STATIC_FRONTIER_V1\n22 1 0 0 0 8 2 0 0 0 1 0 0 -1\n"
    "11 2 1 1 9 4 4 3 0 0 1 0 0 -1\n22 3 1 1 9 2 1 0 0 0 1 0 0 -1\n22 4 1 1 9 1 1 0 0 0 1 0 0 -1\n");
  if(!frontierIn.empty()) {
    std::ifstream in(frontierIn); std::string line; std::getline(in,line); std::size_t count=0;
    while(std::getline(in,line)) ++count;
    if(std::getenv("C8_TEST_BAD_FEED")) ++count;
    save("FRONTIER_FEED.json",{{"mode","bounded-high-energy-first"},{"roots_consumed",count},
      {"batches",1},{"maximum_imported_batch",count}});
  }
  fixture(out,id?double(id):-1.);
  return 0;
}
template<class F> static void rejects(F f,std::string const& name) {
  bool rejected=false; try { f(); } catch(std::exception const&) { rejected=true; }
  check(rejected,"Not rejected: "+name);
}
static double first(fs::path const& p) {
  return std::static_pointer_cast<arrow::DoubleArray>(read(p)->column(2)->chunk(0))->Value(0);
}
int main(int argc,char** argv) {
  try {
    if(argc>1 && std::string(argv[1]).rfind("--query-gpu",0)==0) {
      std::cout<<"0, GPU-test-0\n1, GPU-test-1\n2, GPU-test-2\n3, GPU-test-3\n"; return 0;
    }
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--frontier-in" || std::string(argv[i])=="--frontier-out")
      return worker(argc,argv);
    if(argc>3 && std::string(argv[1])=="--merge") {
      std::vector<fs::path> parts; for(int i=3;i<argc;++i) parts.emplace_back(argv[i]);
      std::cout<<merge(parts,argv[2]).dump(2)<<'\n'; return 0;
    }
    char tmp[]="/tmp/c8-native-tests-XXXXXX"; auto directory=mkdtemp(tmp); check(directory,"mkdtemp failed"); fs::path root(directory);
    std::cout<<"Test artifacts: "<<root<<'\n';
    text(root/"abc","abc"); check(digest(root/"abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA256");
    text(root/"roots","C8_STATIC_FRONTIER_V1\n22 1 0 0 0 8 2 0 0 0 1 0 0 -1\n11 2 1 1 9 2 4 3 0 0 1 0 0 -1\n");
    auto a=partition(root/"roots",{root/"s0",root/"s1",root/"s2",root/"s3"});
    check(a["counts"]==Json({1,1,0,0}) && a["weighted_energy_GeV"]==24,"Partition weights/empty workers");
    text(root/"duplicate","C8_STATIC_FRONTIER_V1\n22 1 0 0 0 8 2 0 0 0 1 0 0 -1\n22 1 0 0 0 8 2 0 0 0 1 0 0 -1\n");
    rejects([&]{partition(root/"duplicate",{root/"dup0"});},"duplicate roots");
    fixture(root/"a",3); fixture(root/"b",-2); fixture(root/"c",4);
    auto m=merge({root/"a",root/"b",root/"c"},root/"merge");
    check(first(root/"merge/CoREAS/observers.parquet")==5,"Signed fields must be summed before power");
    check(m["particles/particles.parquet"]["rows"]==3 && m["interactions/interactions.parquet"]["rows"]==1,"Record merge");
    fixture(root/"nan",std::numeric_limits<double>::quiet_NaN());
    rejects([&]{validateOutputs(root/"nan");},"NaN validation");
    rejects([&]{merge({root/"nan"},root/"badnan");},"NaN merge");
    fixture(root/"shift",1,1);
    rejects([&]{merge({root/"a",root/"shift"},root/"badgrid");},"different grid");
    text(root/"b/CoREAS/config.yaml","type: different\n");
    rejects([&]{merge({root/"a",root/"b"},root/"badconfig");},"different observer configuration");
    fs::create_directory(root/"path"); fs::create_symlink(fs::canonical("/proc/self/exe"),root/"path/nvidia-smi");
    std::string path=(root/"path").string()+":"+(getenv("PATH")?getenv("PATH"):""); setenv("PATH",path.c_str(),1);
    Options o; o.worker=fs::canonical("/proc/self/exe"); o.output=root/"four"; o.devices={"0","1","2","3"};
    o.energy=10; o.seed=42; o.memoryFraction=.9; o.timeout=5;
    installSignalHandlers(); run(o);
    check(first(root/"four/merged/CoREAS/observers.parquet")==9,"Four worker sum plus prefix");
    check(json(root/"four/TIMING.json")["parts"].size()==5,"Prefix and four worker times");
    rejects([&]{run(o);},"output reuse");
    o.output=root/"egs4";o.emBackend="kokkos-egs4";o.egs4Stepfc=.0625;
    setenv("C8_TEST_EXPECT_EGS4","1",1);run(o);unsetenv("C8_TEST_EXPECT_EGS4");
    check(first(o.output/"merged/CoREAS/observers.parquet")==9,"EGS4 worker merge");
    check(json(o.output/"CONFIG.json")["em_backend"]=="kokkos-egs4","EGS4 provenance");
    o.emBackend="kokkos-proposal";
    o.events=2; o.output=root/"two_events"; run(o);
    check(fs::exists(o.output/"seed_42/COMPLETE.json") && fs::exists(o.output/"seed_43/COMPLETE.json"),"Two sequential showers");
    o.events=1; o.devices={"0","GPU-test-0"}; o.output=root/"aliases";
    rejects([&]{run(o);},"duplicate physical devices");
    o.devices={"0","1","2","3"}; o.output=root/"failure"; setenv("C8_TEST_FAIL_WORKER","2",1);
    auto start=std::chrono::steady_clock::now(); rejects([&]{run(o);},"failed worker"); unsetenv("C8_TEST_FAIL_WORKER");
    check(!fs::exists(o.output/"COMPLETE.json"),"Failure completion marker");
    check(std::chrono::steady_clock::now()-start<std::chrono::seconds(8),"Failure must stop other workers");
    check(waitpid(-1,nullptr,WNOHANG)==-1 && errno==ECHILD,"Leaked child after failure");
    o.output=root/"badfeed"; setenv("C8_TEST_BAD_FEED","1",1);
    rejects([&]{run(o);},"unconsumed frontier"); unsetenv("C8_TEST_BAD_FEED");
    o.output=root/"timeout"; o.timeout=.2; setenv("C8_TEST_SLEEP","1",1);
    rejects([&]{run(o);},"worker timeout"); unsetenv("C8_TEST_SLEEP");
    check(waitpid(-1,nullptr,WNOHANG)==-1 && errno==ECHILD,"Leaked child after timeout");
    check(!fs::exists(o.output/"COMPLETE.json"),"Timeout completion marker");
    o.output=root/"interrupt"; o.timeout=5; setenv("C8_TEST_SLEEP","1",1);
    std::thread interrupt([] {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      kill(getpid(),SIGTERM);
    });
    bool stopped=false;
    try { run(o); } catch(std::exception const&) { stopped=true; }
    interrupt.join(); unsetenv("C8_TEST_SLEEP"); installSignalHandlers();
    check(stopped && fs::exists(o.output/"INCOMPLETE.json") &&
      !fs::exists(o.output/"COMPLETE.json"),"Interrupted run must stay incomplete");
    check(waitpid(-1,nullptr,WNOHANG)==-1 && errno==ECHILD,"Leaked child after interruption");

    // Canonical comma/space forms and the compatibility alias share one route.
    if(argc>1) {
      std::vector<std::vector<std::string>> selections{
        {"--device","0,1,2,3"}, {"--device","0","1","2","3"},
        {"--devices","0,1,2,3"}};
      for(std::size_t test=0;test<selections.size();++test) {
        auto output=root/("device CLI "+std::to_string(test));
        std::vector<std::string> args{fs::absolute(argv[1]).string()};
        args.insert(args.end(),selections[test].begin(),selections[test].end());
        if(test==1) args.insert(args.end(),{"--em-backend","kokkos-proposal"});
        args.insert(args.end(),{"--multigpu-worker",o.worker.string(),
          "-E","10","-N","2","-s","101","-p","22","--ring","1","-f",output.string()});
        auto child=fork(); check(child>=0,"fork device CLI");
        if(child==0) {
          std::vector<char*> raw;
          for(auto& arg:args) raw.push_back(arg.data());
          raw.push_back(nullptr); execv(raw[0],raw.data()); _exit(127);
        }
        int status=0; waitpid(child,&status,0);
        check(WIFEXITED(status) && WEXITSTATUS(status)==0,"Unified device CLI");
        check(fs::exists(output/"seed_102/COMPLETE.json"),"Unified CLI event count");
        check(first(output/"seed_101/merged/CoREAS/observers.parquet")==9,
          "Unified CLI four worker merge");
        for(unsigned i=0;i<4;++i) {
          auto command=json(output/"seed_101"/("worker_"+std::to_string(i))/"COMMAND.json").get<std::vector<std::string>>();
          check(std::find(command.begin(),command.end(),"--device")==command.end() &&
                std::find(command.begin(),command.end(),"--devices")==command.end(),
                "Public device selector leaked into masked worker");
          auto option=std::find(command.begin(),command.end(),"--kokkos-device");
          check(option!=command.end() && std::next(option)!=command.end() && *std::next(option)=="0",
                "Worker must use ordinal zero inside UUID mask");
        }
      }
    }
    // Exercise public CLI forwarding with a path containing spaces, without any Python.
    if(argc>1) {
      auto child=fork(); check(child>=0,"fork CLI");
      if(child==0) {
        auto bin=fs::absolute(argv[1]).string(),out=(root/"CLI with spaces").string(),w=o.worker.string();
        execl(bin.c_str(),bin.c_str(),"--devices","0,1,2,3","--multigpu-worker",w.c_str(),
          "-E","10","-N","2","-s","101","-p","22","--ring","1","-f",out.c_str(),nullptr); _exit(127);
      }
      int status=0; waitpid(child,&status,0); check(WIFEXITED(status) && WEXITSTATUS(status)==0,"Public CLI");
      check(fs::exists(root/"CLI with spaces/seed_102/COMPLETE.json"),"CLI event count");
    }
    std::cout<<"All native coordination tests passed\n"; return 0;
  } catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
