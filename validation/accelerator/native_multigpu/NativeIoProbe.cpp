#include "../../../applications/detail/air_shower_multigpu/NativeCoordinator.hpp"
#include <iostream>

int main(int argc, char** argv) {
  using namespace c8::multigpu;
  if(argc != 4) return 2;
  try {
    fs::path input=argv[1], output=argv[2];
    if(!fs::create_directory(output)) throw std::runtime_error("Output exists");
    if(std::string(argv[3])=="validate") {
      std::cout<<"Validating closed prefix outputs"<<std::endl;
      validateOutputs(input/"prefix/shower");
      std::cout<<"Validation passed"<<std::endl;
    }
    std::vector<fs::path> shards;
    for(unsigned i=0;i<4;++i) shards.push_back(output/("roots_"+std::to_string(i)+".txt"));
    std::cout<<"Partitioning existing frontier"<<std::endl;
    auto audit=partition(input/"frontier.txt",shards);
    save(output/"PARTITION.json",audit);
    std::cout<<"Partition passed: "<<audit["roots"]<<std::endl;
    return 0;
  } catch(std::exception const& e) {
    std::cerr<<e.what()<<std::endl;return 1;
  }
}
