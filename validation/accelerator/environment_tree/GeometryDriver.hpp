#pragma once
#include <corsika/accelerator/em/common/EnvironmentTreeSnapshot.hpp>
#include <corsika/accelerator/em/common/EnvironmentModelSnapshot.hpp>
#include <corsika/accelerator/em/common/EnvironmentCurvedSnapshot.hpp>
#include <corsika/accelerator/em/common/Types.hpp>
#include <vector>
struct Query { double p[3],v[3],limit; corsika::gpu::em::environment_tree::NodeId node; };
struct Result { double distance; corsika::gpu::em::environment_tree::NodeId node,located; };
struct DeviceResult {
  std::vector<Result> results;
  std::vector<double> kernel_ms;
  double upload_ms{},download_ms{};
};
void initializeDevice(int argc,char** argv);
void finalizeDevice();
char const* deviceName();
DeviceResult runDeviceGeometry(
    std::vector<corsika::gpu::em::environment_tree::Node> const&,
    std::vector<corsika::gpu::em::environment_tree::NodeId> const&,
    std::vector<corsika::gpu::em::environment_tree::NodeId> const&,
    std::vector<Query> const&);
struct ModelQuery {int id;double p[3],direction[3],length,grammage;};
struct ModelResult {double rho,index,grammage,length;};
std::vector<ModelResult> runDeviceModels(
    std::vector<corsika::gpu::em::environment_tree::Model> const&,std::vector<ModelQuery> const&,
    corsika::gpu::em::environment_tree::FullView full={});
bool testNodeQueue();
struct LeptonSequenceResult {
  unsigned count{},fallback{};
  corsika::gpu::em::ProposalFallbackEvent failure{};
  corsika::gpu::em::LeptonTransportRecord records[32]{};
};
std::vector<LeptonSequenceResult> runLeptonSequences(
    corsika::gpu::em::environment_tree::FullView,
    std::vector<corsika::gpu::em::EmParticleState> const&);
struct PhotonSequenceResult {
  unsigned count{},fallback{};
  corsika::gpu::em::PhotonTransportRecord records[16]{};
};
std::vector<PhotonSequenceResult> runPhotonSequences(
    corsika::gpu::em::environment_tree::FullView,
    std::vector<corsika::gpu::em::EmParticleState> const&);
struct CurvedQuery {
  corsika::gpu::em::environment_tree::CurvedAccess track;
  int node;
  double limit;
};
struct CurvedResult {Result result;bool unsupported,incomplete;};
struct CurvedDeviceResult {
  std::vector<CurvedResult> results;
  std::vector<double> kernel_ms;
};
CurvedDeviceResult runDeviceCurved(
    std::vector<corsika::gpu::em::environment_tree::Node> const&,
    std::vector<corsika::gpu::em::environment_tree::NodeId> const&,
    std::vector<corsika::gpu::em::environment_tree::NodeId> const&,
    std::vector<CurvedQuery> const&);
