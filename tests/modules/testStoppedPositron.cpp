#include <corsika/modules/ParticleCut.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <SetupTestStack.hpp>
#include <SetupTestTrajectory.hpp>
#include <corsika/setup/SetupTrajectory.hpp>
#include <catch2/catch_all.hpp>
using namespace corsika;

namespace {
  // Intentionally has no addSecondary(): a disabled physics branch must not
  // require a mutable stack interface, even for a positron below the cut.
  struct ReadOnlyCutParticle {
    Code pid;
    Point position;
    DirectionVector direction;
    Code getPID() const { return pid; }
    HEPEnergyType getKineticEnergy() const { return 0.2_MeV; }
    double getWeight() const { return 3.; }
    TimeType getTime() const { return 0_ns; }
    Point const& getPosition() const { return position; }
    DirectionVector const& getDirection() const { return direction; }
  };
}

TEST_CASE("Disabled stopped annihilation accepts read-only particles", "[stopped-positron]") {
  REQUIRE_FALSE(EnableStoppedPositronAnnihilation);
  auto cs = get_root_CoordinateSystem();
  Point origin(cs, 0_m, 0_m, 0_m);
  auto track = setup::testing::make_track<setup::Trajectory>(
      Line{origin, VelocityVector{cs, {0_m/second, 0_m/second, constants::c}}},
      1_m/constants::c);
  for (auto pid : {Code::Positron, Code::Electron, Code::Photon}) {
    ReadOnlyCutParticle particle{pid, origin, DirectionVector(cs, {0., 0., 1.})};
    Step step(particle, track);
    ParticleCut cut(0.25_MeV, 0.25_MeV, 1_GeV, 1_GeV, 1_GeV, false);
    REQUIRE(cut.doContinuous(step) == ProcessReturn::ParticleAbsorbed);
    CHECK(cut.statistics().stopped_positron_annihilations == 0);
    CHECK(cut.statistics().weighted_medium_rest_mass_input_GeV == 0.);
    CHECK(cut.statistics().weighted_kinetic_energy_GeV ==
          Catch::Approx(3. * 0.2_MeV / 1_GeV));
  }
}

TEST_CASE("Stopped positron endpoint and ledger", "[stopped-positron]") {
  REQUIRE_FALSE(EnableStoppedPositronAnnihilation);
  RNGManager<>::getInstance().registerRandomStream("cascade");
  auto cs = get_root_CoordinateSystem();
  auto const mass = get_mass(Code::Electron);
  Point origin(cs, 0_m, 0_m, 0_m);
  DirectionVector direction(cs, {0.,0.,1.});
  for (auto cutPhoton : {0.25_MeV, mass, 1_MeV}) {
    for (auto kinetic : {0_MeV, 0.2_MeV, 0.25_MeV}) {
      for (auto time : {0_ns, 10_ms, 10_ms+1_ns}) {
        ParticleCut cut(0.25_MeV,cutPhoton,1_GeV,1_GeV,1_GeV,false);
        CHECK(cut.getConfig()["stopped_positron_annihilation"].as<std::string>() ==
              "disabled_original_c8_cut");
        test::Stack stack;
        auto parent=stack.addParticle(std::make_tuple(Code::Photon,1_GeV,direction,origin,time));
        parent.setWeight(7.);
        test::StackView view(parent);
        view.getProjectile().addSecondary(std::make_tuple(Code::Positron,kinetic,direction));
        auto const shouldAnnihilate=EnableStoppedPositronAnnihilation &&
                                   kinetic<0.25_MeV && time<=10_ms;
        auto const shouldTransport=shouldAnnihilate && mass>=cutPhoton;
        cut.doSecondaries(view);
        CHECK(cut.statistics().stopped_positron_annihilations == (shouldAnnihilate?1:0));
        CHECK(cut.statistics().weighted_medium_rest_mass_input_GeV ==
              Catch::Approx(shouldAnnihilate?7.*mass/1_GeV:0.));
        if (shouldTransport) {
          REQUIRE(view.getEntries()==2);
          double sums[3]{};
          for(auto const& p:view){
            CHECK(p.getPID()==Code::Photon); CHECK(p.getWeight()==7.);
            CHECK(p.getEnergy()==mass); CHECK(p.getTime()==time);
            sums[0]+=p.getDirection().getX(cs);sums[1]+=p.getDirection().getY(cs);
            sums[2]+=p.getDirection().getZ(cs);
          }
          for(auto s:sums)CHECK(s==0.);
          CHECK(cut.statistics().weighted_kinetic_energy_GeV==Catch::Approx(7.*kinetic/1_GeV));
          CHECK(cut.statistics().weighted_rest_mass_energy_GeV==0.);
        } else if(shouldAnnihilate) {
          CHECK(view.getEntries()==0);
          CHECK(cut.statistics().weighted_kinetic_energy_GeV==Catch::Approx(7.*(kinetic+2.*mass)/1_GeV));
          CHECK(cut.statistics().weighted_rest_mass_energy_GeV==0.);
        } else {
          auto const removed = kinetic < 0.25_MeV || time > 10_ms;
          CHECK(view.getEntries()==(removed?0:1));
          CHECK(cut.statistics().weighted_kinetic_energy_GeV ==
                Catch::Approx(removed ? 7.*kinetic/1_GeV : 0.));
          CHECK(cut.statistics().weighted_rest_mass_energy_GeV ==
                Catch::Approx(removed ? 7.*mass/1_GeV : 0.));
        }
      }
    }
  }
}

TEST_CASE("Continuous positron cut adds no annihilation photons", "[stopped-positron]") {
  RNGManager<>::getInstance().registerRandomStream("cascade");
  auto cs=get_root_CoordinateSystem();
  Point origin(cs,0_m,0_m,0_m);
  test::Stack stack;
  ParticleCut cut(0.25_MeV,0.25_MeV,1_GeV,1_GeV,1_GeV,false);
  auto p=stack.addParticle(std::make_tuple(Code::Positron,0.2_MeV,
      DirectionVector(cs,{0.,0.,1.}),origin,0_ns));
  p.setWeight(3.);
  auto track=setup::testing::make_track<setup::Trajectory>(
      Line{origin,VelocityVector{cs,{0_m/second,0_m/second,constants::c}}},12_m/constants::c);
  Step step(p,track);
  REQUIRE(cut.doContinuous(step)==ProcessReturn::ParticleAbsorbed);
  p.erase();
  REQUIRE(stack.getEntries()==0);
  CHECK(cut.statistics().stopped_positron_annihilations==0);
  CHECK(cut.statistics().weighted_medium_rest_mass_input_GeV==0.);
  CHECK(cut.statistics().weighted_kinetic_energy_GeV==Catch::Approx(3.*0.2_MeV/1_GeV));
  CHECK(cut.statistics().weighted_rest_mass_energy_GeV==
        Catch::Approx(3.*get_mass(Code::Electron)/1_GeV));
}
