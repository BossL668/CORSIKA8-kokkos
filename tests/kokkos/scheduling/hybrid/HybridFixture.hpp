/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <corsika/kokkos/scheduling/HybridCascade.hpp>
#include <corsika/kokkos/scheduling/FlukaInteractionDeferral.hpp>
#include "ProductionHybridCascade.hpp"
#include "ProductionInteractionCounter.hpp"
#include "ScalarStepFixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

namespace hybrid_fixture {
  using namespace corsika;
  namespace old = corsika::production_hybrid_reference;
  namespace cp = corsika::kokkos;
  using Stack = scalar_fixture::Stack;

  struct Result {
    scalar_fixture::Trace trace_;
    std::vector<std::array<std::uint64_t, 4>> identities_;
    std::vector<std::uint64_t> counts_;
    std::map<std::int32_t, std::uint64_t> scalarCounts_;
    std::vector<std::array<std::uint64_t, 3>> fingerprints_;
    std::vector<std::vector<double>> loads_;
    std::vector<std::array<double, 5>> classes_;
    std::vector<std::uint64_t> workerBatches_, workerRequests_;
    std::vector<std::array<std::uint64_t, 4>> commits_;
    unsigned synchronousCalls_ = 0;
    unsigned countedInteractions_ = 0;
  };

  inline void capture(Stack& stack, scalar_fixture::Node const* node, Result& result) {
    scalar_fixture::capture(stack, node, result.trace_);
    for (auto const& p : stack)
      result.identities_.push_back(
          {p.getHistoryId(), p.getParentHistoryId(), p.getGeneration(), p.getStepId()});
  }

  struct Output {
    scalar_fixture::Trace& trace_;
    void startOfShower() { trace_.calls_.push_back("output-start"); }
    void endOfShower() { trace_.calls_.push_back("output-end"); }
  };

  template <bool Old>
  struct Model {
    using worker_model_type =
        std::conditional_t<Old, old::HadronicWorkerModel, cp::HadronicWorkerModel>;
    static constexpr auto hadronic_worker_model = worker_model_type::Fluka;
    unsigned calls_ = 0;
    template <typename View>
    void doInteraction(View& view, Code, Code, FourMomentum const& p4,
                       FourMomentum const&) {
      ++calls_;
      auto const cs = p4.getSpaceLikeComponents().getCoordinateSystem();
      view.addSecondary(
          std::make_tuple(Code::Photon, 2_GeV, DirectionVector(cs, {0., 0., 1.})));
    }
    CrossSectionType getCrossSection(Code, Code, FourMomentum const&,
                                     FourMomentum const&) const {
      return 1.e12_mb;
    }
  };

  template <bool Old>
  using Counter =
      std::conditional_t<Old, old::InteractionCounter<Model<Old>>,
                         InteractionCounter<Model<Old>, cp::FlukaInteractionDeferral>>;

  template <bool Old>
  struct Processes : scalar_fixture::Processes {
    Counter<Old>& counter_;
    Result& result_;
    bool workers_ = false;
    bool router_ = false;
    Processes(scalar_fixture::Mode mode, scalar_fixture::Node const* node, Result& result,
              Counter<Old>& counter, bool workers, bool router)
        : scalar_fixture::Processes{mode, node, result.trace_}
        , counter_(counter)
        , result_(result)
        , workers_(workers)
        , router_(router) {}
    template <typename View, typename Rng>
    ProcessReturn selectInteraction(View& view, FourMomentum const& p4,
                                    NuclearComposition const& composition, Rng& rng,
                                    CrossSectionType sample) {
      if (!workers_)
        return scalar_fixture::Processes::selectInteraction(view, p4, composition, rng,
                                                            sample);
      trace_.calls_.push_back("worker-interaction");
      auto const cs = p4.getSpaceLikeComponents().getCoordinateSystem();
      FourMomentum target(get_mass(Code::Oxygen),
                          MomentumVector(cs, {0_GeV, 0_GeV, 0_GeV}));
      counter_.doInteraction(view, view.getProjectile().getPID(), Code::Oxygen, p4,
                             target);
      return ProcessReturn::Interacted;
    }
    template <typename View>
    void doSecondaries(View& view) {
      auto const parent = view.getProjectile();
      for (auto const& child : view)
        result_.commits_.push_back({parent.getHistoryId(), child.getHistoryId(),
                                    child.getParentHistoryId(), child.getGeneration()});
      scalar_fixture::Processes::doSecondaries(view);
    }
    void initCascadeEquations() { trace_.calls_.push_back("equations-init"); }
    template <typename TStack>
    void doCascadeEquations(TStack&) {
      trace_.calls_.push_back("equations-do");
    }
    void doStack(Stack& stack) {
      trace_.calls_.push_back("stack");
      capture(stack, node_, result_);
      for (auto& p : stack) {
        if (p.isErased()) continue;
        if (workers_) {
          // Controlled responses terminate here; pending projectiles must not be
          // compacted or discarded before the production coordinator commits them.
          if (p.getPID() != Code::Proton && p.getPID() != Code::PiPlus) p.erase();
        } else if (!router_ || p.getStepId() != 0) {
          p.erase();
        }
      }
    }
  };

  struct BasicRouter {
    using Particle = Stack::particle_type;
    scalar_fixture::Trace& trace_;
    std::vector<transport::TransportIdentity> staged_;
    explicit BasicRouter(scalar_fixture::Trace& trace)
        : trace_(trace) {}
    bool canRoute(Particle const& particle, transport::StepId step) {
      trace_.calls_.push_back("can-route");
      return step == 0 &&
             (particle.getPID() == Code::Photon || particle.getPID() == Code::Electron ||
              particle.getPID() == Code::MuMinus);
    }
    void stage(Particle const& particle, transport::HistoryId history,
               transport::HistoryId parent, transport::Generation generation,
               transport::StepId step) {
      trace_.calls_.push_back("stage");
      REQUIRE(particle.getHistoryId() == history);
      staged_.push_back({history, parent, generation, step});
    }
    bool pending() const { return !staged_.empty(); }
    std::size_t advanceOneWavefrontAndReturn(Stack&) {
      trace_.calls_.push_back("router-advance");
      staged_.clear();
      return 0;
    }
  };

  struct Router : BasicRouter {
    CoordinateSystemPtr coordinates_;
    scalar_fixture::Node const* node_;
    bool interleave_, returnParticles_, forceDecay_, preserve_;
    std::set<transport::HistoryId> returned_;
    Router(scalar_fixture::Trace& trace, CoordinateSystemPtr coordinates,
           scalar_fixture::Node const* node, bool interleave, bool returns, bool decay,
           bool preserve)
        : BasicRouter(trace)
        , coordinates_(coordinates)
        , node_(node)
        , interleave_(interleave)
        , returnParticles_(returns)
        , forceDecay_(decay)
        , preserve_(preserve) {}
    bool preserveLogicalNodes() const { return preserve_; }
    bool readyForScalarInterleave() const { return interleave_ && pending(); }
    bool consumeForcedDecay(transport::HistoryId history, transport::StepId) {
      trace_.calls_.push_back("consume-decay");
      return forceDecay_ && returned_.erase(history) != 0;
    }
    std::size_t advanceOneWavefrontAndReturn(Stack& stack) {
      trace_.calls_.push_back("router-advance");
      std::size_t returned = 0;
      if (returnParticles_) {
        for (auto identity : staged_) {
          auto particle = stack.addParticle(std::make_tuple(
              Code::MuMinus, 10_GeV, DirectionVector(coordinates_, {0., 0., 1.}),
              Point(coordinates_, 0_m, 0_m, 0_m), 0_s));
          identity.stepId_ += 1;
          particle.setTransportIdentity(identity);
          particle.setNode(node_);
          // One controlled forced decay is enough to exercise the hook. Its
          // secondary must be absorbed rather than creating an endless mock chain.
          if (identity.generation_ == 0) returned_.insert(identity.historyId_);
          ++returned;
        }
      }
      staged_.clear();
      return returned;
    }
    void endOfShower() { trace_.calls_.push_back("router-end"); }
  };

  template <bool Old>
  struct Implementation;
  template <>
  struct Implementation<false> {
    template <typename Tracking, typename Process, typename Out, typename TStack,
              typename Em = cp::DisabledHybridEmRouter>
    using Cascade = cp::HybridCascade<Tracking, Process, Out, TStack, Em>;
    using Pool = cp::HadronicProcessPool;
    using Config = cp::HybridHadronicWorkerConfig;
  };
  template <>
  struct Implementation<true> {
    template <typename Tracking, typename Process, typename Out, typename TStack,
              typename Em = old::DisabledHybridEmRouter>
    using Cascade = old::HybridCascade<Tracking, Process, Out, TStack, Em>;
    using Pool = old::HadronicProcessPool;
    using Config = old::HybridHadronicWorkerConfig;
  };

  inline void reset_rng(std::uint64_t seed) {
    auto& manager = RNGManager<>::getInstance();
    if (!manager.isRegistered("cascade")) manager.registerRandomStream("cascade");
    manager.setSeed(seed);
    manager.getRandomStream("cascade").reset();
  }

  inline void compare(Result const& current, Result const& production) {
    CHECK(current.trace_.calls_ == production.trace_.calls_);
    CHECK(current.trace_.arguments_ == production.trace_.arguments_);
    CHECK(current.trace_.particles_ == production.trace_.particles_);
    CHECK(current.trace_.events_ == production.trace_.events_);
    CHECK(current.trace_.rng_ == production.trace_.rng_);
    CHECK(current.trace_.threw_ == production.trace_.threw_);
    CHECK(current.identities_ == production.identities_);
    CHECK(current.counts_ == production.counts_);
    CHECK(current.scalarCounts_ == production.scalarCounts_);
    CHECK(current.fingerprints_ == production.fingerprints_);
    CHECK(current.loads_ == production.loads_);
    CHECK(current.classes_ == production.classes_);
    CHECK(current.workerBatches_ == production.workerBatches_);
    CHECK(current.workerRequests_ == production.workerRequests_);
    CHECK(current.commits_ == production.commits_);
    CHECK(current.synchronousCalls_ == production.synchronousCalls_);
    CHECK(current.countedInteractions_ == production.countedInteractions_);
  }
} // namespace hybrid_fixture
