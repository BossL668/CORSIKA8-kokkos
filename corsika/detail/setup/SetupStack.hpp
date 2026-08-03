/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/stack/CombinedStack.hpp>
#include <corsika/stack/GeometryNodeStackExtension.hpp>
#include <corsika/stack/TransportIdentityStackExtension.hpp>
#include <corsika/stack/VectorStack.hpp>
#include <corsika/stack/WeightStackExtension.hpp>
#include <corsika/stack/history/HistorySecondaryProducer.hpp>
#include <corsika/stack/history/HistoryStackExtension.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/IMediumModel.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>

namespace corsika {

  namespace setup::detail {
    template <typename TEnvironment>
    class StackGenerator {
    private:
      using env_type = TEnvironment;

      // ------------------------------------------
      // add geometry node data to stack. This is fundamentally needed
      // for robust tracking through multiple volumes.

      // the GeometryNode stack needs to know the type of geometry-nodes from the
      // environment:

      template <typename TStackIter>
      using SetupGeometryDataInterface =
          typename node::MakeGeometryDataInterface<TStackIter, env_type>::type;

      // combine particle data stack with geometry information for tracking
      template <typename TStackIter>
      using StackWithGeometryInterface =
          CombinedParticleInterface<VectorStack::pi_type, SetupGeometryDataInterface,
                                    TStackIter>;

      using StackWithGeometry =
          CombinedStack<typename VectorStack::stack_data_type,
                        node::GeometryData<env_type>, StackWithGeometryInterface,
                        DefaultSecondaryProducer>;

      template <class T>
      using StackWithGeometry_PI_type = typename StackWithGeometry::template pi_type<T>;

      // ------------------------------------------
      // add weight data to stack. This is fundamentally needed
      // for thinning.

      // the "pure" weight stack (interface)
      template <typename TStackIter>
      using SetupWeightDataInterface =
          typename weights::MakeWeightDataInterface<TStackIter>::type;

      // combine geometry-node-vector data stack with weight information for tracking
      template <typename TStackIter>
      using StackWithWeightInterface =
          CombinedParticleInterface<StackWithGeometry_PI_type, SetupWeightDataInterface,
                                    TStackIter>;

    public:
      // the combined stack data: particle + geometry + weight
      using StackWithWeight =
          CombinedStack<typename StackWithGeometry::stack_data_type, weights::WeightData,
                        StackWithWeightInterface, DefaultSecondaryProducer>;

    private:
      template <typename T>
      using StackWithWeight_PI_type = typename StackWithWeight::template pi_type<T>;

      // ------------------------------------------
      // Add stable scheduler-facing identities. This stack is opt-in so that the
      // existing setup::Stack type and CPU application behavior remain unchanged.

      template <typename TStackIter>
      using SetupTransportIdentityDataInterface =
          typename transport::MakeTransportIdentityDataInterface<TStackIter>::type;

      template <typename TStackIter>
      using StackWithTransportIdentityInterface =
          CombinedParticleInterface<StackWithWeight_PI_type,
                                    SetupTransportIdentityDataInterface, TStackIter>;

    public:
      using StackWithTransportIdentity =
          CombinedStack<typename StackWithWeight::stack_data_type,
                        transport::TransportIdentityData,
                        StackWithTransportIdentityInterface, DefaultSecondaryProducer>;

    private:
      template <typename T>
      using StackWithTransportIdentity_PI_type =
          typename StackWithTransportIdentity::template pi_type<T>;

      // ------------------------------------------
      // Add [OPTIONAL] history data to stack, too.
      // This keeps the entire lineage of particles in memory.

      template <typename TStackIter>
      using StackWithHistoryInterface =
          CombinedParticleInterface<StackWithWeight_PI_type,
                                    history::HistoryEventDataInterface, TStackIter>;

    public:
      using StackWithHistory =
          CombinedStack<typename StackWithWeight::stack_data_type,
                        history::HistoryEventData, StackWithHistoryInterface,
                        history::HistorySecondaryProducer>;

    private:
      template <typename TStackIter>
      using StackWithTransportIdentityAndHistoryInterface =
          CombinedParticleInterface<StackWithTransportIdentity_PI_type,
                                    history::HistoryEventDataInterface, TStackIter>;

    public:
      using StackWithTransportIdentityAndHistory =
          CombinedStack<typename StackWithTransportIdentity::stack_data_type,
                        history::HistoryEventData,
                        StackWithTransportIdentityAndHistoryInterface,
                        history::HistorySecondaryProducer>;
    };

  } // namespace setup::detail

} // namespace corsika
