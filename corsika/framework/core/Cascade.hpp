/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/corsika.hpp>

#include <corsika/framework/process/ProcessReturn.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/random/ExponentialDistribution.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/framework/random/UniformRealDistribution.hpp>
#include <corsika/framework/stack/SecondaryView.hpp>
#include <corsika/framework/geometry/FourVector.hpp>

#include <corsika/media/Environment.hpp>

#include <corsika/stack/history/HistoryStackExtension.hpp>

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

namespace corsika {

  /**
   *
   * The Cascade class is constructed from template arguments making
   * it very versatile. Via the template arguments physics models are
   * plugged into the cascade simulation.
   *
   * <b>TTracking</b> must be a class according to the
   * TrackingInterface providing the functions:
   *
   * @code
   * auto getTrack(particle_type const& p)</auto>,
   * with the return type <code>geometry::Trajectory<Line>
   * @endcode
   *
   * <b>TProcessList</b> must be a ProcessSequence.   *
   * <b>Stack</b> is the storage object for particle data, i.e. with
   * particle class type `Stack::particle_type`.
   */
  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack>
  class Cascade {

    typedef typename TStack::stack_view_type stack_view_type;

    typedef typename TStack::particle_type particle_type;

    typedef std::remove_pointer_t<decltype(std::declval<particle_type>().getNode())>
        volume_tree_node_type;

    typedef typename volume_tree_node_type::IModelProperties medium_interface_type;

  public:
    /**
     * @name constructors
     * @{
     * Cascade class cannot be default constructed, but needs a valid
     * list of physics processes for configuration at construct time.
     */
    Cascade() = delete;
    Cascade(Cascade const&) = default;
    Cascade(Cascade&&) = default;
    ~Cascade() = default;
    Cascade& operator=(Cascade const&) = default;
    Cascade(Environment<medium_interface_type> const& env, TTracking& tr,
            TProcessList& pl, TOutput& out, TStack& stack);
    //! @}

    /**
     * set the nodes for all particles on the stack according to their numerical
     * position.
     */
    void setNodes();

    /**
     * The Run function is the main simulation loop, which processes
     * particles from the Stack until the Stack is empty.
     */
    void run();

    /**
     * Force an interaction of the top particle of the stack at its current position.
     * Note that setNodes() or an equivalent procedure needs to be called first if you
     * want to call forceInteraction() for the primary interaction.
     * Incompatible with forceDecay()
     */
    void forceInteraction();

    /**
     * Force an decay of the top particle of the stack at its current position.
     * Note that setNodes() or an equivalent procedure needs to be called first if you
     * want to call forceDecay() for the primary interaction.
     * Incompatible with forceInteraction()
     */
    void forceDecay();

  private:
    /**
     * The Step function is executed for each particle from the
     * stack. It will calcualte geometric transport of the particles,
     * and apply continuous and stochastic processes to it, which may
     * lead to energy losses, scattering, absorption, decays and the
     * production of secondary particles.
     *
     * New particles produced in one step are subject to further
     * processing, e.g. thinning, etc.
     */
    void step(particle_type& vParticle);

    ProcessReturn decay(stack_view_type& view, InverseTimeType initial_inv_decay_time);
    ProcessReturn interaction(stack_view_type& view, FourMomentum const& projectileP4,
                              NuclearComposition const& composition,
                              CrossSectionType const initial_cross_section);
    void setEventType(stack_view_type& view, history::EventType);

    // data members
    Environment<medium_interface_type> const& environment_;
    TTracking& tracking_;
    TProcessList& sequence_;
    TOutput& output_;
    TStack& stack_;
    default_prng_type& rng_ = RNGManager<>::getInstance().getRandomStream("cascade");
    bool forceInteraction_;
    bool forceDecay_;
    unsigned int count_ = 0;

    // but this here temporarily. Should go into dedicated file later:
    const char* c8_ascii_ =
        R"V0G0N(
  ,ad8888ba,     ,ad8888ba,    88888888ba    ad88888ba   88  88      a8P          db              ad88888ba   
 d8"'    `"8b   d8"'    `"8b   88      "8b  d8"     "8b  88  88    ,88'          d88b            d8"     "8b  
d8'            d8'        `8b  88      ,8P  Y8,          88  88  ,88"           d8'`8b           Y8a     a8P  
88             88          88  88aaaaaa8P'  `Y8aaaaa,    88  88,d88'           d8'  `8b           "Y8aaa8P"   
88             88          88  88""""88'      `"""""8b,  88  8888"88,         d8YaaaaY8b          ,d8"""8b,   
Y8,            Y8,        ,8P  88    `8b            `8b  88  88P   Y8b       d8""""""""8b        d8"     "8b  
 Y8a.    .a8P   Y8a.    .a8P   88     `8b   Y8a     a8P  88  88     "88,    d8'        `8b       Y8a     a8P  
  `"Y8888Y"'     `"Y8888Y"'    88      `8b   "Y88888P"   88  88       Y8b  d8'          `8b       "Y88888P"
       )V0G0N";
  };

} // namespace corsika

#include <corsika/detail/framework/core/Cascade.inl>
