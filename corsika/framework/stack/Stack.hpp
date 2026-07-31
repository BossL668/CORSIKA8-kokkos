/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/stack/StackIteratorInterface.hpp>
#include <corsika/framework/stack/DefaultSecondaryProducer.hpp>
#include <corsika/framework/stack/SecondaryView.hpp>

#include <stdexcept>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>

/**
  @file Stack.hpp

  Description of particle stacks

  In the CORSIKA 8 framework particle data is always stored in
  particle stacks. A particle is, thus, always a reference (iterator)
  to an entry on a stack, e.g.

  \verbatim
  ModelStack stack;
  stack.begin(); // returns an iterator: StackIterator, ConstStackIterator

  *stack.begin(); // return a reference to ParticleInterfaceType, which is the class
provided by the user to read/write particle properties

  \endverbatim

  All functionality and algorithms for stack data access is located in the namespace
corsika::stack

  The minimal example of how to use this is shown in stack_example.cc
**/

namespace corsika {

  /**
     The Stack class provides (and connects) the main particle data storage machinery.

     The StackDataType type is the user-provided bare data storage
     object. This can be of any complexity, from a simple struct
     (fortran common block), to a combination of different and
     distributed data sources.

     The user-provided ParticleInterface template type is the base
     class type of the StackIteratorInterface class (CRTP) and must
     provide all functions to read single particle data from the
     StackDataType, given an 'unsigned int' index.

     <b>Important:</b> ParticleInterface must inherit from ParticleBase.

     The Stack implements the
     std-type begin/end function to allow integration in normal for
     loops, ranges, etc.


    The template argument MSecondaryProducer is only needed because of gitlab Issue 161

    Due to a limitation of clang the corsika::MakeView does not work. Thus,
    MSecondaryProducer is needed here to fully define a SecondaryView class.
**/

  template <typename TStackData, template <typename> typename MParticleInterface,
            template <typename T1, template <class> class T2> class MSecondaryProducer =
                DefaultSecondaryProducer>
  class Stack {

    typedef typename std::remove_reference<TStackData>::type value_type;

  public:
    typedef TStackData stack_data_type; ///< this is the type of the
                                        ///< user-provided data structure

    typedef SecondaryView<TStackData, MParticleInterface, MSecondaryProducer>
        stack_view_type;

    template <typename TStackIterator>
    using pi_type = MParticleInterface<TStackIterator>;

    /**
     * Via the StackIteratorInterface and ConstStackIteratorInterface
     * specialization, the type of the stack_iterator_type
     * template class is declared for a particular stack data
     * object. Using CRTP, this also determines the type of
     * MParticleInterface template class simultaneously.
     */
    typedef StackIteratorInterface<value_type, MParticleInterface, MSecondaryProducer,
                                   Stack>
        stack_iterator_type;

    typedef ConstStackIteratorInterface<value_type, MParticleInterface,
                                        MSecondaryProducer, Stack>
        const_stack_iterator_type;

    /**
     * this is the full type of the user-declared MParticleInterface
     */
    typedef typename stack_iterator_type::particle_interface_type particle_interface_type;
    /**
     * In all programming context, the object to access, copy, and
     * transport particle data is via the stack_iterator_type
     */
    typedef stack_iterator_type particle_type;

    /**
     * create a new Stack, if there is already data associated prepare
     * needed initialization.
     */
    Stack()
        : nDeleted_(0)
        , data_()
        , deleted_(std::vector<bool>(data_.getSize(), false)) {}

    Stack(Stack&) = delete; ///< since Stack can be very big, we don't want to copy it

    Stack& operator=(Stack&) =
        delete; ///< since Stack can be very big, we don't want to copy it

    /**
     * if TStackData is a reference member we *HAVE* to initialize
     * it in the constructor, this is typically needed for SecondaryView
     */
    template <typename UType = TStackData,
              typename = typename std::enable_if<std::is_reference<UType>::value>::type>
    Stack(TStackData vD)
        : nDeleted_(0)
        , data_(vD)
        , deleted_(std::vector<bool>(data_.getSize(), false)) {}

    /**
     * This constructor takes any argument and passes it on to the
     * TStackData user class. If the user did not provide a suited
     * constructor this will fail with an error message.
     *
     * Furthermore, this is disabled with enable_if for SecondaryView
     * stacks, where the inner data container is always a reference
     * and cannot be initialized here.
     */
    template <typename... TArgs, typename UType = TStackData,
              typename = typename std::enable_if<std::is_reference<UType>::value>::type>
    Stack(TArgs... args)
        : nDeleted_(0)
        , data_(args...)
        , deleted_(std::vector<bool>(data_.getSize(), false)) {}

    /**
     * @name Most generic proxy methods for TStackData data_
     * @{
     */
    unsigned int getCapacity() const { return data_.getCapacity(); }

    unsigned int getErased() const { return nDeleted_; }

    unsigned int getEntries() const { return getSize() - getErased(); }

    template <typename... TArgs>
    void clear(TArgs... args);

    /**
     * Reserve a globally unique contiguous transport-history range when the
     * selected stack data provides TransportIdentityData.
     *
     * The dependent return type keeps this method absent for ordinary stacks.
     * Hybrid device schedulers use it immediately before a branching kernel so
     * CPU and GPU secondary IDs cannot overlap.
     */
    template <typename U = value_type>
    auto reserveTransportHistoryIds(std::uint64_t count)
        -> decltype(std::declval<U&>().reserveHistoryIds(count)) {
      return data_.reserveHistoryIds(count);
    }
    ///@}

    /**
     * @name These are functions required by std containers and std loops
     * @{
     */
    stack_iterator_type begin();

    stack_iterator_type end();

    stack_iterator_type last();

    const_stack_iterator_type begin() const;

    const_stack_iterator_type end() const;

    const_stack_iterator_type last() const;

    const_stack_iterator_type cbegin() const;

    const_stack_iterator_type cend() const;

    const_stack_iterator_type clast() const;

    stack_iterator_type at(unsigned int i);

    const_stack_iterator_type at(unsigned int i) const;

    stack_iterator_type first();

    const_stack_iterator_type cfirst() const;

    stack_iterator_type getNextParticle();

    /**
     * increase stack size, create new particle at end of stack
     */
    template <typename... TArgs>
    stack_iterator_type addParticle(const TArgs... v);

    void swap(stack_iterator_type a, stack_iterator_type b);

    void copy(stack_iterator_type a, stack_iterator_type b);

    void copy(const_stack_iterator_type a, stack_iterator_type b);

    void erase(stack_iterator_type p);
    /**
     * delete this particle
     */

    void erase(particle_interface_type p);

    /**
     * check if there are no further non-deleted particles on stack
     */
    bool isEmpty();

    /**
     * check if this particle was already deleted
     */

    bool isErased(const stack_iterator_type& p) const;

    bool isErased(const const_stack_iterator_type& p) const;

    bool isErased(const particle_interface_type& p) const;

    /**
     * Function to ultimatively remove the last entry from the stack,
     * if it was marked as deleted before. If this is not the case,
     * the function will just return false and do nothing.
     */
    bool purgeLastIfDeleted();

    /**
     * Function to ultimatively remove all entries from the stack
     * marked as deleted.
     *
     * Careful: this will re-order the entries on the stack, since
     * "gaps" in the stack are filled with entries from the back
     * (copied).
     */
    void purge();

    unsigned int getSize() const;

    std::string asString() const;

  protected:
    /**
     * increase stack size, create new particle at end of stack, related to parent
     * particle/projectile
     *
     * This should only get internally called from a
     * StackIterator::addSecondary via ParticleBase
     */
    template <typename... TArgs>
    stack_iterator_type addSecondary(stack_iterator_type& parent, const TArgs... v);

    void swap(unsigned int const a, unsigned int const b);

    void copy(unsigned int const a, unsigned int const b);

    bool isErased(unsigned int const i) const;

    void erase(unsigned int const i);

    /**
     * will remove from storage the element i. This is a helper
     * function for SecondaryView.
     */
    void purge(unsigned int i);

    /**
     * Function to perform eventual transformation from
     * StackIterator::getIndex() to index in data stored in
     * TStackData data_. By default (and in almost all cases) this
     * should just be identiy. See class SecondaryView for an alternative implementation.
     */
    unsigned int getIndexFromIterator(const unsigned int vI) const;
    /**
     * @name Return reference to TStackData object data_ for data access
     * @{
     */

    value_type& getStackData();

    const value_type& getStackData() const;

    friend class StackIteratorInterface<value_type, MParticleInterface,
                                        MSecondaryProducer, Stack>;
    friend class ConstStackIteratorInterface<value_type, MParticleInterface,
                                             MSecondaryProducer, Stack>;
    template <typename T1,                     //=TStackData,
              template <typename> typename M1, //=MParticleInterface,
              template <class T2, template <class> class T3> class M2>
    friend class SecondaryView;

    friend class ParticleBase<stack_iterator_type>;

  protected:
    unsigned int nDeleted_ = 0;

  private:
    TStackData data_; ///< this in general holds all the data and can be quite big
    std::vector<bool> deleted_; ///< bit field to flag deleted entries
  };                            // namespace corsika

} // namespace corsika

#include <corsika/detail/framework/stack/Stack.inl>
