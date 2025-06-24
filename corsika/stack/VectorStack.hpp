/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/stack/Stack.hpp>

#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>

#include <string>
#include <tuple>
#include <vector>

namespace corsika {

  /**
   * Example of a particle object on the stack.
   */

  template <typename TStackIterator>
  class ParticleInterface : public ParticleBase<TStackIterator> {

  private:
    typedef ParticleBase<TStackIterator> super_type;

  public:
    /**
     * particle data information content.
     *
     * PID, Ekin, direction, position, time.
     */
    typedef std::tuple<Code, HEPEnergyType, DirectionVector, Point, TimeType>
        particle_data_type;

    /**
     * secondary particle data information content.
     *
     * PID, Ekin, direction.
     */
    typedef std::tuple<Code, HEPEnergyType, DirectionVector> secondary_data_type;

    /**
     * secondary particle data information content with position and time update.
     *
     * PID, Ekin, direction, delta-Position, delta-Time.
     */
    typedef std::tuple<Code, HEPEnergyType, DirectionVector, Vector<length_d>, TimeType>
        secondary_extended_data_type;

    std::string asString() const;

    /**
     * Set data of new particle.
     *
     * @param v tuple containing of type particle_data_type
     *
     */
    void setParticleData(particle_data_type const& v);

    /**
     * Set data of new particle.
     *
     * @param parent parent particle
     * @param v tuple containing of type secondary_data_type.
     */
    void setParticleData(ParticleInterface<TStackIterator> const& parent,
                         secondary_data_type const& v);

    /**
     * Set data of new particle.
     *
     * @param parent parent particle
     * @param v tuple containing of type secondary_extended_data_type.
     */
    void setParticleData(ParticleInterface<TStackIterator> const& parent,
                         secondary_extended_data_type const& v);

    ///! Set particle corsika::Code
    void setPID(Code const id) {
      super_type::getStackData().setPID(super_type::getIndex(), id);
    }

    ///! Set energy
    void setEnergy(HEPEnergyType const& e) {
      super_type::getStackData().setKineticEnergy(super_type::getIndex(),
                                                  e - this->getMass());
    }

    ///! Set kinetic energy
    void setKineticEnergy(HEPEnergyType const& ekin) {
      super_type::getStackData().setKineticEnergy(super_type::getIndex(), ekin);
    }

    //! Set direction
    void setDirection(DirectionVector const& v) {
      super_type::getStackData().setDirection(super_type::getIndex(), v);
    }
    //! Set position
    void setPosition(Point const& v) {
      super_type::getStackData().setPosition(super_type::getIndex(), v);
    }
    //! Set time
    void setTime(TimeType const& v) {
      super_type::getStackData().setTime(super_type::getIndex(), v);
    }

    //! Get corsika::Code
    Code getPID() const {
      return super_type::getStackData().getPID(super_type::getIndex());
    }
    //! Get PDG code
    PDGCode getPDG() const { return get_PDG(getPID()); }
    //! Get kinetic energy
    HEPEnergyType getKineticEnergy() const {
      return super_type::getStackData().getKineticEnergy(super_type::getIndex());
    }
    //! Get direction
    DirectionVector const& getDirection() const {
      return super_type::getStackData().getDirection(super_type::getIndex());
    }
    //! Get position
    Point const& getPosition() const {
      return super_type::getStackData().getPosition(super_type::getIndex());
    }
    //! Get time
    TimeType getTime() const {
      return super_type::getStackData().getTime(super_type::getIndex());
    }
    /**
     * @name derived quantities
     *
     * @{
     */
    //! Get velocity
    VelocityVector getVelocity() const {
      return this->getMomentum() / this->getEnergy() * constants::c;
    }
    //! Get momentum
    MomentumVector getMomentum() const {
      auto const P = calculate_momentum(this->getEnergy(), this->getMass());
      return super_type::getStackData().getDirection(super_type::getIndex()) * P;
    }
    //! Get mass of particle
    HEPMassType getMass() const { return get_mass(this->getPID()); }

    //! Get electric charge
    ElectricChargeType getCharge() const { return get_charge(this->getPID()); }

    //! Get total energy
    HEPEnergyType getEnergy() const { return this->getKineticEnergy() + this->getMass(); }

    //! Get energy per nucleon
    HEPEnergyType getEnergyNN() const {
      return (is_nucleus(this->getPID())
                  ? this->getEnergy() / get_nucleus_A(this->getPID())
                  : this->getEnergy());
    }

    //! Get charge number
    int16_t getChargeNumber() const { return get_charge_number(this->getPID()); }
    ///@}
  };

  /**
   * Memory implementation of the most simple particle stack object.
   *
   * @note if we ever want to have off-shell particles, we need to
   *       add momentum as HEPMomentumType, and a lot of care.
   */

  class VectorStackImpl {

  public:
    typedef std::vector<Code> code_vector_type;
    typedef std::vector<HEPEnergyType> kinetic_energy_vector_type;
    typedef std::vector<Point> point_vector_type;
    typedef std::vector<TimeType> time_vector_type;
    typedef std::vector<DirectionVector> direction_vector_type;

    VectorStackImpl() = default;

    VectorStackImpl(VectorStackImpl const& other) = default;

    VectorStackImpl(VectorStackImpl&& other) = default;

    VectorStackImpl& operator=(VectorStackImpl const& other) = default;

    VectorStackImpl& operator=(VectorStackImpl&& other) = default;

    void dump() const {}

    void clear();

    unsigned int getSize() const { return dataPID_.size(); }
    unsigned int getCapacity() const { return dataPID_.size(); }

    void setPID(size_t i, Code const id) { dataPID_[i] = id; }
    void setKineticEnergy(size_t i, HEPEnergyType const& e) { dataEkin_[i] = e; }
    void setDirection(size_t i, DirectionVector const& v) { direction_[i] = v; }
    void setPosition(size_t i, Point const& v) { position_[i] = v; }
    void setTime(size_t i, TimeType const& v) { time_[i] = v; }

    Code getPID(size_t i) const { return dataPID_[i]; }
    HEPEnergyType getKineticEnergy(size_t i) const { return dataEkin_[i]; }
    DirectionVector const& getDirection(size_t i) const { return direction_[i]; }
    Point const& getPosition(size_t i) const { return position_[i]; }
    TimeType getTime(size_t i) const { return time_[i]; }

    /**
     *   Function to copy particle at location i2 in stack to i1.
     */
    void copy(size_t const i1, size_t const i2);

    /**
     *   Function to copy particle at location i2 in stack to i1.
     */
    void swap(size_t const i1, size_t const i2);

    void incrementSize();
    void decrementSize();

  private:
    /// the actual memory to store particle data
    code_vector_type dataPID_;
    kinetic_energy_vector_type dataEkin_;
    direction_vector_type direction_;
    point_vector_type position_;
    time_vector_type time_;

  }; // end class VectorStackImpl

  typedef Stack<VectorStackImpl, ParticleInterface> VectorStack;

} // namespace corsika

#include <corsika/detail/stack/VectorStack.inl>
