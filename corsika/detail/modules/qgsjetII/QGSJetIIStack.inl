/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <qgsjet-II-public.hpp>

namespace corsika::qgsjetII {

  inline void QGSJetIIStackData::clear() {
    ::QGSJetII04::qgarr12_->nsp = 0;
    ::QGSJetII04::qgarr13_->nsf = 0;
    ::QGSJetII04::qgarr55_->nwt = 0;
  }
  inline unsigned int QGSJetIIStackData::getSize() const {
    return ::QGSJetII04::qgarr12_->nsp;
  }
  inline unsigned int QGSJetIIStackData::getCapacity() const { return nptmax; }

  inline void QGSJetIIStackData::setId(const unsigned int i, const int v) {
    ::QGSJetII04::qgarr14_->ich[i] = v;
  }
  inline void QGSJetIIStackData::setEnergy(const unsigned int i, const HEPEnergyType v) {
    ::QGSJetII04::qgarr14_->esp[i][0] = v / 1_GeV;
  }

  inline void QGSJetIIStackData::setMomentum(const unsigned int i,
                                             const MomentumVector& v) {
    auto tmp = v.getComponents();
    auto constexpr invGeV = 1 / 1_GeV;
    ::QGSJetII04::qgarr14_->esp[i][2] = tmp[0] * invGeV;
    ::QGSJetII04::qgarr14_->esp[i][3] = tmp[1] * invGeV;
    ::QGSJetII04::qgarr14_->esp[i][1] = tmp[2] * invGeV;
  }

  inline int QGSJetIIStackData::getId(const unsigned int i) const {
    return ::QGSJetII04::qgarr14_->ich[i];
  }
  inline HEPEnergyType QGSJetIIStackData::getEnergy(const int i) const {
    return ::QGSJetII04::qgarr14_->esp[i][0] * 1_GeV;
  }
  inline MomentumVector QGSJetIIStackData::getMomentum(
      const unsigned int i, const CoordinateSystemPtr& CS) const {
    QuantityVector<hepmomentum_d> components = {
        ::QGSJetII04::qgarr14_->esp[i][2] * 1_GeV,
        ::QGSJetII04::qgarr14_->esp[i][3] * 1_GeV,
        ::QGSJetII04::qgarr14_->esp[i][1] * 1_GeV};
    return MomentumVector(CS, components);
  }

  inline void QGSJetIIStackData::copy(const unsigned int i1, const unsigned int i2) {
    ::QGSJetII04::qgarr14_->ich[i2] = ::QGSJetII04::qgarr14_->ich[i1];
    for (unsigned int i = 0; i < 4; ++i)
      ::QGSJetII04::qgarr14_->esp[i2][i] = ::QGSJetII04::qgarr14_->esp[i1][i];
  }

  inline void QGSJetIIStackData::swap(const unsigned int i1, const unsigned int i2) {
    std::swap(::QGSJetII04::qgarr14_->ich[i1], ::QGSJetII04::qgarr14_->ich[i2]);
    for (unsigned int i = 0; i < 4; ++i)
      std::swap(::QGSJetII04::qgarr14_->esp[i1][i], ::QGSJetII04::qgarr14_->esp[i2][i]);
  }

  inline void QGSJetIIStackData::incrementSize() { ::QGSJetII04::qgarr12_->nsp++; }
  inline void QGSJetIIStackData::decrementSize() {
    if (::QGSJetII04::qgarr12_->nsp > 0) { ::QGSJetII04::qgarr12_->nsp--; }
  }

  template <typename StackIteratorInterface>
  inline void ParticleInterface<StackIteratorInterface>::setParticleData(
      const int vID, const HEPEnergyType vE, const MomentumVector& vP,
      const HEPMassType) {
    setPID(vID);
    setEnergy(vE);
    setMomentum(vP);
  }

  template <typename StackIteratorInterface>
  inline void ParticleInterface<StackIteratorInterface>::setParticleData(
      ParticleInterface<StackIteratorInterface>& /*parent*/, const int vID,
      const HEPEnergyType vE, const MomentumVector& vP, const HEPMassType) {
    setPID(vID);
    setEnergy(vE);
    setMomentum(vP);
  }

  template <typename StackIteratorInterface>
  inline void ParticleInterface<StackIteratorInterface>::setEnergy(
      const HEPEnergyType v) {
    getStackData().setEnergy(getIndex(), v);
  }

  template <typename StackIteratorInterface>
  inline HEPEnergyType ParticleInterface<StackIteratorInterface>::getEnergy() const {
    return getStackData().getEnergy(getIndex());
  }

  template <typename StackIteratorInterface>
  inline void ParticleInterface<StackIteratorInterface>::setPID(const int v) {
    getStackData().setId(getIndex(), v);
  }

  template <typename StackIteratorInterface>
  inline corsika::qgsjetII::QgsjetIICode
  ParticleInterface<StackIteratorInterface>::getPID() const {
    return static_cast<corsika::qgsjetII::QgsjetIICode>(getStackData().getId(getIndex()));
  }

  template <typename StackIteratorInterface>
  inline MomentumVector ParticleInterface<StackIteratorInterface>::getMomentum(
      const CoordinateSystemPtr& CS) const {
    return getStackData().getMomentum(getIndex(), CS);
  }

  template <typename StackIteratorInterface>
  inline void ParticleInterface<StackIteratorInterface>::setMomentum(
      const MomentumVector& v) {
    getStackData().setMomentum(getIndex(), v);
  }

} // namespace corsika::qgsjetII
