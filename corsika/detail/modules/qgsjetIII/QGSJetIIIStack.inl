/*
 * (c) Copyright 2025 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::qgsjetIII {

  inline void QGSJetIIIStackData::clear() {
    ::QGSJetIII::qgarr12_->nsp = 0;
    ::QGSJetIII::qgarr13_->nsf = 0;
    ::QGSJetIII::qgarr55_->nwt = 0;
  }
  inline unsigned int QGSJetIIIStackData::getSize() const { return ::QGSJetIII::qgarr12_->nsp; }
  inline unsigned int QGSJetIIIStackData::getCapacity() const { return nptmax; }

  inline void QGSJetIIIStackData::setId(const unsigned int i, const int v) {
    ::QGSJetIII::qgarr14_->ich[i] = v;
  }
  inline void QGSJetIIIStackData::setEnergy(const unsigned int i, const HEPEnergyType v) {
    ::QGSJetIII::qgarr14_->esp[i][0] = v / 1_GeV;
  }

  inline void QGSJetIIIStackData::setMomentum(const unsigned int i,
                                             const MomentumVector& v) {
    auto tmp = v.getComponents();
    ::QGSJetIII::qgarr14_->esp[i][2] = tmp[0] / 1_GeV;
    ::QGSJetIII::qgarr14_->esp[i][3] = tmp[1] / 1_GeV;
    ::QGSJetIII::qgarr14_->esp[i][1] = tmp[2] / 1_GeV;
  }

  inline int QGSJetIIIStackData::getId(const unsigned int i) const {
    return ::QGSJetIII::qgarr14_->ich[i];
  }
  inline HEPEnergyType QGSJetIIIStackData::getEnergy(const int i) const {
    return ::QGSJetIII::qgarr14_->esp[i][0] * 1_GeV;
  }
  inline MomentumVector QGSJetIIIStackData::getMomentum(
      const unsigned int i, const CoordinateSystemPtr& CS) const {
    QuantityVector<hepmomentum_d> components = {::QGSJetIII::qgarr14_->esp[i][2] * 1_GeV,
                                                ::QGSJetIII::qgarr14_->esp[i][3] * 1_GeV,
                                                ::QGSJetIII::qgarr14_->esp[i][1] * 1_GeV};
    return MomentumVector(CS, components);
  }

  inline void QGSJetIIIStackData::copy(const unsigned int i1, const unsigned int i2) {
    ::QGSJetIII::qgarr14_->ich[i2] = ::QGSJetIII::qgarr14_->ich[i1];
    for (unsigned int i = 0; i < 4; ++i) ::QGSJetIII::qgarr14_->esp[i2][i] = ::QGSJetIII::qgarr14_->esp[i1][i];
  }

  inline void QGSJetIIIStackData::swap(const unsigned int i1, const unsigned int i2) {
    std::swap(::QGSJetIII::qgarr14_->ich[i1], ::QGSJetIII::qgarr14_->ich[i2]);
    for (unsigned int i = 0; i < 4; ++i)
      std::swap(::QGSJetIII::qgarr14_->esp[i1][i], ::QGSJetIII::qgarr14_->esp[i2][i]);
  }

  inline void QGSJetIIIStackData::incrementSize() { ::QGSJetIII::qgarr12_->nsp++; }
  inline void QGSJetIIIStackData::decrementSize() {
    if (::QGSJetIII::qgarr12_->nsp > 0) { ::QGSJetIII::qgarr12_->nsp--; }
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
  inline corsika::qgsjetIII::QgsjetIIICode
  ParticleInterface<StackIteratorInterface>::getPID() const {
    return static_cast<corsika::qgsjetIII::QgsjetIIICode>(getStackData().getId(getIndex()));
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

} // namespace corsika::qgsjetIII
