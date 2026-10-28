Particle Storage and Stack
==========================

Particles in memory are stored in a Stack. The Stack handles the memory management and access to particle data
properties. The stack can be extended with additional fields that then become part of the particle object.

The standard Stack object in CORSIKA 8 is the NuclearStackExtension, further extended with internal
fields like `geometry node` and `weight` or `history`. But the important part is the `NuclearStackExtension`.

There are several ways to create particles:
* novel particles on the stack:
  *  stack::addParticle(particle_data_type const&)
  *  stack::addParticle(nuclear_particle_data_type const&)
  *  stack::addParticle(particle_data_momentum_type const&)
  *  stack::addParticle(nuclear_particle_data_momentum_type const&)
* secondary particles:
  *  stack::addSecondary(parent const&, particle_data_type const&)
  *  stack::addSecondary(parent const&, nuclear_particle_data_type const&)
  *  stack::addSecondary(parent const&, particle_data_momentum_type const&)
  *  stack::addSecondary(parent const&, nuclear_particle_data_momentum_type const&)
  or directly:
  *  particle::addSecondary(particle_data_type const&)
  *  particle::addSecondary(nuclear_particle_data_type const&)
  *  particle::addSecondary(particle_data_momentum_type const&)
  *  particle::addSecondary(nuclear_particle_data_momentum_type const&)

The content of these method parameters are:

*  particle_data_type: {PID [corsika::Code], kinetic energy [HEPEnergyType], direction [DirectionVector], position [Point], time[TimeType]}
*  particle_data_momentum_type: {PID [corsika::Code], momentum [MomentumVector], position [Point], time[TimeType]}
*  nuclear_particle_data_type: {PID [corsika::Code], kinetic energy [HEPEnergyType], direction [DirectionVector], position [Point], time[TimeType], A [unsigned int], Z [unsigned int]}
*  nuclear_particle_data_momentum_type: {PID [corsika::Code], momentum [MomentumVector], position [Point], time[TimeType], A [unsigned int], Z [unsigned int]}
