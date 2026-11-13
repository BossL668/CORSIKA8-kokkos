Configure a simulation
======================

Separate three choices when preparing a run: the primary and physics models,
the environment and observers, and the computing resources. Record each choice
explicitly for comparisons. A backend flag does not fully specify the physics
or the geometry.

Primary particle and transport settings
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Option
     - Meaning
   * - ``--pdg``
     - Primary PDG code, for example 2212 (proton), 22 (photon), 11 (electron)
       or 13 (negative muon).
   * - ``-A`` and ``-Z``
     - Nuclear mass and atomic numbers. Use ``-A 56 -Z 26`` for iron-56,
       instead of ``--pdg``.
   * - ``--energy``
     - Total primary energy in GeV. For a nucleus, this is the whole nucleus's
       energy, not energy per nucleon.
   * - ``--zenith``, ``--azimuth``
     - Arrival-direction angles in degrees; both default to zero.
   * - ``--emcut``, ``--hadcut``, ``--mucut``, ``--taucut``
     - Tracking thresholds in kinetic energy, in GeV. Defaults are
       0.0005, 0.3, 0.3 and 0.3, respectively.
   * - ``--max-deflection-angle``
     - Magnetic tracking limit in radians; default 0.2.
   * - ``--emthin``, ``--max-weight``
     - Electromagnetic thinning controls. ``--emthin`` defaults to
       :math:`10^{-6}`; set it to zero for an unthinned run. A zero maximum
       weight requests the program's automatic weight limit.
   * - ``--seed``, ``--nevent``
     - Random seed and number of showers. Use an explicit nonzero seed for
       reproducible configuration records.

For example, a 100 TeV, vertically incident iron nucleus is specified by
``-A 56 -Z 26 --energy 1e5 --zenith 0 --azimuth 0``. This alone does not specify
cuts, thinning, atmosphere, magnetic field or antenna layout.

Physics models and execution
----------------------------

``--hadronModel`` selects the high-energy hadronic model. The default is
``SIBYLL-2.3d``; the application also lists ``QGSJet-II.04``, ``QGSJet-III``,
``EPOS-LHC`` and ``Pythia8``. The required model libraries and data must be
available. ``--hadronModelTransitionEnergy`` sets the high-/low-energy switch
in GeV, with a default of about 79.4 GeV.

The low-energy hadronic model is selected at build time: FLUKA when
``WITH_FLUKA=ON``, otherwise UrQMD. Selecting a GPU does not replace these
hadronic models.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Selection
     - Effect
   * - ``--em-backend proposal``
     - Native C8 CPU electromagnetic path using PROPOSAL.
   * - ``--em-backend kokkos-proposal``
     - Kokkos transport using PROPOSAL physics and native table data.
   * - ``--em-backend kokkos-egs4``
     - Optional EGS4 implementation; requires a build with
       ``-DCORSIKA_ENABLE_EGS4=ON``.
   * - ``--kokkos-execution``
     - Chooses a compiled execution space, such as ``openmp``, ``cuda`` or
       ``hip``. This is distinct from the electromagnetic model choice.
   * - ``--radio-backend``
     - Selects ``cpu`` or ``kokkos`` radio execution. The air application's
       ``kokkos-proposal`` path requires ``kokkos`` here.

The air application constructs both CoREAS and ZHS. ``--radio-backend`` does
not select between these two formalisms; the current entry point has no
CoREAS-only or ZHS-only command-line switch.

Environment and magnetic field
------------------------------

The current air application constructs a five-layer US standard atmosphere
(``AtmosphereId::USStdBK``), with the refractivity
normalization set in the source. This scene is shared by the native and Kokkos
paths; Kokkos exports its supported geometry and medium information for device
use.

``--observation-level`` sets the particle observation height in metres above
the reference Earth radius (default 0). ``--injection-height`` sets the primary
injection height in the same convention (default 112750 m). The horizontal core
position is set with ``--shower-core-x`` and ``--shower-core-y``.

``--geomagnetic-model`` accepts ``constant`` (default), ``IGRF13`` or ``IGRF14``.
The constant field is :math:`(50\,\mu\mathrm{T},0,0)` in the application's root
coordinate system. ``--geomagnetic-year`` sets the IGRF evaluation year
(default 2027).

.. important::

   The current IGRF scene uses a site fixed in the application: latitude
   42.5527 degrees, longitude 86.4153816422 degrees, altitude 2680.444195 m.
   There are no latitude, longitude, field-component or atmosphere-profile
   CLI options in this entry point. Changing ``--observation-level`` does not
   change that IGRF evaluation site. A different site or atmospheric model
   requires changing the scene construction and rebuilding, or using an
   application that exposes that configuration.

This distinction matters when reproducing another study's condition card.
Do not assume an old benchmark's scene is the application's current default.

Antenna layout and radio windows
--------------------------------

Supply an antenna file with ``--antenna-file``. Each data row contains three
whitespace-separated values in metres: **north, west, up**. Empty lines and
lines starting with ``#`` are ignored.

For five observers at a height of 1000 m, create:

.. code-block:: bash

   cat > antennas-1000m.txt <<'EOF'
   # north_m west_m up_m
   0 0 1000
   100 0 1000
   0 100 1000
   -100 0 1000
   0 -100 1000
   EOF

Then add ``--antenna-file antennas-1000m.txt --observation-level 1000`` to the
run command. Antenna coordinates refer to the application's fixed local
north-west-up origin. The third coordinate is height above the reference Earth
radius; it is not an offset above the observation plane. Moving the shower
core does not automatically translate the antenna array.

Check the log's loaded-antenna count. An unreadable file or skipped malformed
rows must not be mistaken for the intended array.

For the small first-run example, ``--ring 1`` generates eight points at a radius
of 25 m. More generally, ``--ring r`` uses radius :math:`25r` metres. These
generated observers are at height zero, so use a file for an elevated array.
Without an antenna file or a positive ring selection, no radio observers are
configured.

Radio time sampling is controlled by:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Option
     - Meaning and default
   * - ``--radio-sampling-rate-ghz``
     - Sampling rate, 1 GHz (1 ns spacing).
   * - ``--radio-window-duration-ns``
     - Observer window duration, 400 ns.
   * - ``--radio-pretrigger-ns``
     - Start of antenna-file windows before the geometrical direct-arrival
       time, 10 ns.

These settings define time sampling and recording windows, not a detector
frequency filter. Check that the window contains the signal before comparing
waveforms or integrating fluence.

Source references
-----------------

See the `primary, model and scene configuration
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/applications/c8_air_shower.cpp>`_
and `backend defaults
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/BackendOptions.hpp>`_.
