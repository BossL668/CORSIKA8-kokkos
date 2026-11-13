User guide
==========

This guide takes you from a Linux installation to a configured air shower and
its output. It then explains how to use CPU threads, a GPU or several GPUs
without confusing execution settings with physics settings.

.. note::

   The commands and defaults were checked against the application and build
   scripts at `d5297c43
   <https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397>`_
   (12 November 2026). They describe ``c8_air_shower``. For another revision,
   check its installed ``--help`` and matching source. These examples are a
   starting configuration, not a physics-validation or performance benchmark.

.. toctree::
   :maxdepth: 1

   installation
   first-shower
   configuration
   parallel-execution
   outputs

If the program is already installed, start with :doc:`first-shower`. Before a
large simulation campaign, retain the source revision, full command, seed,
model data and antenna file with the results. This makes the configuration
reviewable when the code or computing platform changes.
