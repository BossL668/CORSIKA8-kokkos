.. raw:: html

   <p class="guide-eyebrow">USER AND DEVELOPER DOCUMENTATION</p>

CORSIKA 8 with Kokkos
=====================

.. container:: guide-lead

   A guide to using, understanding, developing and maintaining CORSIKA 8
   with Kokkos.

.. raw:: html

   <div class="guide-status" role="note">
     <span class="status-dot" aria-hidden="true"></span>
     <strong>Documentation in preparation</strong>
     <span>The overview, user guide, application integration, execution, particle-data, environment, physics-data, transport and radio chapters are available. Further architecture and maintenance chapters are being developed.</span>
   </div>

Guide objectives
----------------

.. raw:: html

   <div class="guide-cards">
     <a class="guide-card" href="overview.html">
       <span class="card-kicker">01 / UNDERSTAND</span>
       <h3>Understand the implementation</h3>
       <p>Understand the software architecture, component responsibilities and integration with the CORSIKA 8 framework.</p>
       <span class="card-link">Start with the overview <span aria-hidden="true">&#8594;</span></span>
     </a>
     <a class="guide-card" href="development.html">
       <span class="card-kicker">02 / DEVELOP &amp; MAINTAIN</span>
       <h3>Develop and maintain</h3>
       <p>Support continued development and long-term maintenance through documented interfaces, design decisions and review practices.</p>
       <span class="card-link">Open development and maintenance <span aria-hidden="true">&#8594;</span></span>
     </a>
     <a class="guide-card" href="validation.html">
       <span class="card-kicker">03 / VALIDATE</span>
       <h3>Validate changes</h3>
       <p>Assess correctness and performance with reproducible tests, benchmarks and regression checks across supported backends.</p>
       <span class="card-link">Open validation and performance <span aria-hidden="true">&#8594;</span></span>
     </a>
   </div>

Inside this guide
-----------------

Start with the :doc:`usage/index` to install the application, run a shower and
inspect its output. The :doc:`application` chapter follows the actual function
calls, arguments and results from the main application into the library.
The :doc:`execution` chapter continues through particle routing, batch execution
and process coordination. The :doc:`particles` chapter defines particle-state
representations, conversion interfaces and backend storage. The
:doc:`environment` chapter follows native scene export, device queries and
extension requirements. The :doc:`physics-data` chapter documents native model
preparation, table storage and query contracts. The :doc:`transport` chapter
follows candidate sampling, particle propagation and interaction final states.
The :doc:`radio` chapter covers track collection, projection, waveform
accumulation and propagation-specific interfaces. Further chapters will
cover backend configuration, validation and maintenance.

.. toctree::
   :maxdepth: 1
   :caption: Getting oriented

   overview

.. toctree::
   :maxdepth: 2
   :caption: Using the application

   usage/index

.. toctree::
   :maxdepth: 2
   :caption: Architecture and data

   application
   execution
   particles
   environment
   physics-data
   transport
   radio

.. toctree::
   :maxdepth: 1
   :caption: Development and maintenance

   backends
   validation
   development
   decisions
   api
   about

.. container:: guide-reference

   Related resources: `CORSIKA 8 documentation <https://corsika-8.readthedocs.io/>`_
   · `Project source <https://github.com/BossL668/CORSIKA8-kokkos>`_
