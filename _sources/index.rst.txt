.. raw:: html

   <p class="guide-eyebrow">DEVELOPER DOCUMENTATION</p>

CORSIKA 8 with Kokkos
=====================

.. container:: guide-lead

   Technical documentation for the implementation, development and long-term
   maintenance of CORSIKA 8 with Kokkos.

.. raw:: html

   <div class="guide-status" role="note">
     <span class="status-dot" aria-hidden="true"></span>
     <strong>Documentation in preparation</strong>
     <span>Chapter outlines are available. Technical content will be added and reviewed progressively.</span>
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

The chapters cover architecture, implementation, development and verification.
Select a chapter to view its scope and planned coverage.

.. toctree::
   :maxdepth: 1
   :caption: Getting oriented

   overview
   application

.. toctree::
   :maxdepth: 1
   :caption: Architecture and data

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
