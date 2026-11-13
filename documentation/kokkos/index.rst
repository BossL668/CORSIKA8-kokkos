.. raw:: html

   <p class="guide-eyebrow">DEVELOPER &amp; MAINTENANCE GUIDE</p>

CORSIKA 8 with Kokkos
=====================

.. container:: guide-lead

   A guide to understanding, extending and maintaining the Kokkos backend.

.. raw:: html

   <div class="guide-status" role="note">
     <span class="status-dot" aria-hidden="true"></span>
     <strong>Documentation scaffold</strong>
     <span>The navigation is ready. Technical chapters will be added next.</span>
   </div>

Choose a starting point
-----------------------

.. raw:: html

   <div class="guide-cards">
     <a class="guide-card" href="overview.html">
       <span class="card-kicker">01 / UNDERSTAND</span>
       <h3>Understand the architecture</h3>
       <p>Follow the application interface, particle data and execution flow.</p>
       <span class="card-link">Start with the overview <span aria-hidden="true">&#8594;</span></span>
     </a>
     <a class="guide-card" href="development.html">
       <span class="card-kicker">02 / DEVELOP</span>
       <h3>Make a change</h3>
       <p>Find the extension points for environments, models and particle state.</p>
       <span class="card-link">Open the development guide <span aria-hidden="true">&#8594;</span></span>
     </a>
     <a class="guide-card" href="validation.html">
       <span class="card-kicker">03 / VERIFY</span>
       <h3>Check the result</h3>
       <p>Find validation procedures, timing definitions and diagnostic guidance.</p>
       <span class="card-link">Open validation and performance <span aria-hidden="true">&#8594;</span></span>
     </a>
   </div>

Inside this guide
-----------------

The chapters below are reserved for the implementation guide. Each chapter
currently contains its scope and an outline, rather than claims about the code.

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
