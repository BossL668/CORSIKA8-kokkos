Repository organization and coding workflow
============================================

This page describes the repository at `d5297c43`_. Use its build configuration,
public headers and test registrations when preparing a change. The
`official coding conventions`_ are the upstream style reference; they describe
a different documentation branch and do not by themselves change this source
tree's compiler or dependency requirements.

Conventions and the language standard
-------------------------------------

The official conventions specify ``.hpp`` declarations, ``.inl`` inline
implementations and ``.cpp`` compilation units. They use PascalCase class names,
snake_case free functions and namespaces, camelCase methods, and a trailing
underscore on data members. They also call for documented public interfaces,
explicit ownership, project logging, appropriate tests and physics validation.
Their CMake guidance uses lower-case commands, upper-case variables, a space
before the opening parenthesis and two-space indentation. That page recommends
C++23 and newer standard-library interfaces. Consult the linked page for the
complete upstream policy.

**This source baseline uses C++17.** Its `top-level CMake configuration`_ checks
the compiler's C++17 support and sets:

.. literalinclude:: ../../../CMakeLists.txt
   :language: cmake
   :lines: 112-113
   :lineno-match:

The `Kokkos target definitions`_ also request ``cxx_std_17`` and require
Kokkos 4.x, starting at 4.7; they reject Kokkos 5. The device-test helper requests
CUDA or HIP standard 17. New code for this branch must therefore remain usable
with that configuration. Introducing C++23 facilities requires a separate,
reviewed toolchain and dependency migration, including the device compiler and
installed-consumer checks. A formatter's ``Standard: Auto`` setting does not
raise the compiler language standard. See :doc:`../backends/build` for the
dependency boundary.

Repository responsibilities
----------------------------

The `Kokkos library layout`_ describes the public-header and implementation
split. The actual CMake targets determine which dependencies a component
requires and what is installed.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Location
     - Responsibility at this baseline
   * - ``applications/``
     - Air, ice and firn executable entry points. Their CMake file links them
       to ``CORSIKA8::CascadeApplication``.
   * - ``corsika/kokkos/``
     - Public application, data, environment, transport, radio, scheduling
       and runtime declarations.
   * - ``corsika/detail/kokkos/``
     - Inline and template implementations, normally mirroring the public
       header path with an ``.inl`` suffix.
   * - ``src/kokkos/``
     - Compiled implementations and Kokkos target definitions, including
       runtime ownership, native bridges and backend factories.
   * - ``corsika/framework/``, ``corsika/media/``, ``corsika/modules/``
     - Native C8 interfaces and models that remain available to scalar C8
       code. Compiled counterparts live under ``src/``.
   * - ``modules/``, ``externals/``, ``third_party/``, ``dependencies/``
     - Physics packages and data integration, bundled dependencies, dependency
       recipes and backend build profiles. Check the relevant recipe and
       CMake registration before changing a dependency.
   * - ``tests/kokkos/``, ``tests/application_support/``
     - Subsystem tests and application integration checks. Build and installed
       consumer contracts are also registered through ``cmake/`` helpers.
   * - ``examples/``, ``documentation/``, ``tools/``
     - Examples, user/developer documentation and supporting programs.

Place reusable application preparation in
``corsika/kokkos/application/cascade/`` and its corresponding implementation
directory. The library layout explicitly keeps application implementation
helpers out of ``applications/``. Scene-specific policies belong below the
atmosphere, ice or firn directories; shared particle routing, transport and
runtime services retain their common interfaces.

Names do not always describe the full current scope. The layout document
identifies ``FlatAtmosphereRadioSnapshot`` as also carrying homogeneous ice
propagation and ``AirShowerCoordinator`` as the shared process coordinator.
Extend the existing interface when its contract applies. A broad rename or a
copied scene-specific implementation would affect consumers beyond the name's
original application.

Headers, implementations and public documentation
-------------------------------------------------

``ParticleSoA.hpp`` declares the storage and view interfaces and includes
``corsika/detail/kokkos/data/ParticleSoA.inl`` after the namespace closes.
By contrast, ``Runtime.hpp`` declares a host interface implemented in
``src/kokkos/runtime/Runtime.cpp``. The extension alone does not identify
whether a header exposes device data, owns storage or launches work; preserve
the contract described by the relevant component chapter.

``Runtime.hpp`` provides an example of useful declaration-level documentation:

.. literalinclude:: ../../../corsika/kokkos/runtime/Runtime.hpp
   :language: cpp
   :lines: 33-44
   :lineno-match:

Its surrounding class documentation describes thread affinity, ownership and
destruction order. Follow that level of precision for new interfaces: state
units, valid ranges, memory accessibility, ownership, lifetime, synchronization
and failure behavior where they matter. Keep public API documentation in the
header and explain non-obvious algorithmic choices beside the implementation.
Moving a definition into an ``.inl`` file does not remove the need to document
the declaration that callers read.

The observed source is not uniformly compliant with every current upstream
rule. For example, ``ParticleSoA`` deliberately exposes storage fields and
``ParticleSoARawView`` contains non-owning pointers. The top-level and
application CMake files mix command capitalization and spacing. Treat these as
existing interfaces and style inconsistencies to assess, rather than evidence
that the upstream recommendations have changed. Any ownership redesign or
public-name change needs its own compatibility review; mechanical formatting
does not establish its correctness.

Write new public guide text and API explanations in English, consistent with
the repository's English ``README.md``, library layout and public headers.
The `documentation map`_ distinguishes bilingual beginner guides from dated
research records, many of which are Chinese. Preserve those records as evidence
and summarize the applicable result in the maintained guide, with its source
version and limitations. Rebuild the guide after editing cross-references or
source excerpts; the :download:`build instructions <../BUILDING.rst>` give the
documentation-only procedure.

Formatting scope and tooling
-----------------------------

The repository's `.clang-format`_ specifies two-space indentation, spaces
instead of tabs, a 90-column limit and left-aligned pointer declarations. It
sets ``SortIncludes: false``. Use this file rather than a formatter's built-in
style. Formatting does not enforce naming, ownership or API documentation.

`CONTRIBUTING.md`_ requires formatting before submission, but its reference to
``do-clang-format.sh`` is stale: the checked-in tool is `do-clang-format.py`_.
Its behavior is important when reviewing its output:

* With no options, it selects tracked differences from local ``master`` plus
  untracked, non-ignored files. This can include the entire feature-branch
  difference, not just edits from the current session.
* It considers ``.hpp``, ``.cpp`` and ``.inl`` files, skips deleted paths and
  symbolic links, and applies its explicit path exclusions. Those exclusions
  are not a general detector for all vendor or generated directories.
* The default mode prints files requiring formatting and exits with status 1
  if any need changes; it does not edit them. ``--apply`` writes the selected
  files. ``--all`` changes selection to the directory tree below the current
  directory.
* ``CLANG_FORMAT`` selects the formatter command; otherwise the tool invokes
  ``clang-format``. Its ``--docker`` mode selects ``clang-format-8`` in
  ``corsika/devel:clang-8``.

From the repository root, with a local ``master`` reference and
``clang-format-8`` installed, the existing non-mutating check is:

.. code-block:: console

   CLANG_FORMAT=clang-format-8 python3 do-clang-format.py
   git diff --check

The first command does not accept a replacement comparison branch. If local
``master`` is absent or unrelated to the intended change, review the changed
paths against the actual target branch and use the editor or formatter on
explicitly selected files. Inspect the diff after formatting. Do not combine
``--all`` and ``--apply`` as routine preparation for a focused patch: that can
rewrite unrelated code and preserved comparison implementations.

The checked-in `GitLab CI configuration`_ runs the full-tree check in the
``corsika/devel:clang-8`` image. It marks that job ``allow_failure: true``.
This is evidence of the configured formatter generation and advisory CI job,
not proof that a formatting failure blocks merging. Match the branch's
formatter version: different releases can produce different edits from the
same configuration.

Preparing a focused change
---------------------------

``CONTRIBUTING.md`` describes issue-based development on branches followed by
a reviewed merge request. For this implementation, prepare the change around
the interface or behavior being modified:

1. Identify the public declaration, implementation, owning target, consumers
   and nearest tests. Keep retained ``Reference.hpp``, ``reference/`` and
   ``oracle/`` comparison implementations independent of routine production
   refactoring; :doc:`../validation/tests` explains their role.
2. Add source files and dependencies to the responsible CMake target. The
   Kokkos subdirectory is conditional on ``CORSIKA_ENABLE_KOKKOS``; adding a
   public include must not silently make native-only consumers require Kokkos.
   Use the existing device-test registration helper when a test source needs
   CUDA or HIP compilation.
3. Update the header contract, relevant guide page and focused tests alongside
   the implementation. For a behavioral change, describe the triggering input
   and the intended result so reviewers can assess what the tests establish.
4. Review formatting and the final diff, then run checks appropriate to the
   affected target and backend. Record the configuration and results. Use
   :doc:`../validation` to distinguish build checks, numerical comparisons,
   physics validation and performance measurements.

Keep build artifacts outside the source tree's source directories: the
top-level CMake file rejects an in-source build. Installed include paths,
exported targets and optional-dependency boundaries are part of the public
interface, so a successful local application build alone is insufficient when
those contracts change.

.. _d5297c43: https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397
.. _official coding conventions: https://corsika-8.readthedocs.io/en/documentation-upgrade/code/coding_conventions.html
.. _top-level CMake configuration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/CMakeLists.txt
.. _Kokkos target definitions: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/CMakeLists.txt
.. _Kokkos library layout: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/README.md
.. _documentation map: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/documentation/USER_GUIDE_INDEX.md
.. _.clang-format: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/.clang-format
.. _CONTRIBUTING.md: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/CONTRIBUTING.md
.. _do-clang-format.py: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/do-clang-format.py
.. _GitLab CI configuration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/.gitlab-ci.yml
