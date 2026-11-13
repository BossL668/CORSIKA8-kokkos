Installation
============

The build helper creates an independent installation for each backend. Choose
OpenMP for CPU execution, CUDA for NVIDIA GPUs, or HIP for a compatible ROCm/DTK
system. You do not need an OpenMP installation before building a GPU backend.
The application is a native binary; Python supplies build and optional analysis
tools.

Prepare the host tools and FLUKA
--------------------------------

The documented build uses C++17, GCC 11 or newer with a compatible GFortran,
CMake 3.24--3.x and Conan 2. The repository pins Kokkos to 4.7.03. Start with one
compile job and allow space for source, dependency caches and model data in
addition to simulation output.

On Ubuntu, install the host tools:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y build-essential gfortran git curl ca-certificates \
     python3-venv python3-dev pkg-config autoconf automake libtool bison flex \
     patch unzip bzip2 xz-utils perl
   export CC=/usr/bin/gcc
   export CXX=/usr/bin/g++
   export FC=/usr/bin/gfortran

On a cluster, use the site's matching compiler modules. Keep these compilers
consistent with the Conan profile, including when Conda supplies the build tools.

FLUKA is enabled by default. It is a separately licensed prerequisite and is
not downloaded by the helper. Prepare a complete compatible installation,
including its runtime data, and set its location:

.. code-block:: bash

   export FLUPRO="$HOME/opt/fluka"  # replace with your installation
   export FLUFOR=gfortran
   export GFORFLU="$FC"
   test -r "$FLUPRO/libflukahp.a"
   ar t "$FLUPRO/libflukahp.a" | grep -x ndmhep.o

The C8 interface expects ``libflukahp.a`` containing ``ndmhep.o``. A package with
a different interface is not interchangeable. See the `FLUKA installation
instructions <https://www.fluka.eu/Fluka/www/html/fluka.php?id=ins_run&mm2=3>`_
and the repository's `installation notes
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/README.md#install-on-linux>`_.

For an intentional build without FLUKA, append ``-DWITH_FLUKA=OFF`` to both the
check and build commands below. The air application then uses UrQMD for its
low-energy hadronic model, so the physics configuration changes.

Choose one build-tool environment
---------------------------------

Use either Conda or a Python virtual environment. Activate it again in each
terminal used for building, together with the compiler and FLUKA variables.

**Conda.** If necessary, first install `Miniforge
<https://github.com/conda-forge/miniforge#install>`_ or `Miniconda
<https://www.anaconda.com/docs/getting-started/installation>`_ for your operating
system. Then create an environment:

.. code-block:: bash

   conda create -n corsika8 --override-channels -c conda-forge \
     python=3.11 pip cmake=3.31 ninja numpy=1.26.4 pyyaml lxml
   conda activate corsika8
   python -m pip install 'conan>=2,<3' particle==0.25.1 hepunits==2.4.1

**Without Conda.** On Ubuntu 22.04/24.04, use the system Python in a virtual
environment:

.. code-block:: bash

   python3 -m venv "$HOME/venvs/corsika8"
   source "$HOME/venvs/corsika8/bin/activate"
   python -m pip install --upgrade pip
   python -m pip install 'conan>=2,<3' 'cmake>=3.24,<4' ninja \
     particle==0.25.1 hepunits==2.4.1 numpy==1.26.4 pyyaml lxml

Get the source
--------------

Use a recursive clone so the required submodules are present:

.. code-block:: bash

   mkdir -p "$HOME/corsika8-workspace"
   cd "$HOME/corsika8-workspace"
   git clone --recursive --branch CORSIKA8-kokkos \
     https://github.com/BossL668/CORSIKA8-kokkos.git
   cd CORSIKA8-kokkos
   git submodule status --recursive
   conan profile path default || conan profile detect
   conan profile show -pr:h default -pr:b default
   conan remote list

A leading ``-`` in submodule status indicates a missing submodule; run
``git submodule update --init --recursive``. Check that the Conan compiler
version matches the active C++ compiler. With GCC, the profile should use
``libstdc++11``. If ConanCenter is absent, add it once:

.. code-block:: bash

   conan remote add conancenter https://center2.conan.io

The helper exports this repository's dependency recipes and prepares the model
data. IGRF14 coefficients are bundled and installed in the normal ``GeoMag``
resource directory. Do not combine headers from these recipes with unrelated
older PROPOSAL libraries.

Build the selected backend
--------------------------

Run these commands from the source directory. ``--check`` checks prerequisites
without fetching dependencies or compiling. The full command prepares data,
resolves dependencies, configures, compiles and installs the program.

**CPU / OpenMP**

.. code-block:: bash

   bash tools/build_kokkos.sh openmp --check
   C8_BUILD_JOBS=1 bash tools/build_kokkos.sh openmp

**NVIDIA / CUDA**

A working driver and a compatible development toolkit containing ``nvcc`` are
required. Check both; the version displayed by ``nvidia-smi`` is not the
installed compiler version.

.. code-block:: bash

   nvidia-smi
   nvcc --version
   bash tools/build_kokkos.sh cuda --check
   C8_BUILD_JOBS=1 bash tools/build_kokkos.sh cuda

The helper detects the local NVIDIA GPU architecture. On a mixed-GPU host,
set ``C8_CUDA_DEVICE`` to the intended build target, for example
``C8_CUDA_DEVICE=0 bash tools/build_kokkos.sh cuda``. This selects the build
target, not the device used by a later simulation. A known target can also be
selected with ``C8_KOKKOS_PROFILE``, for example ``cuda-volta70`` for V100/V100S.
The toolkit must support that target.

**HIP / ROCm or Hygon DTK**

Activate the site's GPU software environment and select a profile matching
both its compiler and GPU architecture. For the supplied Z100L / DTK 22.10.1
profile:

.. code-block:: bash

   source /opt/dtk-22.10.1/env.sh
   hipcc --version
   export C8_KOKKOS_PROFILE=hip-vega906-dtk221001
   bash tools/build_kokkos.sh hip --check
   C8_BUILD_JOBS=1 bash tools/build_kokkos.sh hip

Adjust the environment path to the installation on your machine. The
``hip-vega90a`` profile is an example for a different ROCm/compiler setup;
it is not a universal HIP profile. The account also needs access to the GPU
device files, normally ``/dev/kfd`` and ``/dev/dri/renderD*``. Ask the system
administrator if compilation works but device access is denied.

Locate the installation
-----------------------

With the default workspace, the result is:

.. code-block:: text

   corsika8-workspace/
   ├── CORSIKA8-kokkos/          source
   ├── build/<backend>/deps/    dependency and build configuration
   └── install/<backend>/
       ├── bin/c8_air_shower
       └── share/corsika/data/

``C8_BUILD_JOBS`` controls compilation concurrency, not shower threads.
``C8_WORKSPACE`` changes the build/install root, while ``CONAN_HOME`` changes
Conan's separate cache. Set these before building when using another disk.
Use separate build directories after changing toolchain or GPU architecture.

Continue with :doc:`first-shower` to select the executable and runtime data.
The `build helper
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tools/build_kokkos.sh>`_
and `backend profiles
<https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397/externals/conan>`_
are the source references for this page.
