from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import get


class KokkosConan(ConanFile):
    name = "kokkos"
    version = "4.7.03"
    user = "c8gpu"
    channel = "stable"
    package_type = "static-library"

    license = "BSD-3-Clause"
    url = "https://github.com/kokkos/kokkos"
    description = "Kokkos performance-portability library for CORSIKA 8"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "backend": ["openmp", "cuda", "cuda_openmp", "hip", "sycl"],
        "architecture": ["ANY"],
        "fPIC": [True, False],
    }
    default_options = {
        "backend": "openmp",
        "architecture": "NONE",
        "fPIC": True,
    }

    exports_sources = "LICENSE.c8gpu"

    def source(self):
        get(
            self,
            url=(
                "https://github.com/kokkos/kokkos/releases/download/"
                "4.7.03/kokkos-4.7.03.tar.gz"
            ),
            sha256=(
                "969e7933b9426219b220f08036e489b3226e6d8cd24eecf2c5b80df8c37443c0"
            ),
            strip_root=True,
        )

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        backend = str(self.options.backend)
        tc.cache_variables["CMAKE_CXX_STANDARD"] = "17"
        tc.cache_variables["CMAKE_CXX_STANDARD_REQUIRED"] = True
        tc.cache_variables["BUILD_SHARED_LIBS"] = False
        tc.cache_variables["Kokkos_ENABLE_TESTS"] = False
        tc.cache_variables["Kokkos_ENABLE_EXAMPLES"] = False
        tc.cache_variables["Kokkos_ENABLE_BENCHMARKS"] = False
        tc.cache_variables["Kokkos_ENABLE_DEPRECATED_CODE_4"] = False
        tc.cache_variables["Kokkos_ENABLE_SERIAL"] = backend != "openmp"
        tc.cache_variables["Kokkos_ENABLE_OPENMP"] = backend in ("openmp", "cuda_openmp")
        tc.cache_variables["Kokkos_ENABLE_CUDA"] = backend in ("cuda", "cuda_openmp")
        tc.cache_variables["Kokkos_ENABLE_HIP"] = backend == "hip"
        tc.cache_variables["Kokkos_ENABLE_SYCL"] = backend == "sycl"
        tc.cache_variables["Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE"] = (
            backend in ("cuda", "cuda_openmp", "hip")
        )
        tc.cache_variables["Kokkos_ENABLE_CUDA_UVM"] = False
        tc.cache_variables["Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC"] = False

        architecture = str(self.options.architecture)
        if architecture != "NONE":
            tc.cache_variables[f"Kokkos_ARCH_{architecture.upper()}"] = True
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "Kokkos")
        self.cpp_info.set_property("cmake_target_name", "Kokkos::kokkos")
        self.cpp_info.libs = [
            "kokkosalgorithms",
            "kokkoscontainers",
            "kokkoscore",
        ]
        self.cpp_info.system_libs = ["dl", "pthread"]
        if str(self.options.backend) in ("openmp", "cuda_openmp"):
            self.cpp_info.cxxflags.append("-fopenmp")
            self.cpp_info.sharedlinkflags.append("-fopenmp")
            self.cpp_info.exelinkflags.append("-fopenmp")
        if str(self.options.backend) == "sycl":
            # CMakeDeps reconstructs the target instead of loading Kokkos' own
            # exported target. Propagate SYCL compilation AND device linking.
            self.cpp_info.cxxflags.append("-fsycl")
            self.cpp_info.sharedlinkflags.append("-fsycl")
            self.cpp_info.exelinkflags.append("-fsycl")
        self.cpp_info.defines.append(
            "C8_KOKKOS_PACKAGE_BACKEND_" + str(self.options.backend).upper()
        )
        architecture = str(self.options.architecture).upper()
        self.cpp_info.defines.append(
            "C8_KOKKOS_PACKAGE_ARCH_" + architecture
        )
