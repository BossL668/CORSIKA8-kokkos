import os

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import (
    apply_conandata_patches,
    copy,
    export_conandata_patches,
    get,
    rmdir,
)
from conan.tools.microsoft import is_msvc
from conan.tools.scm import Version


required_conan_version = ">=2.0.0"


class ProposalC8GpuConan(ConanFile):
    """PROPOSAL 7.6.2 with a const native-interpolation export API."""

    name = "proposal"
    version = "7.6.2"
    user = "c8gpu"
    channel = "stable"
    homepage = "https://github.com/tudo-astroparticlephysics/PROPOSAL"
    license = "LGPL-3.0"
    package_type = "library"
    description = "PROPOSAL with C8 GPU read-only interpolation exports"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_python": [True, False],
    }
    default_options = {"shared": False, "fPIC": True, "with_python": False}
    exports_sources = "patches/*"

    def export_sources(self):
        export_conandata_patches(self)

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self, src_folder="src")

    def requirements(self):
        self.requires(
            "cubicinterpolation/0.1.5@c8gpu/stable",
            transitive_headers=True,
            transitive_libs=True,
            # The exported POD layout is an ABI surface.  A new patched
            # CubicInterpolation package revision must never reuse an older
            # PROPOSAL binary with the same nominal package ID.
            package_id_mode="recipe_revision_mode",
        )
        self.requires(
            "spdlog/[>=1.11 <2]", transitive_headers=True, transitive_libs=True
        )
        self.requires("nlohmann_json/[~3.11]", transitive_headers=True)
        if self.options.with_python:
            self.requires("pybind11/2.10.1")

    def validate(self):
        if is_msvc(self) and self.options.shared:
            raise ConanInvalidConfiguration("shared MSVC builds are unsupported")
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, "14")
        minimum = {
            "Visual Studio": "15",
            "msvc": "191",
            "gcc": "5",
            "clang": "5",
            "apple-clang": "5",
        }.get(str(self.settings.compiler), False)
        if minimum and Version(self.settings.compiler.version) < minimum:
            raise ConanInvalidConfiguration("the selected compiler is too old")

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)

    def generate(self):
        toolchain = CMakeToolchain(self)
        toolchain.cache_variables["BUILD_TESTING"] = False
        toolchain.cache_variables["BUILD_PYTHON"] = self.options.with_python
        toolchain.cache_variables["BUILD_DOCUMENTATION"] = False
        toolchain.generate()
        CMakeDeps(self).generate()

    def build(self):
        apply_conandata_patches(self)
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(
            self,
            "LICENSE.md",
            self.source_folder,
            os.path.join(self.package_folder, "licenses"),
        )
        CMake(self).install()
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "PROPOSAL")
        self.cpp_info.set_property("cmake_target_name", "PROPOSAL::PROPOSAL")
        self.cpp_info.libs = ["PROPOSAL"]
