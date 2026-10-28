from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.files import (
    apply_conandata_patches,
    copy,
    export_conandata_patches,
    get,
    rmdir,
)
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.microsoft import check_min_vs, is_msvc
import os


required_conan_version = ">=2.0.0"


class CubicInterpolationC8GpuConan(ConanFile):
    """Version-locked CubicInterpolation with read-only table exports."""

    name = "cubicinterpolation"
    version = "0.1.5"
    user = "c8gpu"
    channel = "stable"
    description = "Cubic interpolation with C8 GPU read-only export API"
    license = "MIT"
    homepage = "https://github.com/tudo-astroparticlephysics/cubic_interpolation"
    package_type = "library"
    settings = "os", "arch", "compiler", "build_type"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

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
        self.requires("boost/1.85.0")
        self.requires("eigen/3.4.0")

    def validate(self):
        required = ("filesystem", "math", "serialization")
        missing = any(
            getattr(self.dependencies["boost"].options, f"without_{name}", True)
            for name in required
        )
        if self.dependencies["boost"].options.header_only or missing:
            raise ConanInvalidConfiguration(
                f"{self.ref} requires Boost filesystem, math and serialization"
            )
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, "14")
        if not check_min_vs(self, 192, raise_invalid=False):
            raise ConanInvalidConfiguration("Visual Studio 2019 or newer is required")
        if is_msvc(self) and self.options.shared:
            raise ConanInvalidConfiguration("shared MSVC builds are unsupported")

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)

    def generate(self):
        toolchain = CMakeToolchain(self)
        toolchain.variables["BUILD_EXAMPLE"] = False
        toolchain.variables["BUILD_DOCUMENTATION"] = False
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
            "LICENSE",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )
        CMake(self).install()
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "CubicInterpolation")
        self.cpp_info.set_property(
            "cmake_target_name", "CubicInterpolation::CubicInterpolation"
        )
        self.cpp_info.libs = ["CubicInterpolation"]
        self.cpp_info.requires = [
            "boost::headers",
            "boost::filesystem",
            "boost::math",
            "boost::serialization",
            "eigen::eigen",
        ]
