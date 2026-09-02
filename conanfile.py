from conan import ConanFile
from conan.tools.cmake import cmake_layout,CMakeToolchain, CMakeDeps

class Pkg(ConanFile):
    generators = "CMakeDeps", #"CMakeToolchain",
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "with_kokkos": [True, False],
        "kokkos_backend": ["openmp", "cuda", "hip", "sycl"],
        "kokkos_architecture": ["ANY"],
    }
    default_options = {
		'with_kokkos': False,
		'kokkos_backend': 'openmp',
		'kokkos_architecture': 'NONE',
		'readline*:shared': 'True',
		'arrow*:shared': 'False',
		'arrow*:parquet': 'True',
		'arrow*:fPIC': 'False',
		'arrow*:with_re2': 'True',
		'arrow*:with_protobuf': 'False',
		'arrow*:with_openssl': 'False',
		'arrow*:with_gflags': 'False',
		'arrow*:with_glog': 'False',
		'arrow*:with_grpc': 'False',
		'arrow*:with_utf8proc': 'False',
		'arrow*:with_zstd': 'False',
		'arrow*:with_bz2': 'False',
		'arrow*:with_lz4': 'True',
		'arrow*:with_thrift': 'True',
		'arrow*:with_boost': 'True',
		'boost*:without_container': 'True',
		'boost*:without_context': 'True',
		'boost*:without_contract': 'True',
		'boost*:without_coroutine': 'True',
		'boost*:without_date_time': 'True',
		'boost*:without_fiber': 'True',
		'boost*:without_filesystem': 'False',
		'boost*:without_graph': 'True',
		'boost*:without_graph_parallel': 'True',
		'boost*:without_iostreams': 'False',
		'boost*:without_json': 'True',
		'boost*:without_locale': 'True',
		'boost*:without_log': 'True',
		'boost*:without_math': 'False',
		'boost*:without_mpi': 'True',
		'boost*:without_nowide': 'True',
		'boost*:without_program_options': 'True',
		'boost*:without_python': 'True',
		'boost*:without_serialization': 'False',
		'boost*:without_stacktrace': 'True',
		'boost*:without_system': 'False',
		'boost*:without_test': 'True',
		'boost*:without_thread': 'True',
		'boost*:without_timer': 'True',
		'boost*:without_type_erasure': 'True',
		'boost*:without_wave': 'True'
}

    def configure(self):
        self.options['arrow'].with_boost = True
        self.options['arrow'].parquet = True
        self.options['arrow'].with_thrift = True
        if self.options.with_kokkos:
            self.options['kokkos'].backend = self.options.kokkos_backend
            self.options['kokkos'].architecture = self.options.kokkos_architecture
        
    def requirements(self):
        self.requires("spdlog/1.14.1", force=True)
        self.requires("catch2/3.6.0")
        self.requires("bzip2/1.0.8")
        self.requires("boost/1.85.0", force=True)
        self.requires("eigen/3.4.0")
        self.requires("zlib/1.3.1")
        self.requires("yaml-cpp/0.8.0")
        self.requires("cli11/1.9.1")
        self.requires("arrow/16.1.0")
        # Version-locked, read-only export API for --gpu-physics-source
        # proposal-native. The patch does not change the scalar PROPOSAL path.
        self.requires("proposal/7.6.2@c8gpu/stable")
        if self.options.with_kokkos:
            self.requires("kokkos/4.7.03@c8gpu/stable")

    def build_requirements(self):
        self.tool_requires("readline/8.0")
        self.tool_requires("bison/[>1.0]")
        
    def generate(self):
        tc = CMakeToolchain(self)
        tc.absolute_paths = True
        if self.options.with_kokkos:
            tc.cache_variables["CORSIKA_KOKKOS_BACKEND"] = str(
                self.options.kokkos_backend
            ).upper()
            architecture = str(self.options.kokkos_architecture).upper()
            tc.cache_variables["CORSIKA_KOKKOS_ARCHITECTURE"] = architecture
            cuda_architectures = {
                "KEPLER35": "35",
                "MAXWELL50": "50",
                "MAXWELL52": "52",
                "MAXWELL53": "53",
                "PASCAL60": "60",
                "PASCAL61": "61",
                "VOLTA70": "70",
                "VOLTA72": "72",
                "TURING75": "75",
                "AMPERE80": "80",
                "AMPERE86": "86",
                "ADA89": "89",
                "HOPPER90": "90",
            }
            if str(self.options.kokkos_backend) == "cuda":
                if architecture not in cuda_architectures:
                    raise ValueError(
                        "Kokkos CUDA requires a supported explicit "
                        "kokkos_architecture (for example ADA89)"
                    )
                tc.cache_variables["CORSIKA_KOKKOS_CUDA_ARCHITECTURES"] = (
                    cuda_architectures[architecture]
                )
        tc.generate()
        
