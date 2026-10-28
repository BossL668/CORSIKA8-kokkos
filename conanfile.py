from conan import ConanFile
from conan.tools.cmake import cmake_layout,CMakeToolchain, CMakeDeps

class Pkg(ConanFile):
    generators = "CMakeDeps", #"CMakeToolchain",
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "with_proposal_export": [True, False],
        "with_kokkos": [True, False],
        "kokkos_backend": ["openmp", "cuda", "cuda_openmp", "hip", "sycl"],
        "kokkos_architecture": ["ANY"],
    }
    default_options = {
        "with_proposal_export": False,
        "with_kokkos": False,
        "kokkos_backend": "openmp",
        "kokkos_architecture": "NONE",
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
        if self.options.with_kokkos:
            self.options['kokkos'].backend = self.options.kokkos_backend
            self.options['kokkos'].architecture = self.options.kokkos_architecture
        self.options['arrow'].with_boost = True
        self.options['arrow'].parquet = True
        self.options['arrow'].with_thrift = True

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
        if self.options.with_kokkos or self.options.with_proposal_export:
            self.requires("nlohmann_json/3.11.3")
            self.requires("openssl/3.5.2")
        if self.options.with_kokkos:
            self.requires("kokkos/4.7.03@c8gpu/stable")
        if self.options.with_proposal_export:
            # These read-only export extensions change the dependency ABI.
            # All consumers must use this isolated, revision-locked graph.
            self.requires(
                "proposal/7.6.2@c8gpu/stable#de991fa019e37dc07f04543cb2d626eb"
            )
            self.requires(
                "cubicinterpolation/0.1.5@c8gpu/stable#b5aca7d283bcf748b9c2a9fc7ec09b63",
                override=True,
            )
        else:
            self.requires("proposal/7.6.2")

    def build_requirements(self):
        self.tool_requires("readline/8.0")
        self.tool_requires("bison/[>1.0]")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.absolute_paths = True
        tc.variables["CONAN_CMAKE_DIR"] = self.generators_folder
        tc.variables["CORSIKA_ENABLE_PROPOSAL_EXPORT"] = bool(
            self.options.with_proposal_export
        )
        if self.options.with_kokkos:
            tc.variables["CORSIKA_ENABLE_KOKKOS"] = True
            tc.variables["CORSIKA_KOKKOS_BACKEND"] = str(self.options.kokkos_backend).upper()
            if str(self.options.kokkos_backend) in ("cuda", "cuda_openmp"):
                architectures = {"VOLTA70": "70", "VOLTA72": "72", "TURING75": "75",
                                 "AMPERE80": "80", "AMPERE86": "86", "ADA89": "89",
                                 "HOPPER90": "90", "BLACKWELL120": "120"}
                architecture = str(self.options.kokkos_architecture).upper()
                if architecture not in architectures:
                    raise ValueError("Select an explicit supported CUDA architecture")
                tc.variables["CMAKE_CUDA_ARCHITECTURES"] = architectures[architecture]
        tc.generate()
