"""Link unrendered API types to their declarations after normal Sphinx lookup."""

from collections import defaultdict
from pathlib import Path
import os
import xml.etree.ElementTree as ET

from docutils import nodes


# Native declarations outside the XML input set, checked at the guide baseline.
# These are exact symbol mappings, not suffix or wildcard fallback rules.
NATIVE_DECLARATIONS = {
    "corsika::HEPEnergyType": ("corsika/framework/core/PhysicalUnits.hpp", 83),
    "corsika::Code": ("corsika/framework/core/ParticleProperties.hpp", 84),
    "corsika::CoordinateSystemPtr": ("corsika/framework/geometry/CoordinateSystem.hpp", 34),
    "corsika::OutputManager": ("corsika/output/OutputManager.hpp", 21),
    "corsika::AtmosphereId": ("corsika/media/CORSIKA7Atmospheres.hpp", 28),
    "corsika::AtmosphereId::USStdBK": ("corsika/media/CORSIKA7Atmospheres.hpp", 48),
    "corsika::Environment": ("corsika/media/Environment.hpp", 37),
    "corsika::VolumeTreeNode": ("corsika/media/VolumeTreeNode.hpp", 18),
    "corsika::radio::RadioTrackSegment": ("corsika/modules/radio/RadioTrackSegment.hpp", 16),
    "corsika::proposal::NativeInteractionCalculatorView": (
        "corsika/modules/proposal/NativeCalculatorView.hpp", 36),
    "corsika::proposal::NativeContinuousCalculatorView": (
        "corsika/modules/proposal/NativeCalculatorView.hpp", 101),
    "corsika::transport::HistoryId": ("corsika/stack/TransportIdentityStackExtension.hpp", 20),
    "corsika::transport::StepId": ("corsika/stack/TransportIdentityStackExtension.hpp", 21),
    "corsika::transport::Generation": ("corsika/stack/TransportIdentityStackExtension.hpp", 22),
    "corsika::kokkos::RuntimeLease": ("corsika/kokkos/runtime/RuntimeConfig.hpp", 17),
}

NAMESPACE_DIRECTORIES = {
    "corsika": "corsika",
    "corsika::kokkos": "corsika/kokkos",
    "corsika::kokkos::application": "corsika/kokkos/application",
    "corsika::kokkos::coordinator": "corsika/kokkos/scheduling",
    "corsika::kokkos::particle_transfer": "corsika/kokkos/data",
    "corsika::kokkos::radio": "corsika/kokkos/radio",
    "corsika::kokkos::radio::firn": "corsika/kokkos/radio/firn",
    "corsika::radio": "corsika/modules/radio",
    "corsika::proposal": "corsika/modules/proposal",
}

EXTERNAL_DECLARATIONS = {
    "CLI": "https://cliutils.github.io/CLI11/",
    "CLI::App": "https://cliutils.github.io/CLI11/class_c_l_i_1_1_app.html",
    "Kokkos": "https://kokkos.org/kokkos-core-wiki/API/core-index.html",
    "Kokkos::View": "https://kokkos.org/kokkos-core-wiki/API/core/view/view.html",
    "boost": "https://www.boost.org/libraries/",
    "boost::optional": "https://www.boost.org/doc/libs/1_86_0/libs/optional/doc/html/index.html",
    "boost::span": "https://www.boost.org/doc/libs/1_86_0/libs/core/doc/html/core/span.html",
    "std::filesystem": "https://en.cppreference.com/w/cpp/filesystem.html",
    "std::filesystem::path": "https://en.cppreference.com/w/cpp/filesystem/path.html",
}


def untemplated(name):
    """Keep qualification while removing balanced template argument lists."""
    output = []
    depth = 0
    for character in name:
        if character == "<":
            depth += 1
        elif character == ">" and depth:
            depth -= 1
        elif not depth:
            output.append(character)
    return "".join(output).strip().removeprefix("typename ")


def text(element):
    return "".join(element.itertext()) if element is not None else ""


class DeclarationLinks:
    """Resolve exact names and lexical scopes; refuse ambiguous declarations."""

    def __init__(self, xml_directory, revision):
        self.base = "https://github.com/BossL668/CORSIKA8-kokkos/"
        self.revision = revision
        self.names = defaultdict(set)
        self.refids = defaultdict(set)
        self.aliases = {}
        for path in sorted(Path(xml_directory).glob("*.xml")):
            if path.name == "index.xml":
                continue
            compound = ET.parse(path).find("compounddef")
            if compound is None:
                continue
            kind = compound.get("kind")
            scope = untemplated(compound.findtext("compoundname", ""))
            if kind in ("class", "struct", "union", "namespace"):
                self.add(scope, compound.get("id"), compound.find("location"))
            for member in compound.findall("./sectiondef/memberdef"):
                qualified = member.findtext("qualifiedname")
                if not qualified and kind in ("class", "struct", "union", "namespace"):
                    qualified = scope + "::" + member.findtext("name", "")
                if not qualified:
                    continue
                qualified = untemplated(qualified)
                self.add(qualified, member.get("id"), member.find("location"))
                if member.get("kind") == "typedef":
                    self.aliases[qualified] = untemplated(text(member.find("type")))
                if member.get("kind") == "enum":
                    for value in member.findall("enumvalue"):
                        name = qualified + "::" + value.findtext("name", "")
                        self.add(name, value.get("id"), member.find("location"))
        for name, (path, line) in NATIVE_DECLARATIONS.items():
            self.names[name].add(self.source_url(path, line))
        self.names["corsika::transport"].add(
            self.source_url("corsika/stack/TransportIdentityStackExtension.hpp", 18))
        self.aliases["corsika::kokkos::coordinator::fs"] = "std::filesystem"
        for name, path in NAMESPACE_DIRECTORIES.items():
            self.names[name] = {self.base + "tree/" + revision + "/" + path}
        for name, url in EXTERNAL_DECLARATIONS.items():
            # A native forward declaration of a dependency's namespace does
            # not replace the dependency's own reference documentation.
            self.names[name] = {url}

    def source_url(self, path, line=None):
        url = self.base + "blob/" + self.revision + "/" + path
        return url + "#L" + str(line) if line else url

    def add(self, name, refid, location):
        if location is None:
            return
        path = location.get("file", "")
        # XML is generated from repository-relative inputs. Reject external paths.
        if not path.startswith("corsika/") or ".." in Path(path).parts:
            return
        url = self.source_url(path, location.get("line"))
        self.names[name].add(url)
        if refid:
            self.refids[refid].add(url)

    @staticmethod
    def unique(candidates):
        return next(iter(candidates)) if len(candidates) == 1 else None

    def resolve(self, target, scope=(), visited=frozenset()):
        absolute = target.startswith("::")
        name = untemplated(target).lstrip(":")
        if (name, tuple(scope)) in visited:
            return None
        visited = visited | {(name, tuple(scope))}
        prefixes = [()] if absolute else [tuple(scope[:i]) for i in range(len(scope), -1, -1)]
        for prefix in prefixes:
            qualified = "::".join((*prefix, name))
            if qualified in self.names:
                return self.unique(self.names[qualified])
            if qualified in self.aliases:
                return self.resolve(self.aliases[qualified], prefix, visited)
            # Resolve dependent aliases such as topology_type::node_type from
            # their lexical declaration, never by a global suffix search.
            parts = name.split("::")
            for i in range(len(parts) - 1, 0, -1):
                alias = "::".join((*prefix, *parts[:i]))
                if alias in self.aliases:
                    expanded = self.aliases[alias] + "::" + "::".join(parts[i:])
                    return self.resolve(expanded, prefix, visited)
        return None


def initialize(app):
    xml = os.environ.get("C8_DOCS_DOXYGEN_XML")
    if not xml:
        project = app.config.breathe_default_project
        xml = app.config.breathe_projects.get(project)
    if xml:
        app._api_declaration_links = DeclarationLinks(xml, app.config.api_source_revision)


def missing_reference(app, env, node, content):
    links = getattr(app, "_api_declaration_links", None)
    if links is None:
        return None
    target = node.get("reftarget", "")
    if node.get("refdomain") == "cpp" and node.get("reftype") == "identifier":
        key = node.get("cpp:parent_key")
        scope = tuple(untemplated(str(item[0])) for item in getattr(key, "data", ()))
        url = links.resolve(target, scope)
    elif node.get("refdomain") == "std" and node.get("reftype") == "ref":
        url = links.unique(links.refids.get(target, set()))
    else:
        return None
    if url is None:
        return None
    return nodes.reference("", "", content, refuri=url, internal=False)


def setup(app):
    app.add_config_value("api_source_revision", "d5297c43c041a22b43c65c322d9b51bc7ec8e397", "env")
    app.connect("builder-inited", initialize)
    app.connect("missing-reference", missing_reference)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
