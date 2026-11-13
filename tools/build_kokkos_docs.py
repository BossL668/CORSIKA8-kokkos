#!/usr/bin/env python3
"""Build the standalone developer guide and check its internal links."""

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlsplit


class PageLinks(HTMLParser):
    """Collect local links, asset references and HTML anchors."""

    def __init__(self):
        super().__init__()
        self.anchors = set()
        self.links = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if attributes.get("id"):
            self.anchors.add(attributes["id"])
        if tag == "a" and attributes.get("name"):
            self.anchors.add(attributes["name"])
        for key in ("href", "src"):
            if attributes.get(key):
                self.links.append(attributes[key])


def check_site(output):
    pages = {}
    for path in sorted(output.rglob("*.html")):
        parser = PageLinks()
        parser.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = parser

    errors = []
    checked = 0
    for path, page in pages.items():
        for link in page.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            if parts.path.startswith("/"):
                target = output / unquote(parts.path.lstrip("/"))
            else:
                target = path.parent / unquote(parts.path) if parts.path else path
            if target.is_dir():
                target = target / "index.html"
            target = target.resolve()
            checked += 1
            if not target.is_file():
                errors.append(f"{path.relative_to(output)}: missing target {link}")
            elif parts.fragment and target in pages:
                if unquote(parts.fragment) not in pages[target].anchors:
                    errors.append(f"{path.relative_to(output)}: missing anchor {link}")

    report = {"html_pages": len(pages), "internal_references": checked,
              "errors": errors, "output": str(output)}
    (output.parent / "site-check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError("Internal link check failed:\n" + "\n".join(errors))
    print(f"Checked {len(pages)} HTML pages and {checked} internal references.")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=root / "build/kokkos-docs/html")
    args = parser.parse_args()
    output = args.output.resolve()
    source = root / "documentation/kokkos"
    subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "-n", "-W",
         "--keep-going", "-j", "1", str(source), str(output)], check=True
    )
    (output / ".nojekyll").touch()
    check_site(output)
    print(f"Preview: {output / 'index.html'}")


if __name__ == "__main__":
    main()
