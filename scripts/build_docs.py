"""Execute the example notebooks and stage them for the mkdocs site.

Notebooks are committed with their outputs stripped, so the rendered site is built
from freshly executed copies. Each notebook runs top to bottom at full budget (use a
GPU), from its own directory so relative data paths and ``plotting.py`` resolve,
and the executed copy is written under ``build/docs/`` — gitignored, so outputs
never reach the main branch. The README becomes the site's home page.

    python scripts/build_docs.py                 # execute new/changed notebooks
    python scripts/build_docs.py --rerun         # execute every notebook
    python scripts/build_docs.py examples/gaussian_processes/Hawkes_process.ipynb
    mkdocs serve                                 # preview at http://127.0.0.1:8000
    mkdocs gh-deploy --force --no-history        # publish to the gh-pages branch

A notebook is skipped when its executed copy is newer than the source; pass
``--rerun`` after changing the library.
"""

import argparse
import re
import time
from pathlib import Path

import nbclient
import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DOCS = ROOT / "build" / "docs"
REPO_URL = "https://github.com/FlowInference/flow-inference/blob/main/"


def execute(source: Path, target: Path) -> None:
    nb = nbformat.read(source, as_version=4)
    nbclient.NotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(source.parent)}},
    ).execute()
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, target)


def write_index() -> None:
    # Links into examples/*.ipynb resolve to the rendered pages; everything else
    # in the README (LICENSE, CITATION.cff, examples/) points back to GitHub.
    readme = (ROOT / "README.md").read_text()
    readme = re.sub(
        r"\]\((?!https?://|#)([^)]+?)(?<!\.ipynb)\)",
        lambda m: f"]({REPO_URL}{m.group(1)})",
        readme,
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "index.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="*", type=Path, help="default: all")
    parser.add_argument("--rerun", action="store_true", help="ignore cached copies")
    args = parser.parse_args()

    sources = [p.resolve() for p in args.notebooks] or sorted(EXAMPLES.rglob("*.ipynb"))
    sources = [p for p in sources if ".ipynb_checkpoints" not in p.parts]
    write_index()
    for source in sources:
        rel = source.relative_to(ROOT)
        target = DOCS / rel
        if (
            not args.rerun
            and target.exists()
            and target.stat().st_mtime > source.stat().st_mtime
        ):
            print(f"cached   {rel}")
            continue
        print(f"running  {rel}", flush=True)
        start = time.perf_counter()
        execute(source, target)
        print(f"done     {rel} ({time.perf_counter() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
