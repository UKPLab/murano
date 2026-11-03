"""Provenance check for a candidate reference implementation.

    python repo_check.py https://github.com/owner/repo [--authors "Kevin Wang,Neel Nanda"]

Prints JSON: the pinned commit SHA, the SPDX license and what it allows
(``adapt-ok``: code may be adapted with attribution; ``oracle-only``: read and
run it to produce test oracles, but write murano's code from the paper), and
the reputation signals the dossier's provenance table needs. Uses ``gh api``,
so it works wherever ``gh auth status`` does.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone

# Permissive licenses compatible with redistributing adapted code under
# Apache-2.0. Anything else (GPL family, MPL, no license, NOASSERTION) is
# oracle-only: copyleft or unknown terms must not leak into murano's source.
ADAPT_OK = {
    "MIT",
    "MIT-0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Apache-2.0",
    "ISC",
    "0BSD",
    "Unlicense",
    "CC0-1.0",
}


def license_verdict(spdx: str | None) -> str:
    return "adapt-ok" if spdx in ADAPT_OK else "oracle-only"


def parse_repo(url: str) -> tuple[str, str]:
    m = re.search(r"github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", url)
    if not m:
        raise ValueError(f"not a GitHub repository URL: {url!r}")
    return m.group(1), m.group(2).removesuffix(".git")


def owner_matches(owner_login: str, owner_name: str | None, authors: list[str]) -> bool:
    """True when an author's surname appears in the repo owner's login or display name."""
    hay = f"{owner_login} {owner_name or ''}".lower()
    return any(a.split()[-1].lower() in hay for a in authors if a.strip())


def _gh(path: str) -> dict:
    out = subprocess.run(
        ["gh", "api", path], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(out)


def check(url: str, authors: list[str]) -> dict:
    owner, name = parse_repo(url)
    repo = _gh(f"repos/{owner}/{name}")
    sha = _gh(f"repos/{owner}/{name}/commits/{repo['default_branch']}")["sha"]
    tree = _gh(f"repos/{owner}/{name}/git/trees/{sha}?recursive=1").get("tree", [])
    paths = [t["path"] for t in tree]
    spdx = (repo.get("license") or {}).get("spdx_id")
    pushed = datetime.fromisoformat(repo["pushed_at"].replace("Z", "+00:00"))
    owner_name = _gh(f"users/{repo['owner']['login']}").get("name") if authors else None
    return {
        "repo": repo["full_name"],
        "pinned": f"https://github.com/{repo['full_name']}/tree/{sha}",
        "sha": sha,
        "license": spdx,
        "verdict": license_verdict(spdx),
        "stars": repo["stargazers_count"],
        "archived": repo["archived"],
        "fork": repo["fork"],
        "months_since_push": round(
            (datetime.now(timezone.utc) - pushed).days / 30.44, 1
        ),
        "has_tests": any(
            re.search(r"(^|/)tests?/|(^|/)test_[^/]*\.py$", p) for p in paths
        ),
        "python_files": sum(p.endswith(".py") for p in paths),
        "owner_is_author": owner_matches(repo["owner"]["login"], owner_name, authors)
        if authors
        else None,
        "owner_type": repo["owner"]["type"],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("url")
    ap.add_argument("--authors", default="", help="comma-separated paper authors")
    a = ap.parse_args(argv)
    print(
        json.dumps(
            check(a.url, [s for s in a.authors.split(",") if s.strip()]), indent=2
        )
    )


if __name__ == "__main__":
    sys.exit(main())
