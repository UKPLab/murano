"""Paper metadata, search, impact score, and code links for the murano scout.

Stdlib only, so it runs anywhere the loop runs. Every subcommand prints JSON
(``search``, ``meta`` and ``impact`` print JSON Lines, one paper per line,
ready for ``ledger.py filter``).

    python paper.py search "activation patching" --since 2026-10-01
    python paper.py search "logit lens" --limit 100             # classics: by relevance
    python paper.py meta 2211.00593 2310.01405 DOI:10.18653/v1/2024.emnlp-main.965
    python paper.py impact 2211.00593 2310.01405 [--repo 2310.01405=owner/name] [--viral ID]
    python paper.py code-links 2211.00593

Every source is keyless: the arXiv API (search, metadata), OpenAlex (citations,
venues; ``--source openalex`` search), Hugging Face papers (the author-linked
repo) and GitHub (repo stars). Keyless OpenAlex has a small daily budget per
IP (about 100 searches), so the scout spends it on ``impact`` only. No source gives reliable citation counts for arXiv ML papers
without a key: OpenAlex barely parses preprint reference lists and undercounts
them by an order of magnitude. ``impact`` therefore reads OpenAlex citations as
a lower bound and adds adoption signals (top venue, GitHub stars of the
official code). Set ``OPENALEX_API_KEY`` or ``GITHUB_TOKEN`` for higher limits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date

OPENALEX = "https://api.openalex.org/works"
ARXIV = "https://export.arxiv.org/api/query"
_ATOM = {"a": "http://www.w3.org/2005/Atom"}

# Venues whose acceptance alone counts as an impact signal. Long names match as
# substrings; abbreviations must match as whole words ("acl" must not hit "eacl").
TOP_VENUE_NAMES = (
    "neural information processing systems",
    "international conference on machine learning",
    "learning representations",
    "association for computational linguistics",
    "empirical methods in natural language processing",
    "transactions on machine learning research",
    "conference on language modeling",
)
TOP_VENUE_ABBREVS = {
    "neurips",
    "nips",
    "icml",
    "iclr",
    "acl",
    "emnlp",
    "naacl",
    "tmlr",
    "colm",
}
# GitHub stars of the official code, calibrated on papers whose citation-based
# Impact is known (RepE 1056 -> 3, Tuned Lens 622 -> 3, Transcoders 214 -> 2).
# ponytail: fixed thresholds; recalibrate if star counts inflate across the field.
STARS_FOR = {3: 500, 2: 150, 1: 30}

_ARXIV = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
_GITHUB = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")


def normalize_id(raw: str) -> str:
    """Return ``ARXIV:<id>`` or ``DOI:<doi>`` for ``raw``, else ``raw`` unchanged.

    Accepts arXiv IDs with or without version, prefix or URL, and DOIs with or
    without ``https://doi.org/``.
    """
    s = raw.strip()
    low = s.lower()
    if low.startswith("doi:") or "doi.org/" in low or low.startswith("10."):
        doi = re.sub(r"^(doi:|https?://(dx\.)?doi\.org/)", "", low)
        return f"DOI:{doi}"
    m = _ARXIV.search(s)
    if m and ("arxiv" in low or re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", s)):
        return f"ARXIV:{m.group(1)}"
    return s


def norm_title(title: str | None) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).split())


class SourceUnavailable(RuntimeError):
    """A source kept answering 429 or could not resolve the paper."""


def _get(url: str, *, tries: int = 4) -> bytes:
    headers = {"User-Agent": "murano-scout/1"}
    if "api.github.com" in url and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=30
            ) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            if attempt < tries - 1:
                time.sleep(2 ** (attempt + 1))
    raise SourceUnavailable(f"rate-limited: {url}")


def _openalex(path: str = "", **params: str) -> dict:
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    return json.loads(_get(f"{OPENALEX}{path}?{urllib.parse.urlencode(params)}"))


def _openalex_candidate(w: dict) -> dict:
    doi = (w.get("doi") or "").lower().removeprefix("https://doi.org/") or None
    locs = w.get("locations") or []
    urls = [doi or ""] + [loc.get("landing_page_url") or "" for loc in locs]
    m = next(
        (m for u in urls if "arxiv" in u.lower() and (m := _ARXIV.search(u))), None
    )
    venues = [(loc.get("source") or {}).get("display_name") for loc in locs]
    return {
        "arxiv": m.group(1) if m else None,
        "doi": doi,
        "title": w.get("title"),
        "year": w.get("publication_year"),
        "publicationDate": w.get("publication_date"),
        "venue": next((v for v in venues if v), None),
        "citationCount": w.get("cited_by_count"),
    }


def _arxiv_query(terms: str | None, *, ids: list[str] | None = None, **params: str):
    q = dict(params)
    if terms:
        q["search_query"] = terms
    if ids:
        q["id_list"] = ",".join(ids)
        q["max_results"] = str(len(ids))
    root = ET.fromstring(_get(f"{ARXIV}?{urllib.parse.urlencode(q)}"))
    for entry in root.findall("a:entry", _ATOM):
        published = (entry.findtext("a:published", "", _ATOM) or "")[:10]
        m = _ARXIV.search(entry.findtext("a:id", "", _ATOM) or "")
        if not m:  # arXiv answers an unknown id with an error entry
            continue
        yield (
            entry,
            {
                "arxiv": m.group(1),
                "doi": None,
                "title": " ".join((entry.findtext("a:title", "", _ATOM) or "").split()),
                "year": int(published[:4]) if published else None,
                "publicationDate": published or None,
                "venue": "arXiv",
                "citationCount": None,
            },
        )


def search_openalex(query: str, since: str | None, limit: int) -> list[dict]:
    """Title-and-abstract search over OpenAlex, most relevant first."""
    # Same AND semantics as search_arxiv; relevance order, since citation order
    # surfaces long off-topic abstracts that merely contain every word.
    terms = " AND ".join(f'"{t}"' if " " in t else t for t in shlex.split(query))
    filters = [f"title_and_abstract.search:{terms}"]
    if since:
        filters.append(f"from_publication_date:{since}")
    data = _openalex(filter=",".join(filters), per_page=str(min(limit, 200)))
    return [_openalex_candidate(w) for w in data.get("results") or []]


def search_arxiv(query: str, since: str | None, limit: int) -> list[dict]:
    """ML/NLP preprints: newest first with ``since``, most relevant first without it."""
    # arXiv ORs bare words; AND them and stay inside the ML/NLP categories.
    terms = " AND ".join(
        f'all:"{t}"' if " " in t else f"all:{t}" for t in shlex.split(query)
    )
    rows = _arxiv_query(
        f"({terms}) AND (cat:cs.LG OR cat:cs.CL OR cat:cs.AI)",
        sortBy="submittedDate" if since else "relevance",
        sortOrder="descending",
        max_results=str(limit),
    )
    return [
        row for _, row in rows if not since or (row["publicationDate"] or "") >= since
    ]


def _openalex_meta(doi: str) -> dict:
    try:
        w = _openalex(f"/{urllib.parse.quote(f'doi:{doi}', safe=':/')}")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        raise SourceUnavailable(f"OpenAlex has no DOI {doi}") from e
    out = _openalex_candidate(w)
    inv = w.get("abstract_inverted_index") or {}
    words = {i: word for word, idxs in inv.items() for i in idxs}
    out["abstract"] = " ".join(words[i] for i in sorted(words)) or None
    out["authors"] = [
        (a.get("author") or {}).get("display_name") for a in w.get("authorships") or []
    ]
    out["url"] = w.get("id")
    return out


def meta(paper_ids: list[str], today: date | None = None) -> list[dict]:
    """Title, abstract, authors and date: one arXiv call for all arXiv IDs, OpenAlex per DOI."""
    nids = [normalize_id(p) for p in paper_ids]
    aids = [n.removeprefix("ARXIV:") for n in nids if n.startswith("ARXIV:")]
    found = {}
    if aids:
        for entry, row in _arxiv_query(None, ids=aids):
            row["abstract"] = " ".join(
                (entry.findtext("a:summary", "", _ATOM) or "").split()
            )
            row["authors"] = [
                a.findtext("a:name", "", _ATOM)
                for a in entry.findall("a:author", _ATOM)
            ]
            row["url"] = f"https://arxiv.org/abs/{row['arxiv']}"
            found[f"ARXIV:{row['arxiv']}"] = row
    out = []
    for raw, nid in zip(paper_ids, nids):
        if nid.startswith("DOI:"):
            row = _openalex_meta(nid.removeprefix("DOI:"))
        elif nid in found:
            row = found[nid]
        else:
            raise SourceUnavailable(f"cannot resolve {raw}")
        row["months"] = months_since(row["publicationDate"], row["year"], today)
        out.append(row)
    return out


def openalex_records(title: str) -> list[dict]:
    """Every OpenAlex record (preprint, venue version, ...) with exactly this title."""
    data = _openalex(filter=f"title.search:{norm_title(title)}", per_page="25")
    return [
        w
        for w in data.get("results") or []
        if norm_title(w.get("title")) == norm_title(title)
    ]


def repo_stars(arxiv_id: str | None, repo: str | None) -> tuple[str | None, int | None]:
    """Stars of ``repo``, else of the repo the authors linked on Hugging Face papers."""
    if not repo and arxiv_id:
        try:
            hf = json.loads(_get(f"https://huggingface.co/api/papers/{arxiv_id}"))
        except urllib.error.HTTPError:
            return None, None
        repo = (
            "/".join((hf.get("githubRepo") or "").rstrip("/").split("/")[-2:]) or None
        )
    if not repo:
        return None, None
    gh = json.loads(_get(f"https://api.github.com/repos/{repo}"))
    return repo, gh.get("stargazers_count")


def months_since(
    pub_date: str | None, year: int | None, today: date | None = None
) -> float:
    today = today or date.today()
    if pub_date:
        d = date.fromisoformat(pub_date[:10])
    elif year:
        d = date(year, 7, 1)
    else:
        return 0.0
    return max((today - d).days / 30.44, 1.0)


def is_top_venue(*venues: str | None) -> bool:
    for v in venues:
        if not v:
            continue
        low = v.lower()
        if any(name in low for name in TOP_VENUE_NAMES):
            return True
        if TOP_VENUE_ABBREVS & set(re.findall(r"[a-z]+", low)):
            return True
    return False


def impact_score(m: dict, viral: bool = False) -> dict:
    """Deterministic 0-3 Impact score; the anchors live in murano-scout/SKILL.md.

    ``citationCount`` is a lower bound (OpenAlex), so stars of the official code
    (``githubStars``) can lift the score where citations are undercounted.
    """
    c = m.get("citationCount") or 0
    stars = m.get("githubStars") or 0
    months = m.get("months") or months_since(m.get("publicationDate"), m.get("year"))
    rate = c / months
    top = is_top_venue(m.get("venue"), m.get("publicationVenue"))
    hits = {
        3: [
            (
                f"{c} citations, {rate:.1f}/month",
                c >= 300 or (months < 18 and rate >= 10),
            ),
            (f"{stars} GitHub stars", stars >= STARS_FOR[3]),
        ],
        2: [
            (f"{c} citations", c >= 100),
            (
                f"{rate:.1f} citations/month at {months:.0f} months",
                months < 18 and c >= 20 and rate >= 20 / 6,
            ),
            ("top venue", top),
            ("viral signal", viral),
            (f"{stars} GitHub stars", stars >= STARS_FOR[2]),
        ],
        1: [
            (f"{c} citations", c >= 20),
            (f"{stars} GitHub stars", stars >= STARS_FOR[1]),
        ],
    }
    score = next((s for s in (3, 2, 1) if any(hit for _, hit in hits[s])), 0)
    reasons = (
        [r for r, hit in hits[score] if hit]
        if score
        else [f"{c} citations, {stars} stars at {months:.0f} months"]
    )
    # Too new to judge is not the same as unimportant: the scout defers these
    # instead of rejecting them, and ledger.py due brings them back later.
    return {
        "impact": score,
        "reasons": reasons,
        "deferrable": score < 2 and months < 12,
    }


def impact(paper_ids: list[str], repos: dict[str, str], viral: set[str]) -> list[dict]:
    """Impact per paper: one arXiv call for all metadata, then OpenAlex and stars per paper."""
    out = []
    for pid, m in zip(paper_ids, meta(paper_ids)):
        try:
            records = openalex_records(m["title"])
            m["repo"], m["githubStars"] = repo_stars(m.get("arxiv"), repos.get(pid))
        except SourceUnavailable as e:
            out.append({"id": pid, "error": "source_unavailable", "detail": str(e)})
            continue
        venues = {
            (loc.get("source") or {}).get("display_name")
            for w in records
            for loc in w.get("locations") or []
        }
        m["citationCount"] = sum(w.get("cited_by_count") or 0 for w in records)
        m["publicationVenue"] = "; ".join(sorted(v for v in venues if v)) or None
        out.append(
            {
                "id": pid,
                "title": m["title"],
                "citations_lower_bound": m["citationCount"],
                "venues": m["publicationVenue"],
                "repo": m["repo"],
                "stars": m["githubStars"],
                **impact_score(m, pid in viral),
            }
        )
    return out


def code_links(arxiv_id: str) -> list[str]:
    """GitHub repos linked from the arXiv abstract page, HTML paper, or HF papers."""
    aid = normalize_id(arxiv_id).removeprefix("ARXIV:")
    found: list[str] = []
    for url in (f"https://arxiv.org/abs/{aid}", f"https://arxiv.org/html/{aid}"):
        try:
            text = _get(url).decode("utf-8", "replace")
        except urllib.error.HTTPError:
            continue
        found += [
            f"https://github.com/{o}/{r.removesuffix('.git').rstrip('.')}"
            for o, r in _GITHUB.findall(text)
        ]
    try:
        hf = json.loads(_get(f"https://huggingface.co/api/papers/{aid}"))
        if hf.get("githubRepo"):
            found.append(hf["githubRepo"].rstrip("/"))
    except (urllib.error.HTTPError, json.JSONDecodeError):
        pass
    # The arXiv page chrome and its HTML renderer link to their own repos; never the paper's code.
    chrome = ("https://github.com/arxiv", "https://github.com/brucemiller/latexml")
    return sorted({u for u in found if not u.lower().startswith(chrome)})


def main(argv: list[str] | None = None) -> int | None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--since", help="YYYY-MM-DD; only papers published on or after")
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--source", choices=("arxiv", "openalex"), default="arxiv")
    sub.add_parser("meta").add_argument("ids", nargs="+")
    i = sub.add_parser("impact")
    i.add_argument("ids", nargs="+")
    i.add_argument(
        "--repo",
        action="append",
        default=[],
        metavar="ID=OWNER/NAME",
        help="official code of paper ID, when HF papers lacks it (repeatable)",
    )
    i.add_argument(
        "--viral",
        action="append",
        default=[],
        metavar="ID",
        help="paper ID has an agent-verified viral signal (repeatable)",
    )
    sub.add_parser("code-links").add_argument("id")
    a = ap.parse_args(argv)
    try:
        _run(a)
    except SourceUnavailable as e:
        # The scout treats this as "not screened": no ledger row, retry next tick.
        print(json.dumps({"error": "source_unavailable", "detail": str(e)}))
        return 2


def _run(a: argparse.Namespace) -> None:
    if a.cmd == "search":
        if a.source == "arxiv":
            rows = search_arxiv(a.query, a.since, a.limit)
        else:
            rows = search_openalex(a.query, a.since, a.limit)
        for row in rows:
            print(json.dumps(row))
    elif a.cmd == "meta":
        for row in meta(a.ids):
            print(json.dumps(row))
    elif a.cmd == "impact":
        repos = dict(r.split("=", 1) for r in a.repo)
        for row in impact(a.ids, repos, set(a.viral)):
            print(json.dumps(row))
    else:
        print(json.dumps(code_links(a.id), indent=2))


if __name__ == "__main__":
    sys.exit(main())
