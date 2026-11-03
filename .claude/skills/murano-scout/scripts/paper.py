"""Paper metadata, search, impact score, and code links for the murano scout.

Stdlib only, so it runs anywhere the loop runs. Every subcommand prints JSON
(``search`` prints JSON Lines, one candidate per line, ready for
``ledger.py filter``).

    python paper.py search "activation patching" --since 2026-10-01
    python paper.py search "sparse autoencoder" --source arxiv --since 2026-10-01
    python paper.py search "logit lens" --source s2 --min-citations 100
    python paper.py meta ARXIV:2211.00593
    python paper.py impact ARXIV:2211.00593 [--viral]
    python paper.py code-links 2211.00593

Sources: OpenAlex (default search; fast, keyless) and the arXiv API cover the
recent window. Semantic Scholar does the citation-sorted classics sweep and is
the only source ``impact`` trusts: OpenAlex splits an arXiv paper's citations
across its preprint and venue records and undercounts them several-fold.
``meta`` falls back to OpenAlex (citation count withheld) when S2 throttles.

S2 runs keyless at the shared public rate; set ``S2_API_KEY`` for a dedicated
one, and optionally ``OPENALEX_API_KEY``. Field sets and ID formats follow the
S2 Graph API (the same ones allenai/asta-plugins' semantic-scholar skill uses).
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

S2 = "https://api.semanticscholar.org/graph/v1"
OPENALEX = "https://api.openalex.org/works"
FIELDS = (
    "paperId,externalIds,title,abstract,year,publicationDate,venue,"
    "publicationVenue,citationCount,influentialCitationCount,url,authors.name"
)
SEARCH_FIELDS = "paperId,externalIds,title,year,publicationDate,venue,citationCount"

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

_ARXIV = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
_GITHUB = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")


def normalize_id(raw: str) -> str:
    """Return an S2-style ID (``ARXIV:``, ``DOI:`` or a bare S2 paperId) for ``raw``.

    Accepts arXiv IDs with or without version, prefix or URL, DOIs with or
    without ``https://doi.org/``, and 40-hex S2 paperIds.
    """
    s = raw.strip()
    low = s.lower()
    if low.startswith("doi:") or "doi.org/" in low or low.startswith("10."):
        doi = re.sub(r"^(doi:|https?://(dx\.)?doi\.org/)", "", low)
        return f"DOI:{doi}"
    if re.fullmatch(r"[0-9a-f]{40}", low):
        return low
    m = _ARXIV.search(s)
    if m and ("arxiv" in low or re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", s)):
        return f"ARXIV:{m.group(1)}"
    return s


class S2Unavailable(RuntimeError):
    """Semantic Scholar kept answering 429. Keyless access is throttled to near zero."""


def _get(url: str, *, s2: bool = False, tries: int = 5) -> bytes:
    headers = {"User-Agent": "murano-scout/1"}
    if s2 and os.environ.get("S2_API_KEY"):
        headers["x-api-key"] = os.environ["S2_API_KEY"]
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
    if s2:
        raise S2Unavailable("Semantic Scholar rate-limited this client; set S2_API_KEY")
    raise RuntimeError(f"rate-limited: {url}")


def _candidate(p: dict) -> dict:
    ext = p.get("externalIds") or {}
    return {
        "paperId": p.get("paperId"),
        "arxiv": ext.get("ArXiv"),
        "doi": (ext.get("DOI") or "").lower() or None,
        "title": p.get("title"),
        "year": p.get("year"),
        "publicationDate": p.get("publicationDate"),
        "venue": p.get("venue"),
        "citationCount": p.get("citationCount"),
    }


def search_s2(
    query: str, since: str | None, min_citations: int, limit: int
) -> list[dict]:
    params = {"query": query, "fields": SEARCH_FIELDS, "sort": "citationCount:desc"}
    if since:
        params["publicationDateOrYear"] = f"{since}:"
    if min_citations:
        params["minCitationCount"] = str(min_citations)
    data = json.loads(
        _get(f"{S2}/paper/search/bulk?{urllib.parse.urlencode(params)}", s2=True)
    )
    return [_candidate(p) for p in (data.get("data") or [])[:limit]]


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
        "paperId": None,
        "arxiv": m.group(1) if m else None,
        "doi": doi,
        "title": w.get("title"),
        "year": w.get("publication_year"),
        "publicationDate": w.get("publication_date"),
        "venue": next((v for v in venues if v), None),
        "citationCount": w.get("cited_by_count"),
    }


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


def meta_openalex(paper_id: str) -> dict:
    """Metadata for screening when S2 throttles; citationCount is withheld (see module doc)."""
    nid = normalize_id(paper_id)
    if nid.startswith("ARXIV:"):
        key = f"doi:10.48550/arxiv.{nid.removeprefix('ARXIV:')}"
    elif nid.startswith("DOI:"):
        key = f"doi:{nid.removeprefix('DOI:')}"
    else:
        raise S2Unavailable(f"S2 throttled and OpenAlex cannot resolve {paper_id}")
    try:
        w = _openalex(f"/{urllib.parse.quote(key, safe=':/')}")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        raise S2Unavailable(f"S2 throttled and OpenAlex has no {paper_id}") from e
    out = {**_openalex_candidate(w), "citationCount": None}
    inv = w.get("abstract_inverted_index") or {}
    words = {i: word for word, idxs in inv.items() for i in idxs}
    out["abstract"] = " ".join(words[i] for i in sorted(words)) or None
    out["influentialCitationCount"] = None
    venues = {
        (loc.get("source") or {}).get("display_name")
        for loc in w.get("locations") or []
    }
    out["publicationVenue"] = "; ".join(sorted(v for v in venues if v)) or None
    out["authors"] = [
        (a.get("author") or {}).get("display_name") for a in w.get("authorships") or []
    ]
    out["url"] = w.get("id")
    out["source"] = "openalex"
    return out


def search_arxiv(query: str, since: str | None, limit: int) -> list[dict]:
    """Recent preprints S2 may not have indexed yet, newest first."""
    # arXiv ORs bare words; AND them and stay inside the ML/NLP categories.
    terms = " AND ".join(
        f'all:"{t}"' if " " in t else f"all:{t}" for t in shlex.split(query)
    )
    params = {
        "search_query": f"({terms}) AND (cat:cs.LG OR cat:cs.CL OR cat:cs.AI)",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": str(limit),
    }
    root = ET.fromstring(
        _get(f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}")
    )
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in root.findall("a:entry", ns):
        published = (entry.findtext("a:published", "", ns) or "")[:10]
        if since and published < since:
            continue
        m = _ARXIV.search(entry.findtext("a:id", "", ns) or "")
        out.append(
            {
                "paperId": None,
                "arxiv": m.group(1) if m else None,
                "doi": None,
                "title": " ".join((entry.findtext("a:title", "", ns) or "").split()),
                "year": int(published[:4]) if published else None,
                "publicationDate": published or None,
                "venue": "arXiv",
                "citationCount": None,
            }
        )
    return out


def meta(paper_id: str, today: date | None = None, *, s2_only: bool = False) -> dict:
    try:
        p = json.loads(
            _get(
                f"{S2}/paper/{urllib.parse.quote(normalize_id(paper_id), safe=':')}?fields={FIELDS}",
                s2=True,
            )
        )
    except S2Unavailable:
        if s2_only:
            raise
        out = meta_openalex(paper_id)
    else:
        out = _candidate(p)
        out["abstract"] = p.get("abstract")
        out["influentialCitationCount"] = p.get("influentialCitationCount")
        out["publicationVenue"] = (p.get("publicationVenue") or {}).get("name")
        out["authors"] = [a.get("name") for a in p.get("authors") or []]
        out["url"] = p.get("url")
        out["source"] = "s2"
    out["months"] = months_since(out["publicationDate"], out["year"], today)
    return out


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
    """Deterministic 0-3 Impact score; the anchors live in murano-scout/SKILL.md."""
    c = m.get("citationCount") or 0
    months = m.get("months") or months_since(m.get("publicationDate"), m.get("year"))
    rate = c / months
    top = is_top_venue(m.get("venue"), m.get("publicationVenue"))
    reasons = []
    if c >= 300 or (months < 18 and rate >= 10):
        score = 3
        reasons.append(f"{c} citations, {rate:.1f}/month")
    elif c >= 100 or (months < 18 and c >= 20 and rate >= 20 / 6) or top or viral:
        score = 2
        reasons += [
            r
            for r, hit in (
                (f"{c} citations", c >= 100),
                (
                    f"{rate:.1f} citations/month at {months:.0f} months",
                    months < 18 and c >= 20,
                ),
                ("top venue", top),
                ("viral signal", viral),
            )
            if hit
        ]
    elif c >= 20:
        score = 1
        reasons.append(f"{c} citations")
    else:
        score = 0
        reasons.append(f"{c} citations at {months:.0f} months")
    # Too new to judge is not the same as unimportant: the scout defers these
    # instead of rejecting them, and ledger.py due brings them back later.
    return {
        "impact": score,
        "reasons": reasons,
        "deferrable": score < 2 and months < 12,
    }


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
    s.add_argument("--min-citations", type=int, default=0)
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--source", choices=("openalex", "arxiv", "s2"), default="openalex")
    for name in ("meta", "impact", "code-links"):
        p = sub.add_parser(name)
        p.add_argument("id")
        if name == "impact":
            p.add_argument(
                "--viral", action="store_true", help="agent-verified viral signal"
            )
    a = ap.parse_args(argv)
    try:
        _run(a)
    except S2Unavailable as e:
        # The scout treats this as "not screened": no ledger row, retry next tick.
        print(json.dumps({"error": "s2_unavailable", "detail": str(e)}))
        return 2


def _run(a: argparse.Namespace) -> None:
    if a.cmd == "search":
        if a.source == "s2":
            rows = search_s2(a.query, a.since, a.min_citations, a.limit)
        elif a.min_citations:
            sys.exit(
                "--min-citations needs --source s2 (citation counts elsewhere are unreliable)"
            )
        elif a.source == "arxiv":
            rows = search_arxiv(a.query, a.since, a.limit)
        else:
            rows = search_openalex(a.query, a.since, a.limit)
        for row in rows:
            print(json.dumps(row))
    elif a.cmd == "meta":
        print(json.dumps(meta(a.id), indent=2))
    elif a.cmd == "impact":
        m = meta(a.id, s2_only=True)
        print(
            json.dumps(
                {"id": a.id, "title": m["title"], **impact_score(m, a.viral)}, indent=2
            )
        )
    else:
        print(json.dumps(code_links(a.id), indent=2))


if __name__ == "__main__":
    sys.exit(main())
