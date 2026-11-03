"""Offline checks for the agent-loop scripts (no network, no gh).

uv run pytest .claude/skills/test_agent_scripts.py
"""

import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [
    str(HERE / "murano-scout" / "scripts"),
    str(HERE / "murano-source" / "scripts"),
]

import ledger  # noqa: E402
import paper  # noqa: E402
import repo_check  # noqa: E402

TODAY = date(2026, 11, 7)


def test_normalize_id():
    assert (
        paper.normalize_id("https://arxiv.org/abs/2211.00593v3") == "ARXIV:2211.00593"
    )
    assert paper.normalize_id("arXiv:2211.00593") == "ARXIV:2211.00593"
    assert paper.normalize_id("2211.00593v2") == "ARXIV:2211.00593"
    assert (
        paper.normalize_id("https://doi.org/10.18653/V1/N18-3011")
        == "DOI:10.18653/v1/n18-3011"
    )
    assert paper.normalize_id("A" * 40) == "a" * 40


def test_impact_score():
    old_classic = {"citationCount": 400, "publicationDate": "2022-11-01"}
    assert paper.impact_score({**old_classic, "months": 48})["impact"] == 3
    fast_riser = {"citationCount": 40, "months": 6}
    assert paper.impact_score(fast_riser)["impact"] == 2
    top_venue = {"citationCount": 5, "months": 10, "venue": "ICLR"}
    assert paper.impact_score(top_venue)["impact"] == 2
    fresh = {"citationCount": 3, "months": 2}
    assert paper.impact_score(fresh) == {
        **paper.impact_score(fresh),
        "impact": 0,
        "deferrable": True,
    }
    assert paper.impact_score(fresh, viral=True)["impact"] == 2
    stale = {"citationCount": 3, "months": 30}
    assert paper.impact_score(stale)["deferrable"] is False


def test_top_venue_abbrev_is_whole_word():
    assert paper.is_top_venue(
        "Annual Meeting of the Association for Computational Linguistics"
    )
    assert paper.is_top_venue("ACL 2025")
    assert not paper.is_top_venue("EACL")


def test_openalex_candidate_finds_arxiv_id():
    by_doi = {"doi": "https://doi.org/10.48550/arXiv.2610.32866", "title": "T"}
    assert paper._openalex_candidate(by_doi)["arxiv"] == "2610.32866"
    venue_doi = {
        "doi": "https://doi.org/10.18653/v1/2025.acl-long.1",
        "locations": [
            {"landing_page_url": "https://aclanthology.org/2025.acl-long.1"},
            {
                "landing_page_url": "https://arxiv.org/abs/2502.01234v2",
                "source": {"display_name": "arXiv"},
            },
        ],
    }
    cand = paper._openalex_candidate(venue_doi)
    assert (cand["arxiv"], cand["doi"], cand["venue"]) == (
        "2502.01234",
        "10.18653/v1/2025.acl-long.1",
        "arXiv",
    )
    assert paper._openalex_candidate({"doi": None})["arxiv"] is None


def test_ledger_dedups_aliases_versions_and_titles():
    rows = []
    ledger.upsert(
        rows,
        {"arxiv": "2211.00593v2", "title": "Interpretability in the Wild"},
        status="rejected",
        gate="G1",
        today=TODAY,
    )
    candidates = [
        {"arxiv": "2211.00593", "title": "anything"},  # same arXiv ID, version stripped
        {
            "paperId": "x" * 40,
            "title": "Interpretability in the wild!",
        },  # same title, new ID
        {
            "arxiv": "2501.00001",
            "title": "Interpretability in the Wild: a circuit",
        },  # fuzzy, not close enough
        {"arxiv": "2501.00002", "title": "A New Method"},
        {"doi": "10.1/X", "title": "A new method"},  # in-batch duplicate
    ]
    assert [c.get("arxiv") for c in ledger.unseen(rows, candidates)] == [
        "2501.00001",
        "2501.00002",
    ]


def test_ledger_deferred_only_returns_when_due():
    rows = []
    ledger.upsert(
        rows,
        {"arxiv": "2510.00001", "title": "Fresh"},
        status="deferred",
        today=TODAY,
        recheck_days=90,
    )
    assert ledger.unseen(rows, [{"arxiv": "2510.00001"}]) == []
    assert ledger.due(rows, TODAY) == []
    assert len(ledger.due(rows, date(2027, 2, 6))) == 1


def test_sync_issues_imports_and_human_rejection_wins():
    rows = []
    ledger.upsert(
        rows,
        {"arxiv": "2310.01405", "title": "RepE"},
        status="accepted",
        issue=7,
        today=TODAY,
    )
    issues = [
        {
            "number": 7,
            "title": "RepE",
            "body": "arXiv:2310.01405",
            "state": "CLOSED",
            "stateReason": "NOT_PLANNED",
        },
        {
            "number": 9,
            "title": "x",
            "body": "see https://arxiv.org/abs/2211.00593v1 and 10.1000/xyz.",
            "state": "OPEN",
        },
        {"number": 10, "title": "no paper here 1234.5678", "body": "", "state": "OPEN"},
    ]
    assert ledger.sync_issues(rows, issues, TODAY) == 3
    assert ledger.find(rows, {"arxiv": "2310.01405"})["status"] == "rejected"
    assert ledger.find(rows, {"doi": "10.1000/xyz"})["issue"] == 9
    assert ledger.sync_issues(rows, issues, TODAY) == 0  # idempotent


def test_license_and_repo_parsing():
    assert repo_check.license_verdict("MIT") == "adapt-ok"
    assert repo_check.license_verdict("GPL-3.0") == "oracle-only"
    assert repo_check.license_verdict(None) == "oracle-only"
    assert repo_check.license_verdict("NOASSERTION") == "oracle-only"
    assert repo_check.parse_repo(
        "https://github.com/saprmarks/geometry-of-truth.git"
    ) == ("saprmarks", "geometry-of-truth")
    assert repo_check.parse_repo("https://github.com/a/b/tree/main/src") == ("a", "b")
    assert repo_check.owner_matches(
        "saprmarks", "Samuel Marks", ["Samuel Marks", "Max Tegmark"]
    )
    assert not repo_check.owner_matches("randomdev", None, ["Samuel Marks"])
