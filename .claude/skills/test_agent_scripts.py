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
    str(HERE / "murano-loop" / "scripts"),
]

import ledger  # noqa: E402
import loop  # noqa: E402
import paper  # noqa: E402
import post  # noqa: E402
import repo_check  # noqa: E402
import scout  # noqa: E402

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
    # Undercounted citations: stars of the official code lift the score.
    adopted = {"citationCount": 65, "months": 37, "githubStars": 1056}
    assert paper.impact_score(adopted)["impact"] == 3
    assert paper.impact_score({**adopted, "githubStars": 214})["impact"] == 2
    assert paper.impact_score({**adopted, "githubStars": 35})["impact"] == 1


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


CFG = {
    "scout_min_interval_days": 6,
    "classics_sweep_every_days": 90,
    "backlog_cap": 5,
    "max_open_agent_prs": 2,
    "max_review_rounds": 3,
    "max_sourcing_per_tick": 2,
    "accept": {"min_total": 8, "min_impact": 2},
}


def _item(n, labels, created, body=""):
    return {
        "number": n,
        "labels": [{"name": lab} for lab in labels],
        "createdAt": created,
        "body": body,
    }


def test_tick_plan_orders_oldest_first_and_respects_limits():
    prs = [
        _item(10, ["agent:needs-review"], "2026-11-02"),
        _item(
            11,
            ["agent:changes-requested"],
            "2026-11-01",
            "<!-- agent-review-round: 3 -->",
        ),
        _item(12, ["agent:needs-review", "agent:hold"], "2026-10-01"),
    ]
    issues = [
        _item(20, ["agent:todo"], "2026-11-03"),
        _item(21, ["agent:todo"], "2026-11-01"),
        _item(22, ["agent:todo"], "2026-11-02"),
        _item(23, ["agent:sourced", "has-pr"], "2026-10-01"),
        _item(24, ["agent:sourced"], "2026-10-02"),
    ]
    notes = {"last_scout": "2026-11-01", "last_classics": "never"}
    plan = loop.tick_plan(prs, issues, CFG, notes, TODAY)
    assert plan["review"] == [10]  # held PR skipped
    assert plan["revise"] == {"pr": 11, "round": 3, "block": True}
    # Two agent PRs open (held one excluded) reaches max_open_agent_prs.
    assert plan["implement"] is None
    assert plan["source"] == [21, 22]
    assert (
        plan["backlog"] == 5 and plan["scout"] is False and plan["classics_due"] is True
    )
    fewer = loop.tick_plan(prs[:1], issues[:2], CFG, notes, TODAY)
    # Scouted exactly scout_min_interval_days ago with room in the backlog: due.
    assert fewer["implement"] is None and fewer["scout"] is True
    assert (
        loop.tick_plan([], issues[:2], CFG, {"last_scout": "2026-11-05"}, TODAY)[
            "scout"
        ]
        is False
    )


def test_notes_round_trip(tmp_path):
    path = tmp_path / "notes.md"
    notes = {
        "last_scout": "2026-11-03",
        "postponed": ["2610.1", "10.5281/x"],
        "threads": "- one\n- two",
    }
    loop.write_notes(notes, path)
    back = loop.read_notes(path)
    assert back["postponed"] == ["2610.1", "10.5281/x"]
    assert back["last_scout"] == "2026-11-03" and back["last_classics"] == "never"
    assert back["threads"].strip() == "- one\n- two"
    assert loop.days_since("never", TODAY) is None
    assert loop.days_since("2026-11-01", TODAY) == 6


def test_scout_parsing_helpers():
    assert scout.parse_spec("0-2, 7") == {0, 1, 2, 7}
    assert scout.parse_spec(None) == set()
    assert scout.pid({"arxiv": "2310.01405v2"}) == "2310.01405"
    assert scout.pid({"arxiv": "DOI 10.5281/Zenodo.1"}) == "10.5281/zenodo.1"
    assert scout._ref("2310.01405") == "2310.01405"
    assert scout._ref("10.5281/zenodo.1") == "DOI:10.5281/zenodo.1"
    ok = {g: ["pass", ""] for g in scout.GATES}
    assert scout.first_failed(ok) is None
    assert (
        scout.first_failed({**ok, "G3": ["fail", "gradients"], "G6": ["fail", ""]})
        == "G3"
    )
    assert scout.first_failed({**ok, "G4": "FAIL"}) == "G4"
    assert scout.triage_flag("Steering vectors in language models") == "KL"
    assert scout.triage_flag("Microneedle patch for wound healing").endswith("!")


def test_decide_actions_caps_issues_and_defers():
    def e(i, impact, total, deferrable=False, passed=True, err=None):
        return {
            "id": i,
            "pass": passed,
            "impact": impact,
            "total": total,
            "deferrable": deferrable,
            "impact_error": err,
            "gate": None if passed else "G3",
        }

    entries = [
        e("a", 3, 12),
        e("b", 2, 11),
        e("c", 3, 10),  # three accepts, two slots
        e("d", 0, 9, deferrable=True),
        e("f", 2, 7),
        e("g", None, 9, err="source_unavailable"),
        e("h", 0, 3, passed=False),
    ]
    acts = {x["id"]: act for x, act in scout.decide_actions(entries, 2, CFG)}
    assert acts == {
        "a": "file",
        "b": "file",
        "c": "defer_backlog",
        "d": "defer_new",
        "f": "reject_score",
        "g": "pending",
        "h": "reject_gate",
    }


GOOD_BODY = """> *Generated by the murano agent loop.*

Closes #1

Adds `X`, which finds the heads that carry a prediction.

### Open questions

- Keep the default threshold? Provisional: 0.04, because the paper uses it.

<details><summary>Details for the reviewer</summary>

### Differences from the reference code

- Clips at 0, so masked tokens get no edge.

</details>

<!-- agent-review-round: 0 -->
"""


def test_pr_body_lint_cuts_jargon_and_extra_sections():
    assert post.check("body", GOOD_BODY) == []
    moved_out = (
        GOOD_BODY.replace("<details><summary>Details for the reviewer</summary>\n", "")
        + "<details></details>"
    )
    assert any("move" in e for e in post.check("body", moved_out))
    long_q = GOOD_BODY.replace("because the paper uses it", "because " + "word " * 40)
    assert any("open question is" in e for e in post.check("body", long_q))
    jargon = GOOD_BODY.replace("which finds", "`a` `b` `c` `d` `e` `f` `g`")
    assert any("code spans" in e for e in post.check("body", jargon))
    no_choice = GOOD_BODY.replace("Provisional: 0.04", "Maybe 0.04")
    assert any("Provisional" in e for e in post.check("body", no_choice))


def test_reply_and_review_lint():
    d = post.DISCLAIMER
    assert (
        post.check(
            "reply",
            f"{d}\n\n1. Fixed in abc1234 and def5678: docs say two passes.\n2. Not changed: see test_x.\n",
        )
        == []
    )
    assert post.check("reply", f"{d}\n1. Fixed in abc1234: ok\nAlso a recap.\n")
    assert post.check("reply", f"{d}\n1. Fixed in abc1234: " + "w " * 25)
    review = f"{d}\n<!-- murano-review -->\n## Review, round 1\n\nVERDICT: CHANGES_REQUESTED\n\n1 problem.\n\n### Must fix\n1. `a.py:1`: wrong. Fix: right.\n"
    assert post.check("review", review) == []
    assert post.check("review", review.replace("wrong.", "wrong " * 50))


def test_dossier_issue_question_and_log_lint():
    d = post.DISCLAIMER
    dossier = f"""{d}
<!-- murano-dossier -->
## Dossier: LAT

Adds `SteeringVector(method="lat")`, adapted from MIT code.

### Open questions
None.

<details><summary>Implementation brief</summary>

### Oracles
1. A planted direction is recovered.

</details>
"""
    assert post.check("dossier", dossier) == []
    assert any(
        "inside <details>" in e
        for e in post.check("dossier", dossier.replace("1. A planted", "word " * 600))
    )
    assert any(
        "move" in e
        for e in post.check(
            "dossier",
            dossier.replace(
                "### Open questions", "### Algorithm\n\n### Open questions"
            ),
        )
    )
    issue = f"{d}\n\n### Paper\nT. A. arXiv, 2023.\n\n### Method\nIt reads.\n\n<details>\nscores\n</details>\n"
    assert post.check("issue", issue) == []
    assert post.check("issue", issue.replace("It reads.", "word " * 130))
    q = f"{d}\n<!-- murano-open-question -->\nSign rule? Provisional: reference rule, because the code does it."
    assert post.check("question", q) == []
    assert post.check("question", q.replace("Provisional:", "Maybe"))
    assert post.check("log", f"{d}\n- did one thing") == []
    assert post.check("log", "no disclaimer")
