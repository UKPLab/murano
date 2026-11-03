"""Scouting ledger: every paper the scout has ever screened, so none is screened twice.

The ledger is ``scouted.jsonl`` on the orphan ``agent-state`` branch, checked
out by the loop at ``.agent-state/``. One row per paper:

    {"key": "...", "aliases": ["arxiv:2211.00593", "doi:...", "s2:...", "title:..."],
     "title": "...", "status": "accepted" | "rejected" | "deferred",
     "issue": 123 | null, "gate": "G1".."G6" | "score" | "human" | null,
     "reason": "...", "date": "YYYY-MM-DD", "recheck_after": "YYYY-MM-DD" | null}

    python ledger.py filter < candidates.jsonl      # print only never-seen candidates
    python ledger.py add --arxiv 2211.00593 --title "..." --status rejected --gate G1 --reason "..."
    python ledger.py due                            # deferred rows whose recheck date passed
    python ledger.py sync-issues                    # import papers named in GitHub issues

``filter`` treats deferred rows as seen; ``due`` is the only way back into
screening. Pass ``--ledger PATH`` to use another file.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path

DEFAULT_LEDGER = Path(".agent-state/scouted.jsonl")
FUZZY_TITLE = 0.92

_ARXIV_REF = re.compile(r"arxiv(?:\.org/(?:abs|pdf|html)/|:\s*)(\d{4}\.\d{4,5})", re.I)
_DOI_REF = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>)\]]+)", re.I)


def norm_title(title: str | None) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).split())


def aliases(row: dict) -> set[str]:
    """Every normalized identifier a candidate or ledger row is known by."""
    out = set(row.get("aliases") or [])
    if row.get("arxiv"):
        out.add(
            "arxiv:"
            + re.sub(r"v\d+$", "", str(row["arxiv"]).lower().removeprefix("arxiv:"))
        )
    if row.get("doi"):
        out.add("doi:" + str(row["doi"]).lower().removeprefix("doi:"))
    if row.get("paperId"):
        out.add("s2:" + row["paperId"])
    if row.get("url"):
        out.add(
            "url:"
            + str(row["url"])
            .lower()
            .rstrip("/")
            .removeprefix("https://")
            .removeprefix("http://")
        )
    if row.get("title"):
        out.add("title:" + norm_title(row["title"]))
    return out


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def find(rows: list[dict], candidate: dict) -> dict | None:
    """Return the ledger row matching ``candidate`` by any alias or a near-identical title."""
    cand = aliases(candidate)
    for row in rows:
        if cand & set(row["aliases"]):
            return row
    title = norm_title(candidate.get("title"))
    # ponytail: linear fuzzy scan, fine for thousands of rows; index by title tokens if it grows past that
    if title:
        for row in rows:
            if (
                SequenceMatcher(None, title, norm_title(row.get("title"))).ratio()
                >= FUZZY_TITLE
            ):
                return row
    return None


def unseen(rows: list[dict], candidates: list[dict]) -> list[dict]:
    out, batch = [], []
    for cand in candidates:
        # Also dedup within the batch: two queries often return the same paper.
        if find(rows, cand) is None and find(batch, cand) is None:
            out.append(cand)
            batch.append({"aliases": sorted(aliases(cand)), "title": cand.get("title")})
    return out


def upsert(
    rows: list[dict],
    candidate: dict,
    *,
    status: str,
    today: date,
    issue: int | None = None,
    gate: str | None = None,
    reason: str = "",
    recheck_days: int | None = None,
) -> dict:
    row = find(rows, candidate)
    if row is None:
        row = {"aliases": []}
        rows.append(row)
    row["aliases"] = sorted(set(row["aliases"]) | aliases(candidate))
    row["key"] = next(
        (a for a in row["aliases"] if a.startswith("s2:")), row["aliases"][0]
    )
    row["title"] = candidate.get("title") or row.get("title")
    row.update(
        status=status,
        issue=issue if issue is not None else row.get("issue"),
        gate=gate,
        reason=reason,
        date=today.isoformat(),
    )
    row["recheck_after"] = (
        (today + timedelta(days=recheck_days)).isoformat()
        if status == "deferred" and recheck_days
        else None
    )
    return row


def due(rows: list[dict], today: date) -> list[dict]:
    return [
        r
        for r in rows
        if r["status"] == "deferred"
        and r.get("recheck_after")
        and r["recheck_after"] <= today.isoformat()
    ]


def papers_in_issue(issue: dict) -> list[dict]:
    """Candidates referenced by an issue: one per arXiv ID or DOI in its title and body."""
    text = f"{issue.get('title', '')}\n{issue.get('body') or ''}"
    found = [{"arxiv": a} for a in dict.fromkeys(_ARXIV_REF.findall(text))]
    found += [
        {"doi": d.rstrip(".,;")}
        for d in dict.fromkeys(_DOI_REF.findall(text))
        if "arxiv" not in d.lower()
    ]
    return found


def sync_issues(rows: list[dict], issues: list[dict], today: date) -> int:
    """Mark every paper named in an issue as seen. Returns the number of rows changed.

    Open or completed issues count as accepted. Issues closed as not planned
    count as a human rejection and override an earlier agent decision.
    """
    changed = 0
    for issue in issues:
        rejected = (
            issue.get("state", "").upper() == "CLOSED"
            and (issue.get("stateReason") or "").upper() == "NOT_PLANNED"
        )
        for cand in papers_in_issue(issue):
            row = find(rows, cand)
            if row is not None and not (rejected and row["status"] != "rejected"):
                continue
            cand["title"] = row.get("title") if row else None
            upsert(
                rows,
                cand,
                status="rejected" if rejected else "accepted",
                today=today,
                issue=issue["number"],
                gate="human" if rejected else None,
                reason=f"issue #{issue['number']} "
                + ("closed as not planned" if rejected else "exists"),
            )
            changed += 1
    return changed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("filter")
    sub.add_parser("due")
    s = sub.add_parser("sync-issues")
    s.add_argument(
        "--repo", default=None, help="owner/repo; defaults to the current gh repo"
    )
    add = sub.add_parser("add")
    for flag in ("--arxiv", "--doi", "--s2", "--url", "--title", "--gate", "--reason"):
        add.add_argument(flag, default=None)
    add.add_argument(
        "--status", required=True, choices=("accepted", "rejected", "deferred")
    )
    add.add_argument("--issue", type=int)
    add.add_argument("--recheck-days", type=int, default=90)
    a = ap.parse_args(argv)

    rows, today = load(a.ledger), date.today()
    if a.cmd == "filter":
        for cand in unseen(
            rows, [json.loads(line) for line in sys.stdin if line.strip()]
        ):
            print(json.dumps(cand))
    elif a.cmd == "due":
        for row in due(rows, today):
            print(json.dumps(row))
    elif a.cmd == "add":
        cand = {
            "arxiv": a.arxiv,
            "doi": a.doi,
            "paperId": a.s2,
            "url": a.url,
            "title": a.title,
        }
        if not aliases(cand):
            sys.exit("add needs at least one of --arxiv/--doi/--s2/--url/--title")
        row = upsert(
            rows,
            cand,
            status=a.status,
            today=today,
            issue=a.issue,
            gate=a.gate,
            reason=a.reason or "",
            recheck_days=a.recheck_days,
        )
        save(a.ledger, rows)
        print(json.dumps(row))
    else:
        cmd = [
            "gh",
            "issue",
            "list",
            "--state",
            "all",
            "--limit",
            "5000",
            "--json",
            "number,title,body,state,stateReason",
        ]
        if a.repo:
            cmd += ["--repo", a.repo]
        issues = json.loads(
            subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        )
        n = sync_issues(rows, issues, today)
        save(a.ledger, rows)
        print(
            json.dumps(
                {"issues": len(issues), "rows_changed": n, "ledger_size": len(rows)}
            )
        )


if __name__ == "__main__":
    sys.exit(main())
