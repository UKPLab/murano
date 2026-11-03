"""Deterministic plumbing for the murano loop: preflight, the tick plan, end of tick.

    python loop.py preflight [--dry-run]     # set up, then print the plan (JSON)
    python loop.py plan                      # which tick steps apply, in order (JSON)
    python loop.py end --log body.md [--dry-run]

The model decides nothing here. Everything a tick does that follows from
labels, dates and limits in ``.claude/agent-loop.json`` lives in this file;
the skill keeps only the judgment calls.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CFG = ROOT / ".claude" / "agent-loop.json"
STATE = ROOT / ".agent-state"
NOTES = STATE / "notes.md"
LEDGER = STATE / "scouted.jsonl"
LEDGER_PY = ROOT / ".claude" / "skills" / "murano-scout" / "scripts" / "ledger.py"
LABELS = (
    "agent:todo",
    "agent:sourced",
    "agent:wip",
    "agent:needs-review",
    "agent:changes-requested",
    "agent:ready-to-merge",
    "agent:blocked",
    "agent:hold",
    "agent:log",
)
_ROUND = re.compile(r"<!--\s*agent-review-round:\s*(\d+)\s*-->")


def sh(*cmd: str, write: bool = False, dry: bool = False, check: bool = True) -> str:
    """Run ``cmd`` in the repo root; a write under ``dry`` is printed, not run."""
    if write and dry:
        print(f"DRY-RUN: {shlex.join(cmd)}", file=sys.stderr)
        return ""
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"{shlex.join(cmd)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def gh_json(*args: str) -> list[dict]:
    return json.loads(sh("gh", *args) or "[]")


def load_cfg() -> dict:
    return json.loads(CFG.read_text())


def read_notes(path: Path = NOTES) -> dict:
    """``key: value`` lines (``postponed`` is a JSON list) plus the free-text threads."""
    text = path.read_text() if path.exists() else ""
    head, _, threads = text.partition("## Open threads")
    notes: dict = {"threads": threads.strip("\n")}
    for key, value in re.findall(r"^(\w+): (.*)$", head, re.M):
        notes[key] = json.loads(value) if key == "postponed" else value.strip()
    notes.setdefault("postponed", [])
    return notes


def write_notes(notes: dict, path: Path = NOTES) -> None:
    fields = {k: v for k, v in notes.items() if k != "threads"}
    lines = ["# Agent loop notes", ""]
    for key in ("last_tick", "last_scout", "last_classics"):
        lines.append(f"{key}: {fields.pop(key, 'never')}")
    lines.append(f"postponed: {json.dumps(fields.pop('postponed', []))}")
    lines += [f"{k}: {v}" for k, v in fields.items()]
    lines += ["", "## Open threads", notes.get("threads", "").strip("\n"), ""]
    path.write_text("\n".join(lines))


def days_since(value: str | None, today: date) -> float | None:
    if not value or value == "never":
        return None
    return (today - date.fromisoformat(value)).days


def _labels(item: dict) -> set[str]:
    return {lab["name"] for lab in item.get("labels") or []}


def tick_plan(
    prs: list[dict], issues: list[dict], cfg: dict, notes: dict, today: date
) -> dict:
    """The tick's work queue, oldest first, from open PRs and issues (gh JSON)."""
    prs = sorted(
        (p for p in prs if "agent:hold" not in _labels(p)), key=lambda p: p["createdAt"]
    )
    issues = sorted(
        (i for i in issues if "agent:hold" not in _labels(i)),
        key=lambda i: i["createdAt"],
    )
    revise = next((p for p in prs if "agent:changes-requested" in _labels(p)), None)
    rounds = (
        int(m.group(1))
        if revise and (m := _ROUND.search(revise.get("body") or ""))
        else 0
    )
    open_agent_prs = sum(
        any(lab.startswith("agent:") for lab in _labels(p)) for p in prs
    )
    sourced = [
        i["number"]
        for i in issues
        if "agent:sourced" in _labels(i) and "has-pr" not in _labels(i)
    ]
    todo = [i["number"] for i in issues if "agent:todo" in _labels(i)]
    backlog = sum(bool({"agent:todo", "agent:sourced"} & _labels(i)) for i in issues)
    since_scout = days_since(notes.get("last_scout"), today)
    since_classics = days_since(notes.get("last_classics"), today)
    scout_due = (
        since_scout is None or since_scout >= cfg["scout_min_interval_days"]
    ) and backlog < cfg["backlog_cap"]
    return {
        "review": [p["number"] for p in prs if "agent:needs-review" in _labels(p)],
        "revise": revise
        and {
            "pr": revise["number"],
            "round": rounds,
            "block": rounds >= cfg["max_review_rounds"],
        },
        "implement": sourced[0]
        if sourced and open_agent_prs < cfg["max_open_agent_prs"]
        else None,
        "source": todo[: cfg["max_sourcing_per_tick"]],
        "scout": scout_due,
        "classics_due": since_classics is None
        or since_classics >= cfg["classics_sweep_every_days"],
        "backlog": backlog,
        "open_agent_prs": open_agent_prs,
    }


def plan(cfg: dict) -> dict:
    prs = gh_json(
        "pr",
        "list",
        "--state",
        "open",
        "--limit",
        "200",
        "--json",
        "number,labels,body,createdAt",
    )
    issues = gh_json(
        "issue",
        "list",
        "--state",
        "open",
        "--limit",
        "500",
        "--json",
        "number,labels,createdAt",
    )
    out = tick_plan(prs, issues, cfg, read_notes(), date.today())
    out["idle"] = not (
        out["review"]
        or out["revise"]
        or out["implement"]
        or out["source"]
        or out["scout"]
    )
    return out


def preflight(dry: bool) -> dict:
    cfg = load_cfg()
    base = cfg.get("base_branch", "main")
    sh("gh", "auth", "status")
    if sh("git", "status", "--porcelain"):
        sys.exit("The working tree is dirty: commit or stash before a tick.")
    # Push as gh's account: a stale credential helper may hold another login.
    helpers = sh(
        "git", "config", "--local", "--get-all", "credential.helper", check=False
    )
    if "gh auth git-credential" not in helpers:
        sh(
            "git",
            "config",
            "--local",
            "--replace-all",
            "credential.helper",
            "",
            write=True,
            dry=dry,
        )
        sh(
            "git",
            "config",
            "--local",
            "--add",
            "credential.helper",
            "!gh auth git-credential",
            write=True,
            dry=dry,
        )
    sh("git", "fetch", "-q", "origin")
    sh("git", "switch", "-q", base)
    sh("git", "pull", "-q", "--ff-only", "origin", base)
    sh("uv", "sync", "-q", "--frozen", "--all-extras", "--dev")

    if sh("git", "ls-remote", "--heads", "origin", "agent-state"):
        if not STATE.exists():
            sh("git", "worktree", "add", "-q", str(STATE), "agent-state")
        sh("git", "-C", str(STATE), "pull", "-q", "--ff-only")
    elif not STATE.exists():
        sh("git", "worktree", "add", "-q", "--orphan", "-b", "agent-state", str(STATE))
        write_notes({"postponed": [], "threads": ""})
        LEDGER.touch()
    sh(sys.executable, "-I", str(LEDGER_PY), "--ledger", str(LEDGER), "sync-issues")

    have = {
        lab["name"]
        for lab in gh_json("label", "list", "--search", "agent:", "--json", "name")
    }
    if set(LABELS) - have:
        sh("bash", ".github/scripts/bootstrap-labels.sh", write=True, dry=dry)

    logs = gh_json(
        "issue", "list", "--label", "agent:log", "--state", "open", "--json", "number"
    )
    if logs:
        log_issue = logs[0]["number"]
    else:
        body = (
            "> *Generated by the murano agent loop.*\n\nRun log for the murano agent "
            "loop (see `AGENTS.md` §13). Every tick posts one comment here."
        )
        url = sh(
            "gh",
            "issue",
            "create",
            "--title",
            "Agent loop log",
            "--label",
            "agent:log",
            "--body",
            body,
            write=True,
            dry=dry,
        )
        log_issue = int(url.rsplit("/", 1)[-1]) if url else None
        if url:
            sh("gh", "issue", "pin", url, write=True, dry=dry)

    # A tick holds agent:wip only while it runs, so any claim left now is stale.
    cleared = clear_wip(dry)
    return {"base": base, "log_issue": log_issue, "cleared_wip": cleared, **plan(cfg)}


def clear_wip(dry: bool) -> list[int]:
    cleared = []
    for kind in ("issue", "pr"):
        for item in gh_json(
            kind, "list", "--label", "agent:wip", "--state", "open", "--json", "number"
        ):
            sh(
                "gh",
                kind,
                "edit",
                str(item["number"]),
                "--remove-label",
                "agent:wip",
                write=True,
                dry=dry,
            )
            cleared.append(item["number"])
    return cleared


def end(log: Path, dry: bool) -> dict:
    """Release claims, stamp ``last_tick``, push the memory, post the log comment."""
    cleared = clear_wip(dry)
    notes = read_notes()
    notes["last_tick"] = date.today().isoformat()
    write_notes(notes)
    git = ("git", "-C", str(STATE))
    sh(*git, "add", "-A")
    sh(*git, "commit", "-qm", f"tick {date.today().isoformat()}", check=False)
    sh(*git, "push", "-q", "-u", "origin", "agent-state", write=True, dry=dry)
    logs = gh_json(
        "issue", "list", "--label", "agent:log", "--state", "open", "--json", "number"
    )
    url = (
        sh(
            "gh",
            "issue",
            "comment",
            str(logs[0]["number"]),
            "--body-file",
            str(log),
            write=True,
            dry=dry,
        )
        if logs
        else ""
    )
    return {"released_wip": cleared, "log_comment": url, "bad_bodies": bad_bodies()}


def bad_bodies() -> dict:
    """Open agent PRs whose body fails ``pr_text.py check body``, with the errors."""
    sys.path.insert(0, str(ROOT / ".claude/skills/murano-implement/scripts"))
    import pr_text

    prs = gh_json("pr", "list", "--state", "open", "--json", "number,body,labels")
    return {
        pr["number"]: errs
        for pr in prs
        if any(lab["name"].startswith("agent:") for lab in pr["labels"])
        and (errs := pr_text.check_body(pr["body"]))
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight").add_argument("--dry-run", action="store_true")
    sub.add_parser("plan")
    e = sub.add_parser("end")
    e.add_argument(
        "--log", type=Path, required=True, help="log comment body (markdown)"
    )
    e.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "preflight":
        out = preflight(a.dry_run)
    elif a.cmd == "plan":
        out = plan(load_cfg())
    else:
        out = end(a.log, a.dry_run)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
