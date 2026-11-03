"""The murano scout's deterministic steps, in the order SKILL.md runs them.

    python scout.py collect [--extra posts.jsonl]   # search + dedup -> run/unseen.json
    python scout.py triage                          # numbered titles, likeliest on-topic first
    python scout.py triage --screen 0-9,14 --postpone 20-40 [--used N]
    python scout.py batches                         # one screener brief per 5 papers
    python scout.py record                          # screener output -> pass / first failed gate
    python scout.py g5-search "tuned lens" "logit lens"
    python scout.py g5-fail 2309.16042 "composes from Patch + Sweep"
    python scout.py score [--viral ID ...]          # one impact batch for every gate-passer
    python scout.py decide                          # accept / defer / reject, issue slots, ledger
    python scout.py draft-issue 2310.01405 > body.md
    python scout.py file-issue 2310.01405 body.md --title "..." [--dry-run]
    python scout.py finish                          # notes + the counts block for the log

Run state lives in ``.agent-state/run/`` (ignored on the agent-state branch).
Papers that passed screening but could not be scored persist in
``.agent-state/pending_impact.json`` and are scored by the next ``score``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "murano-loop" / "scripts")]

import ledger  # noqa: E402
import paper  # noqa: E402
from loop import (  # noqa: E402
    ROOT,
    STATE,
    days_since,
    gh_json,
    load_cfg,
    read_notes,
    sh,
    write_notes,
)

RUN = STATE / "run"
PENDING = STATE / "pending_impact.json"
LEDGER = STATE / "scouted.jsonl"
TEMPLATE = HERE.parent / "screener_prompt.md"
GATES = ("G1", "G2", "G3", "G4", "G6")
BATCH = 5

# Title triage pre-sort only: the scout still reads every title it rejects.
_KW = re.compile(
    r"interpretab|circuit|patch|steer|prob(e|ing)|sparse autoencoder|\bsaes?\b|"
    r"transcoder|lens|attribution|mechanis|feature|represent|activation|attention head|"
    r"neuron|causal|intervention|mediation|subspace|direction|residual|superposition|concept",
    re.I,
)
_LM = re.compile(r"language model|\bllms?\b|transformer|\bgpt|\blms?\b|token|nlp", re.I)
_OFF = re.compile(
    r"clinical|patient|medical|cancer|tumou?r|protein|\bgenes?\b|\bcells?\b|\bmri\b|"
    r"histolog|retina|seismic|optical|laser|microneedle|hydrogel|wound|traffic|weather|"
    r"satellite|robot|driving|vision transformer|\bvit\b|image|video|audio|speech|music|"
    r"\beeg\b|molecul|chemi|diffusion model",
    re.I,
)


def _load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _save(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1) + "\n")


def _state() -> dict:
    return _load(RUN / "state.json", {"counts": {}, "filed": []})


def _count(state: dict, **kw: int) -> None:
    for k, v in kw.items():
        state["counts"][k] = state["counts"].get(k, 0) + v


def pid(entry: dict) -> str:
    """Stable id: the arXiv id when there is one, else the lowercase DOI."""
    raw = str(entry.get("arxiv") or entry.get("doi") or entry.get("id") or "")
    raw = re.sub(r"^(doi|arxiv)[: ]+", "", raw.strip(), flags=re.I)
    m = re.fullmatch(r"(\d{4}\.\d{4,5})(v\d+)?", raw)
    return m.group(1) if m else raw.lower()


def _cand(paper_id: str, title: str | None = None) -> dict:
    is_doi = paper_id.startswith("10.")
    return {
        "arxiv": None if is_doi else paper_id,
        "doi": paper_id if is_doi else None,
        "title": title,
    }


def _ref(paper_id: str) -> str:
    """The id form paper.py takes: a bare arXiv id, else ``DOI:<doi>``."""
    return paper_id if re.fullmatch(r"\d{4}\.\d{4,5}", paper_id) else f"DOI:{paper_id}"


def _from_row(row: dict) -> dict:
    al = {a.split(":", 1)[0]: a.split(":", 1)[1] for a in row["aliases"] if ":" in a}
    return {"arxiv": al.get("arxiv"), "doi": al.get("doi"), "title": row.get("title")}


def parse_spec(spec: str | None) -> set[int]:
    """``"0-3,7"`` -> {0, 1, 2, 3, 7}."""
    out: set[int] = set()
    for part in filter(None, (spec or "").replace(" ", "").split(",")):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def triage_flag(title: str) -> str:
    kw, lm, off = _KW.search(title), _LM.search(title), _OFF.search(title)
    flag = "KL" if kw and lm else "K" if kw else "L" if lm else "-"
    return flag + ("!" if off else "")


def _fill_titles(cands: list[dict]) -> None:
    need = [c for c in cands if not c.get("title")]
    arx = [c for c in need if c.get("arxiv")]
    for i in range(0, len(arx), 50):
        try:
            for c, m in zip(
                arx[i : i + 50], paper.meta([c["arxiv"] for c in arx[i : i + 50]])
            ):
                c.update(title=m["title"], publicationDate=m["publicationDate"])
        except paper.SourceUnavailable:
            pass
        time.sleep(3)
    for c in need:
        if c.get("doi") and not c.get("title"):
            try:
                c["title"] = paper.meta([f"DOI:{c['doi']}"])[0]["title"]
            except (paper.SourceUnavailable, urllib.error.HTTPError):
                pass


def collect(extra: Path | None) -> dict:
    cfg, notes, today = load_cfg(), read_notes(), date.today()
    RUN.mkdir(parents=True, exist_ok=True)
    for f in RUN.iterdir():
        f.unlink()
    ignore = STATE / ".gitignore"
    if "run/" not in (ignore.read_text() if ignore.exists() else ""):
        ignore.write_text("run/\n")
    since = notes.get("last_scout")
    if days_since(since, today) is None:
        since = (today - timedelta(days=90)).isoformat()
    gap = days_since(notes.get("last_classics"), today)
    classics = gap is None or gap >= cfg["classics_sweep_every_days"]
    cands: list[dict] = []
    failed: list[str] = []
    for q in cfg["queries"]:
        for window, limit in [(since, 50)] + ([(None, 100)] if classics else []):
            try:
                cands += paper.search_arxiv(q, window, limit)
            except (paper.SourceUnavailable, urllib.error.URLError) as e:
                failed.append(
                    f"{q} [{'classics' if window is None else 'window'}]: {e}"
                )
            time.sleep(3)  # arXiv API etiquette
    found = len(cands)
    cands += [_cand(p) for p in notes["postponed"]]
    if extra:
        cands += [
            json.loads(line) for line in extra.read_text().splitlines() if line.strip()
        ]
    rows = ledger.load(LEDGER)
    pending = {pid(e) for e in _load(PENDING, [])}
    new = [c for c in ledger.unseen(rows, cands) if pid(c) not in pending]
    new += [_from_row(r) for r in ledger.due(rows, today)]
    _fill_titles(new)
    _save(RUN / "unseen.json", new)
    state = {
        "since": since,
        "classics": classics,
        "counts": {},
        "filed": [],
        "search_failures": failed,
    }
    _count(state, identified=found, unseen=len(new))
    _save(RUN / "state.json", state)
    return {
        "since": since,
        "classics": classics,
        "identified": found,
        "unseen": len(new),
        "pending_impact": len(pending),
        "search_failures": failed,
    }


def triage_list() -> str:
    unseen = _load(RUN / "unseen.json", [])
    rank = {"KL": 0, "K": 1, "L": 2, "-": 3}
    order = sorted(
        range(len(unseen)),
        key=lambda i: (
            "!" in triage_flag(unseen[i].get("title") or ""),
            rank[triage_flag(unseen[i].get("title") or "").rstrip("!")],
        ),
    )
    lines = [
        f"{i}\t{triage_flag(unseen[i].get('title') or '')}\t"
        f"{(unseen[i].get('publicationDate') or '')[:7]}\t{unseen[i].get('title')}"
        for i in order
    ]
    head = (
        "# idx  flag (K interp keyword, L LM term, ! off-topic term)  date  title\n"
        "# Likeliest on-topic first. Every index not passed to --screen or --postpone"
        " is rejected (G2, title triage).\n"
    )
    return head + "\n".join(lines)


def triage_apply(screen: set[int], postpone: set[int], used: int) -> dict:
    cfg, unseen = load_cfg(), _load(RUN / "unseen.json", [])
    if screen & postpone:
        sys.exit(
            f"indices in both --screen and --postpone: {sorted(screen & postpone)}"
        )
    if bad := (screen | postpone) - set(range(len(unseen))):
        sys.exit(f"unknown indices: {sorted(bad)}")
    budget = (cfg["max_subagents_per_tick"] - used) * BATCH
    if len(screen) > budget:
        sys.exit(
            f"{len(screen)} papers to screen, budget allows {budget}; postpone the rest"
        )
    rows, today = ledger.load(LEDGER), date.today()
    rejected = 0
    for i, c in enumerate(unseen):
        if i not in screen | postpone:
            ledger.upsert(
                rows,
                c,
                status="rejected",
                today=today,
                gate="G2",
                reason="title triage: not about language-model internals",
            )
            rejected += 1
    ledger.save(LEDGER, rows)
    notes = read_notes()
    notes["postponed"] = [pid(unseen[i]) for i in sorted(postpone) if pid(unseen[i])]
    write_notes(notes)
    _save(RUN / "screen.json", [unseen[i] for i in sorted(screen)])
    state = _state()
    _count(
        state, title_rejected=rejected, postponed=len(postpone), screened=len(screen)
    )
    _save(RUN / "state.json", state)
    return {
        "rejected_G2": rejected,
        "postponed": len(postpone),
        "to_screen": len(screen),
        "batches": math.ceil(len(screen) / BATCH),
    }


def batches() -> list[str]:
    papers = _load(RUN / "screen.json", [])
    template = TEMPLATE.read_text()
    paths = []
    for b in range(0, len(papers), BATCH):
        n = b // BATCH
        lines = [
            f"- arXiv {p['arxiv']}: {p.get('title')}"
            if p.get("arxiv")
            else f"- DOI {p['doi']}: {p.get('title')} (read it with `meta DOI:{p['doi']}` or https://doi.org/{p['doi']})"
            for p in papers[b : b + BATCH]
        ]
        out = RUN / f"out_{n}.json"
        path = RUN / f"batch_{n}.md"
        path.write_text(template.format(root=ROOT, papers="\n".join(lines), out=out))
        paths.append(str(path))
    return paths


def first_failed(entry: dict) -> str | None:
    for gate in GATES:
        v = entry.get(gate)
        if str(v[0] if isinstance(v, list) else v).strip().lower() != "pass":
            return gate
    return None


def record(files: list[Path]) -> dict:
    screened = _load(RUN / "screened.json", {})
    titles = {pid(p): p.get("title") for p in _load(RUN / "screen.json", [])}
    files = files or sorted(RUN.glob("out_*.json"))
    for f in files:
        for e in json.loads(f.read_text()):
            k = pid(e)
            gate = first_failed(e)
            e.update(
                id=k,
                gate=gate,
                title=titles.get(k) or e.get("title"),
                **{"pass": gate is None},
            )
            screened[k] = e
    _save(RUN / "screened.json", screened)
    passing = [k for k, e in screened.items() if e["pass"]]
    return {
        "screened": len(screened),
        "passing": passing,
        "failed": dict(Counter(e["gate"] for e in screened.values() if not e["pass"])),
    }


def g5_search(terms: list[str]) -> dict:
    out = {}
    for t in terms:
        issues = gh_json(
            "issue", "list", "--state", "all", "--search", t, "--json", "number,title"
        )
        prs = gh_json(
            "pr", "list", "--state", "all", "--search", t, "--json", "number,title"
        )
        code = sh(
            "grep",
            "-rli",
            t,
            "src/murano",
            "README.md",
            "docs/src/content",
            check=False,
        )
        out[t] = {"issues": issues, "prs": prs, "files": code.splitlines()}
    return out


def g5_fail(paper_id: str, reason: str) -> dict:
    screened = _load(RUN / "screened.json", {})
    e = screened[pid({"id": paper_id})]
    e.update(gate="G5", g5=reason, **{"pass": False})
    _save(RUN / "screened.json", screened)
    return e


def score(viral: list[str]) -> dict:
    screened = _load(RUN / "screened.json", {})
    for e in _load(PENDING, []):
        screened.setdefault(e["id"], e)
    todo = [e for e in screened.values() if e["pass"] and e.get("impact") is None]
    if not todo:
        return {"scored": 0}
    if not os.environ.get("OPENALEX_API_KEY"):
        print(
            "note: OPENALEX_API_KEY unset; keyless budget is ~100 searches/day per IP "
            "(see SKILL.md before asking the user for a key)",
            file=sys.stderr,
        )
    repos = {
        e["id"]: e["official_repo"]
        for e in todo
        if re.fullmatch(r"[\w.-]+/[\w.-]+", str(e.get("official_repo") or ""))
    }
    ids = [_ref(e["id"]) for e in todo]
    try:
        rows = paper.impact(ids, repos, {pid({"id": v}) for v in viral})
    except paper.SourceUnavailable as err:
        rows = [
            {"id": i, "error": "source_unavailable", "detail": str(err)} for i in ids
        ]
    for e, r in zip(todo, rows):
        e.update(
            impact=r.get("impact"),
            impact_reasons=r.get("reasons"),
            deferrable=r.get("deferrable"),
            impact_error=r.get("error"),
            venues=r.get("venues"),
            repo=r.get("repo"),
            stars=r.get("stars"),
        )
        e["total"] = (e["impact"] or 0) + sum(
            int(e.get(k) or 0) for k in ("generality", "fit", "cost")
        )
    _save(RUN / "screened.json", screened)
    return {
        e["id"]: e.get("impact_error") or f"Impact {e['impact']}, total {e['total']}"
        for e in todo
    }


def decide_actions(
    entries: list[dict], slots: int, cfg: dict
) -> list[tuple[dict, str]]:
    """Pure decision rules; ``slots`` is how many issues may still be filed."""
    acts = [(e, "reject_gate") for e in entries if not e["pass"]]
    for e in sorted(
        (e for e in entries if e["pass"]), key=lambda e: -(e.get("total") or 0)
    ):
        if e.get("impact_error") or e.get("impact") is None:
            acts.append((e, "pending"))
        elif (
            e["impact"] >= cfg["accept"]["min_impact"]
            and e["total"] >= cfg["accept"]["min_total"]
        ):
            acts.append((e, "file" if slots > 0 else "defer_backlog"))
            slots -= 1
        elif e.get("deferrable"):
            acts.append((e, "defer_new"))
        else:
            acts.append((e, "reject_score"))
    return acts


def decide() -> dict:
    cfg, state = load_cfg(), _state()
    screened = _load(RUN / "screened.json", {})
    open_issues = gh_json(
        "issue", "list", "--state", "open", "--limit", "500", "--json", "labels"
    )
    backlog = sum(
        bool({"agent:todo", "agent:sourced"} & {lab["name"] for lab in i["labels"]})
        for i in open_issues
    )
    slots = max(
        0,
        min(
            cfg["max_new_issues_per_scout"] - len(state["filed"]),
            cfg["backlog_cap"] - backlog,
        ),
    )
    live = [e for e in screened.values() if e.get("decided") in (None, "pending")]
    rows, today = ledger.load(LEDGER), date.today()
    pending, summary = [], Counter()
    for e, act in decide_actions(live, slots, cfg):
        cand = _cand(e["id"], e.get("title"))
        why = f"total {e.get('total')}/12, Impact {e.get('impact')} ({'; '.join(e.get('impact_reasons') or [])})"
        if act == "reject_gate":
            ev = e.get(e["gate"]) if e["gate"] != "G5" else ["fail", e.get("g5")]
            text = ev[1] if isinstance(ev, list) and len(ev) > 1 else str(ev)
            prefix = (
                "backend extension: "
                if e["gate"] == "G3" and "backend" not in text
                else ""
            )
            ledger.upsert(
                rows,
                cand,
                status="rejected",
                today=today,
                gate=e["gate"],
                reason=prefix + text,
            )
        elif act == "defer_backlog":
            ledger.upsert(
                rows,
                cand,
                status="deferred",
                today=today,
                gate="score",
                reason="backlog full; " + why,
                recheck_days=7,
            )
        elif act == "defer_new":
            ledger.upsert(
                rows,
                cand,
                status="deferred",
                today=today,
                gate="score",
                reason="Impact too low for its age; " + why,
                recheck_days=cfg["defer_recheck_days"],
            )
        elif act == "reject_score":
            ledger.upsert(
                rows, cand, status="rejected", today=today, gate="score", reason=why
            )
        elif act == "pending":
            pending.append(e)
        e["decided"] = act
        summary[act if act != "reject_gate" else f"reject_{e['gate']}"] += 1
    ledger.save(LEDGER, rows)
    _save(PENDING, pending)
    _save(RUN / "screened.json", screened)
    for k, v in summary.items():
        _count(state, **{k: v})
    _save(RUN / "state.json", state)
    to_file = [
        {"id": e["id"], "title": e.get("title"), "total": e["total"]}
        for e in screened.values()
        if e.get("decided") == "file"
    ]
    return {
        "slots": slots,
        "backlog": backlog,
        "decisions": dict(summary),
        "to_file": to_file,
    }


def _authors(names: list[str]) -> str:
    return (
        ", ".join(names)
        if len(names) <= 4
        else f"{', '.join(names[:4])} et al. ({len(names)} authors)"
    )


def draft_issue(paper_id: str) -> str:
    e = _load(RUN / "screened.json", {})[pid({"id": paper_id})]
    m = paper.meta([_ref(e["id"])])[0]
    venues = [v for v in (e.get("venues") or "").split("; ") if paper.is_top_venue(v)]
    link = (
        f"https://arxiv.org/abs/{m['arxiv']}"
        if m.get("arxiv")
        else f"https://doi.org/{m['doi']}"
    )
    surname = (m["authors"] or ["?"])[0].split()[-1]
    links = [
        f"Official code: https://github.com/{e['repo']} ({e.get('stars')} stars)"
        if e.get("repo")
        else None
    ]
    links += [
        u
        for u in e.get("code_links") or []
        if not e.get("repo") or e["repo"].lower() not in u.lower()
    ]
    gates = (
        " · ".join(
            f"{g} ✓ {(e.get(g) or ['', ''])[1] if isinstance(e.get(g), list) else ''}".rstrip()
            for g in GATES[:4]
        )
        + " · G5 ✓ no Murano step, issue or PR covers it · G6 ✓ "
        + ((e.get("G6") or ["", ""])[1] if isinstance(e.get("G6"), list) else "")
    )
    print(
        f"suggested title: [METHOD] {m['title']} ({surname} et al., {m['year']})",
        file=sys.stderr,
    )
    return f"""> *Generated by the murano agent loop.*

### Paper
{m["title"]}. {_authors(m["authors"])}. {venues[0] if venues else "arXiv"}, {m["year"]}. {link}

### Links
{chr(10).join(f"- {x}" for x in links if x) or "No code found."}

### Method
{e.get("summary", "")}

### Why it matters
Impact {e["impact"]}/3 ({"; ".join(e.get("impact_reasons") or [])}) · Generality {e.get("generality")} · Fit {e.get("fit")} · Cost {e.get("cost")} · total {e["total"]}/12
Gates: {gates}

### Murano mapping
<!-- EDIT: existing steps it reuses; the new step(s) and what they read and write. Screener Fit={e.get("fit")}. -->
"""


def file_issue(paper_id: str, body: Path, title: str, dry: bool) -> dict:
    if "<!-- EDIT" in body.read_text():
        sys.exit("The body still has an <!-- EDIT --> placeholder.")
    url = sh(
        "gh",
        "issue",
        "create",
        "--title",
        title,
        "--label",
        "enhancement",
        "--label",
        "agent:todo",
        "--body-file",
        str(body),
        write=True,
        dry=dry,
    )
    number = int(url.rsplit("/", 1)[-1]) if url else None
    screened, state = _load(RUN / "screened.json", {}), _state()
    e = screened[pid({"id": paper_id})]
    rows = ledger.load(LEDGER)
    ledger.upsert(
        rows,
        _cand(e["id"], e.get("title")),
        status="accepted",
        today=date.today(),
        issue=number,
        reason=f"total {e['total']}/12, Impact {e['impact']}",
    )
    ledger.save(LEDGER, rows)
    e["decided"] = "filed"
    state["filed"].append({"id": e["id"], "issue": number, "url": url})
    _save(RUN / "screened.json", screened)
    _save(RUN / "state.json", state)
    return {"issue": number, "url": url}


def finish() -> str:
    state, notes, today = _state(), read_notes(), date.today().isoformat()
    notes["last_scout"] = today
    if state.get("classics"):
        notes["last_classics"] = today
    write_notes(notes)
    c = state["counts"]
    rejected = {
        k.removeprefix("reject_"): v for k, v in c.items() if k.startswith("reject_")
    }
    lines = [
        f"- window since {state.get('since')}{', plus the classics sweep' if state.get('classics') else ''}",
        f"- {c.get('identified', 0)} search hits, {c.get('unseen', 0)} not yet in the ledger",
        f"- {c.get('title_rejected', 0)} rejected at title triage (G2)",
        f"- {c.get('screened', 0)} screened, {c.get('postponed', 0)} postponed unscreened",
        "- filed: " + (", ".join(f"#{f['issue']}" for f in state["filed"]) or "none"),
        f"- deferred: {c.get('defer_backlog', 0)} for the backlog cap, {c.get('defer_new', 0)} as too new",
        "- rejected after screening: "
        + (", ".join(f"{g} ×{n}" for g, n in sorted(rejected.items())) or "none"),
        f"- waiting for an Impact score: {len(_load(PENDING, []))}",
    ]
    if state.get("search_failures"):
        lines.append(f"- searches that failed: {len(state['search_failures'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect").add_argument(
        "--extra", type=Path, help="JSON lines from WebSearch"
    )
    t = sub.add_parser("triage")
    t.add_argument("--screen")
    t.add_argument("--postpone")
    t.add_argument(
        "--used", type=int, default=0, help="subagents this tick has already used"
    )
    sub.add_parser("batches")
    sub.add_parser("record").add_argument("files", nargs="*", type=Path)
    sub.add_parser("g5-search").add_argument("terms", nargs="+")
    g = sub.add_parser("g5-fail")
    g.add_argument("id")
    g.add_argument("reason")
    sub.add_parser("score").add_argument("--viral", nargs="*", default=[])
    sub.add_parser("decide")
    sub.add_parser("draft-issue").add_argument("id")
    f = sub.add_parser("file-issue")
    f.add_argument("id")
    f.add_argument("body", type=Path)
    f.add_argument("--title", required=True)
    f.add_argument("--dry-run", action="store_true")
    sub.add_parser("finish")
    a = ap.parse_args(argv)

    if a.cmd == "triage" and a.screen is None and a.postpone is None:
        print(triage_list())
        return
    if a.cmd == "draft-issue":
        print(draft_issue(a.id))
        return
    if a.cmd == "finish":
        print(finish())
        return
    out = {
        "collect": lambda: collect(a.extra),
        "triage": lambda: triage_apply(
            parse_spec(a.screen), parse_spec(a.postpone), a.used
        ),
        "batches": batches,
        "record": lambda: record(a.files),
        "g5-search": lambda: g5_search(a.terms),
        "g5-fail": lambda: g5_fail(a.id, a.reason),
        "score": lambda: score(a.viral),
        "decide": decide,
        "file-issue": lambda: file_issue(a.id, a.body, a.title, a.dry_run),
    }[a.cmd]()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
