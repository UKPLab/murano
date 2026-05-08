#!/usr/bin/env bash
# Bootstrap repository labels for the issue-form auto-labeler.
#
# Usage: run once from a clone with `gh auth status` showing a writable token:
#   bash .github/scripts/bootstrap-labels.sh
#
# Idempotent: re-running updates color/description on existing labels.
# Does not delete labels: manual cleanup if you remove categories.

set -euo pipefail

REPO="${REPO:-UKPLab/murano}"

create_or_update() {
  local name="$1" color="$2" desc="$3"
  if gh label list --repo "$REPO" --search "$name" --json name --jq '.[].name' 2>/dev/null | grep -Fxq "$name"; then
    gh label edit "$name" --repo "$REPO" --color "$color" --description "$desc" >/dev/null
    printf '  ✓ updated  %s\n' "$name"
  else
    gh label create "$name" --repo "$REPO" --color "$color" --description "$desc" >/dev/null
    printf '  + created  %s\n' "$name"
  fi
}

echo "Bootstrapping labels on $REPO ..."

echo
echo "Type labels (already exist; refreshing colors)"
create_or_update "bug"           "fc2929" "Something isn't working"
create_or_update "enhancement"   "fbca04" "New feature or request"
create_or_update "documentation" "0075ca" "Improvements or additions to documentation"
create_or_update "compatibility" "5319e7" "Cross-architecture or dependency-version compatibility issue"
create_or_update "reproduction"  "d4c5f9" "Reproduction-gallery submission"

echo
echo "Area labels (which part of Murano is involved)"
create_or_update "area:record"     "1d76db" "Recording / activation extraction"
create_or_update "area:intervene"  "1d76db" "Interventions / generation modification"
create_or_update "area:analysis"   "1d76db" "Analysis / probing / lenses"
create_or_update "area:io"         "1d76db" "I/O / saving artifacts"
create_or_update "area:pipeline"   "1d76db" "Pipeline / Step framework"
create_or_update "area:plotting"   "1d76db" "Plotting / visualization"
create_or_update "area:cross-arch" "1d76db" "Cross-architecture / model loading"

echo
echo "Architecture labels"
create_or_update "arch:llama"   "0e8a16" "Llama family"
create_or_update "arch:gpt2"    "0e8a16" "GPT-2 family"
create_or_update "arch:mistral" "0e8a16" "Mistral family"
create_or_update "arch:qwen"    "0e8a16" "Qwen family"
create_or_update "arch:opt"     "0e8a16" "OPT family"

echo
echo "Hardware labels"
create_or_update "hardware:gpu"       "fef2c0" "Single CUDA GPU"
create_or_update "hardware:multi-gpu" "fef2c0" "Multi-GPU (device_map=auto)"
create_or_update "hardware:cpu"       "fef2c0" "CPU only"
create_or_update "hardware:mps"       "fef2c0" "Apple MPS"

echo
echo "Upstream-cause labels (for compatibility issues)"
create_or_update "upstream:nnsight"      "c5def5" "Suspected upstream cause: nnsight"
create_or_update "upstream:nnterp"       "c5def5" "Suspected upstream cause: nnterp"
create_or_update "upstream:transformers" "c5def5" "Suspected upstream cause: transformers"
create_or_update "upstream:torch"        "c5def5" "Suspected upstream cause: torch"

echo
echo "Status labels"
create_or_update "regression"  "b60205" "Worked in a previous version"
create_or_update "has-pr"      "0e8a16" "Author has opened or is preparing a PR"

echo
echo "Priority labels (user-reported, treated as signal not commitment)"
create_or_update "priority:critical"     "b60205" "Critical: blocking the reporter's work"
create_or_update "priority:high"         "d93f0b" "High: significant impact on productivity"
create_or_update "priority:medium"       "fbca04" "Medium: would be very helpful"
create_or_update "priority:nice-to-have" "c2e0c6" "Nice to have"

echo
echo "Impact labels (for documentation issues)"
create_or_update "impact:high"   "d93f0b" "High: blocks users from using a feature"
create_or_update "impact:medium" "fbca04" "Medium: feature is usable but takes guesswork"
create_or_update "impact:low"    "c2e0c6" "Low: minor confusion or polish"

echo
echo "Done."
