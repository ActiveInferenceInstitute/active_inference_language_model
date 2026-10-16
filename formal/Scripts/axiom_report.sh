#!/usr/bin/env bash
# Regenerate formal/axiom_report.txt.
#
#   ./Scripts/axiom_report.sh          # from formal/
#
# Fails loudly on a build error or a Lean error rather than leaving a stale or
# truncated report in place: the report is only meaningful if the tree it
# describes actually compiled.
set -euo pipefail

cd "$(dirname "$0")/.."

LAKE="${AILM_LAKE:-lake}"
command -v "$LAKE" >/dev/null || { echo "lake not found (set AILM_LAKE)" >&2; exit 2; }
# A missing rg must fail before any work: its absence would otherwise surface
# only after the build and read as an empty report. This checks presence;
# successful operation is still checked where rg runs.
command -v rg >/dev/null || { echo "rg not found (ripgrep is required to count axiom verdicts)" >&2; exit 2; }

PYTHON="${AILM_PYTHON:-../.venv/bin/python}"
"$PYTHON" ../scripts/record_formal_claims.py --generate-axiom-script
"$LAKE" build
report_temp=$(mktemp ./axiom_report.XXXXXXXX.txt)
trap 'if [ -e "$report_temp" ]; then rm -- "$report_temp"; fi' EXIT
if "$LAKE" env lean Scripts/AxiomReport.lean > "$report_temp"; then
  : # Only a successful Lean run may reach publication below.
else
  lean_status=$?
  # Lean puts diagnostics on stdout too; preserve them before cleaning the stage.
  cat "$report_temp" >&2
  exit "$lean_status"
fi

# A report with no verdict lines is the vacuity failure mode: an axiom gate that
# checks nothing reports clean. Refuse to ship one. A missing or failing rg is a
# tool failure, not vacuity — the lookup above proves presence, not operation —
# so the count runs with its status captured rather than inverted.
rg_status=0
lines=$(rg -c "depends on axioms|does not depend on any axioms" "$report_temp") || rg_status=$?
case "$rg_status" in
  0) ;;
  1)
    echo "axiom_report.txt contains no axiom verdicts — refusing" >&2
    exit 3
    ;;
  *)
    echo "rg failed (exit $rg_status) while counting axiom verdicts — refusing" >&2
    exit 4
    ;;
esac
mv -- "$report_temp" axiom_report.txt
echo "axiom_report.txt: $lines verdicts"
