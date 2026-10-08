#!/usr/bin/env bash
set -euo pipefail
ctrs_scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec /home/gpu_04/CtrS_old/SSA-MRN/.conda-env/bin/python "$ctrs_scripts/a6000_followup_suite.py" "${1:-status}"
