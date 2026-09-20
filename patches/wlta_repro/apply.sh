#!/usr/bin/env bash
# Apply the WLTA-reproduction + --ta_source changes to a checkout of the group's code.
#   usage: patches/wlta_repro/apply.sh /path/to/delta_wlta
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dst="${1:?path to the delta_wlta checkout (the dir that contains src/)}"
cp "$here/wlta_fixes.py" "$dst/src/wlta_fixes.py"
patch -p1 -d "$dst" < "$here/train_window_tokenizer.patch"
patch -p1 -d "$dst" < "$here/atba_loss.patch"
python3 -m py_compile "$dst/src/train_window_tokenizer.py" "$dst/src/atba_loss.py" "$dst/src/wlta_fixes.py"
echo "applied + compiled (hybrid_ta is imported from this repo: pip install -e <repo>)"
