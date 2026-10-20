#!/usr/bin/env bash
# Publish precisely the checksummed public release files, never the working database.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${ATLAS_PUBLISH_REMOTE:-https://github.com/Superior-MI-Labs/Operator-Atlas-Experimental.git}"
if [[ "${1:-}" != "--publish" ]]; then
  printf '%s\n' 'Explicit publication required. Review docs/LICENSING.md and run:'
  printf '  bash tools/publish_github.sh --publish\n'
  exit 2
fi
command -v git >/dev/null || { echo 'Git is required'; exit 2; }
# Fail closed if any shipped artifact changed since qualified packaging.
ATLAS_RELEASE_ROOT="$ROOT" python3 - <<'PY'
import os,hashlib,json
from pathlib import Path
r=Path(os.environ['ATLAS_RELEASE_ROOT']);manifest=json.loads((r/'SHA256SUMS.json').read_text())
for name,expected in manifest['files'].items():
    path=r/name
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Aborted: checksummed release file missing or modified: '+name)
assert len(manifest['files'])>=70
print('Verified public release manifest:',len(manifest['files']),'files')
PY
WORK="$(mktemp -d -t atlas-publish-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
git clone --depth 1 "$REMOTE" "$WORK/repo"
ATLAS_RELEASE_ROOT="$ROOT" ATLAS_REPO_CHECKOUT="$WORK/repo" python3 - <<'PY'
import os,json,shutil
from pathlib import Path
r=Path(os.environ['ATLAS_RELEASE_ROOT']);to=Path(os.environ['ATLAS_REPO_CHECKOUT']);m=json.loads((r/'SHA256SUMS.json').read_text())
for filename in [*m['files'],'SHA256SUMS.json']:
    src=r/filename;dest=to/filename
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dest)
print('Prepared checksummed release files. Existing unrelated repository files were not deleted.')
PY
cd "$WORK/repo"
git add -A
if git diff --cached --quiet; then
 echo 'No differences to publish.';exit 0
fi
printf '\nStaged change summary:\n'
git diff --cached --stat | tail -20
# Do not reset or force-push anything. Git rejects conflicting upstream changes.
git commit -m "release: Operator Atlas v1.0 experimental research software"
git push origin HEAD:main
printf '\nPublished to %s\n' "$REMOTE"
