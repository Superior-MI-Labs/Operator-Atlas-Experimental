#!/usr/bin/env python3
"""Build deterministic, privacy-filtered Operator Atlas source release.

Never package local evidence, logs, candidate checkpoints, virtualenv or test caches.
"""
from pathlib import Path
import hashlib,json,sys,zipfile

ROOT=Path(__file__).resolve().parent.parent
NAME=ROOT.name
INCLUDE_DIRS=(
    'atlas','docs','tests','web','tools','examples','assets','.github',
    'models/starter','models/dense-control','models/stacked-factorized','models/stacked-dense',
)
INCLUDE_FILES=(
    '.gitignore','README.md','AGENTS.md','CONTRIBUTING.md','SECURITY.md','launch.sh','pytest.ini','release.json',
    'requirements.txt','requirements-train.txt','data/corpus.txt',
    'graph-desktop.png','graph-mobile.png',
)
IGNORE_PARTS={'.venv','__pycache__','.pytest_cache','logs','runs','candidates','.git'}


def release_files():
    paths=[ROOT/name for name in INCLUDE_FILES]
    for dirname in INCLUDE_DIRS:
        paths.extend(p for p in (ROOT/dirname).rglob('*') if p.is_file())
    paths.extend(p for p in (ROOT/'results').glob('*.json') if p.is_file())
    good={}
    for path in paths:
        rel=path.relative_to(ROOT)
        if any(p in IGNORE_PARTS for p in rel.parts) or path.is_symlink():continue
        good[rel.as_posix()]=path
    return dict(sorted(good.items()))


def build(output):
    output=Path(output).resolve()
    files=release_files()
    assert all(p.exists() for p in files.values())
    sums={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    data={'schema':1,'algorithm':'sha256','version':json.loads((ROOT/'release.json').read_text())['version'],'files':sums}
    (ROOT/'SHA256SUMS.json').write_text(json.dumps(data,indent=2)+'\n')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        # Explicit Unix modes: `launch.sh` must be executable after extraction
        # even if a source checkout lost its executable bit.
        for name,path in {**files,'SHA256SUMS.json':ROOT/'SHA256SUMS.json'}.items():
            entry=zipfile.ZipInfo(f'{NAME}/{name}',date_time=(2026,10,20,0,0,0))
            entry.create_system=3
            entry.compress_type=zipfile.ZIP_DEFLATED
            entry.external_attr=((0o100755 if name=='launch.sh' else 0o100644)<<16)
            z.writestr(entry,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    return {'archive':str(output),'files':len(files)+1,'bytes':output.stat().st_size}


if __name__=='__main__':
    path=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT.parent/'Superior_MI_Labs_Operator_Atlas_v1.0_Research_Release.zip'
    print(json.dumps(build(path),indent=2))
