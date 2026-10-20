"""Exact release permissions and reproducibility, independent of unpack tooling."""
import stat
import subprocess
import zipfile
from pathlib import Path
from tools.package import build

ROOT=Path(__file__).resolve().parents[1]

def test_package_preserves_executable_launcher_and_is_deterministic(tmp_path):
    a=tmp_path/'a.zip';b=tmp_path/'b.zip'
    build(a);build(b)
    assert a.read_bytes()==b.read_bytes()
    with zipfile.ZipFile(a) as archive:
        member=f'{ROOT.name}/launch.sh'
        assert stat.S_IMODE(archive.getinfo(member).external_attr>>16)==0o755
    # Python's zipfile.extractall ignores mode bits; the user's Unix unzip
    # restores them. Test the real installation command, not zipfile semantics.
    subprocess.run(['unzip','-q',str(a),'-d',str(tmp_path/'extracted')],check=True)
    assert (tmp_path/'extracted'/ROOT.name/'launch.sh').stat().st_mode & stat.S_IXUSR
