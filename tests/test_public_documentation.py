"""Publication invariants, separate from neural-research claims."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_published_documentation_links_and_metrics():
    r=subprocess.run([sys.executable,str(ROOT/'tools/audit_publication.py')],capture_output=True,text=True,timeout=15)
    assert r.returncode==0,r.stdout+'\n'+r.stderr

def test_release_includes_research_pages_and_agent_guide():
    sys.path.insert(0,str(ROOT/'tools'))
    from package import release_files
    f=release_files()
    must={'README.md','AGENTS.md','CONTRIBUTING.md','SECURITY.md',
          'docs/WHITEPAPER_V1.md','docs/Operator_Atlas_Whitepaper_v1.pdf',
          'docs/EXPERIMENT_CATALOG.md','docs/REPRODUCIBILITY.md','docs/RESEARCH_DECISION.md',
          'assets/architecture_v1.png','assets/atlas-header.svg','graph-desktop.png',
          'results/historical_sunspots_v1.json','.github/ISSUE_TEMPLATE/reproduction.md'}
    assert not (must-set(f)),must-set(f)
    assert all(not any(x in name for x in ('evidence.sqlite3','events.jsonl','.venv')) for name in f)
