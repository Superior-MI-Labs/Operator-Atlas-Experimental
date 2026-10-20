from pathlib import Path
import numpy as np
from atlas.model import CharacterModel
from atlas.graph import graph
ROOT=Path(__file__).resolve().parent.parent

def test_checkpoint_generation_deterministic_and_traced():
    m=CharacterModel(ROOT/'models/starter');a=m.generate('The mathematical atlas',24,seed=44)
    b=m.generate('The mathematical atlas',24,seed=44)
    assert a==b and len(a['response'])==24
    assert all('top' in t for t in a['trace'])
    assert m.status()['operator_specs']['recurrent']['kind']=='factorized'

def test_checkpoint_has_measured_torch_numpy_parity():
    m=CharacterModel(ROOT/'models/starter')
    assert m.manifest['training']['numpy_torch_max_error'] < 1e-4
    assert m.status()['parameter_bytes'] > 0

def test_graph_has_known_node_targets():
    g=graph();nodes={n['id'] for n in g['nodes']}
    assert nodes and all(edge['from'] in nodes and edge['to'] in nodes for edge in g['edges'])
    assert all((not n.get('operator')) or n['operator'] in ('input','recurrent','head') for n in g['nodes'])
