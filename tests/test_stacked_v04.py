"""Stacked model contract: one operator implementation, checkpoint-derived graph."""
import tempfile
from pathlib import Path
import numpy as np
import pytest
from atlas.model import CharacterModel
from atlas.graph import graph
from atlas.operators import Operator

ROOT=Path(__file__).resolve().parent.parent


def test_two_layer_checkpoint_and_trace():
    model=CharacterModel(ROOT/'models'/'stacked-factorized')
    assert model.layers==2 and model.state_dim==model.hidden*2
    state=np.zeros(model.state_dim,dtype=np.float32)
    state,logits,vectors=model.step(model.unknown,state)
    assert state.shape==(model.state_dim,) and np.isfinite(logits).all()
    assert {'input','recurrent','activation','input_1','recurrent_1','activation_1','logits'}<=vectors.keys()
    assert np.allclose(vectors['activation'],state[:model.hidden])
    assert np.allclose(vectors['activation_1'],state[model.hidden:])
    with pytest.raises(ValueError):model.step(model.unknown,np.zeros(model.hidden,dtype=np.float32))
    assert model.generate('A mathematical model',15,seed=7)==model.generate('A mathematical model',15,seed=7)


def test_graph_matches_real_layer_ops():
    model=CharacterModel(ROOT/'models'/'stacked-factorized')
    g=graph(model);nodes={n['id']:n for n in g['nodes']}
    assert len(nodes)==20 and 'composition' in nodes and g['layers']==2
    for n in nodes.values():
        if n.get('operator'):assert n['operator'] in model.operators
    edges={(e['from'],e['to']) for e in g['edges']}
    assert ('layer1_activation','head') in edges
    assert ('activation','head') not in edges
    assert all(e[0] in nodes and e[1] in nodes for e in edges)


def test_legacy_checkpoint_remains_one_layer():
    m=CharacterModel(ROOT/'models'/'starter')
    assert m.layers==1
    assert graph(m)['layers']==1 and len(graph(m)['nodes'])==16
