import numpy as np,pytest
from atlas.operators import Operator,factorize,quantize
from atlas.selection import select_operator

def test_factorized_direct_parity_multiple_seeds():
    for seed in range(8):
        rng=np.random.default_rng(seed);a=rng.standard_normal((29,6)).astype('f4');b=rng.standard_normal((6,37)).astype('f4')
        op=Operator('x','factorized',(29,37),{'a':a,'b':b});x=rng.standard_normal((13,37)).astype('f4')
        assert np.allclose(op.apply(x),x@(a@b).T,atol=2e-5)
        assert op.payload_bytes() < (29*37*4)

def test_true_sparse_correction_without_dense_residual():
    rng=np.random.default_rng(50);a=rng.standard_normal((12,3)).astype('f4');b=rng.standard_normal((3,16)).astype('f4');w=a@b
    w[2,7]+=5;w[9,0]-=2
    op=factorize('x',w,3,2);x=rng.standard_normal((7,16)).astype('f4')
    assert op.spec()['nonzeros']==2
    assert np.allclose(op.apply(x), x@op.dense_equivalent().T,atol=1e-5)
    assert np.linalg.norm(op.apply(x)-x@w.T)<.15 # approximate low rank learned via SVD

def test_quantization_and_validation():
    rng=np.random.default_rng(2);w=rng.normal(size=(22,16)).astype('f4');q=quantize('q',w)
    assert q.payload_bytes()==22*16+4
    assert np.linalg.norm(q.dense_equivalent()-w)/np.linalg.norm(w)<.013
    with pytest.raises(ValueError):Operator('bad','factorized',(4,6),{'a':np.zeros((4,2)),'b':np.zeros((3,6))})
    with pytest.raises(ValueError):q.apply(np.zeros(4))

def test_selector_uses_dense_for_strict_random_tolerance():
    rng=np.random.default_rng(55);w=rng.normal(size=(24,24)).astype('f4');x=rng.normal(size=(100,24)).astype('f4')
    op,report=select_operator(w,x,.00001,shifted_inputs=rng.normal(size=(35,24)).astype('f4'))
    assert op.kind=='dense'
    assert len(report['alternatives'])>3
    assert report['selected_bytes']==w.nbytes

def test_selector_compresses_low_rank():
    rng=np.random.default_rng(11);w=(rng.normal(size=(28,2))@rng.normal(size=(2,28))).astype('f4')
    op,report=select_operator(w,rng.normal(size=(70,28)).astype('f4'),.001)
    assert op.kind=='factorized'
    assert op.payload_bytes()<w.nbytes
