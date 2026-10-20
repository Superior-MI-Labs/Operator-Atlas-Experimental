import numpy as np,pytest
from atlas.discovery import discover,predict

def test_known_equation_fit():
    x=np.linspace(-2,2,100);y=1.3-.5*x*x+1.1*np.sin(2*x)
    r=discover(list(zip(x,y)))
    assert r['validation_rmse']<1e-5
    assert set(r['terms'])=={'1','x²','sin(2x)'}
    assert np.allclose(predict(r,[-1,0,1]),1.3-.5*np.array([-1,0,1])**2+1.1*np.sin(2*np.array([-1,0,1])),atol=2e-6)

def test_unsupported_data_rejected():
    with pytest.raises(ValueError):discover([[1,2],[2,3]])
    with pytest.raises(ValueError):discover([[1,float('nan')]]*10)
    with pytest.raises(ValueError):discover([[1,i] for i in range(10)])
