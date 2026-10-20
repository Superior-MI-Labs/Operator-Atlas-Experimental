"""Bounded symbolic regression: explicit dictionary search with BIC and held-out evaluation.
This searches a supplied mathematical grammar; it is NOT unrestricted law discovery.
"""
from itertools import combinations
import numpy as np

NAMES=('1','x','x²','x³','sin(x)','cos(x)','sin(2x)','cos(2x)',
       'x·sin(x)','x·cos(x)','x·sin(2x)','x·cos(2x)','x⁴','log(x)','√x','1/x')
def design(x, center=0.0, scale=1.0):
    """Raw-coordinate periodic functions plus an affine polynomial coordinate.

    New coefficients use z=(x-center)/scale; historical models default to
    (0,1). Only positive-domain datasets may select log/sqrt/inverse terms.
    """
    x=np.asarray(x,dtype=np.float64)
    z=(x-center)/scale
    safe=np.maximum(x,1.e-12)
    return np.stack([np.ones_like(x),z,z*z,z*z*z,np.sin(x),np.cos(x),np.sin(2*x),np.cos(2*x),
                     z*np.sin(x),z*np.cos(x),z*np.sin(2*x),z*np.cos(2*x),z**4,
                     np.log(safe),np.sqrt(safe),1/safe],axis=-1)

def discover(points,maximum_terms=3,basis_count=None):
    p=np.asarray(points,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or not (8<=len(p)<=10000) or not np.isfinite(p).all():
        raise ValueError('Provide 8 to 10000 finite [x,y] observations')
    # This version searches raw-coordinate polynomial bases. Refuse badly
    # conditioned magnitudes rather than silently returning unstable coefficients.
    if np.max(np.abs(p[:,0]))>1e9 or np.max(np.abs(p[:,1]))>1e12:
        raise ValueError('Observation magnitude exceeds this bounded basis; rescale the CSV coordinates and values before fitting')
    if np.ptp(p[:,0])<1e-10: raise ValueError('At least two distinct input coordinates required')
    # Sort by seeded shuffle so validation spans the input domain without order bias.
    rng=np.random.default_rng(1729); perm=rng.permutation(len(p)); split=max(5,int(len(p)*.8))
    it,iv=perm[:split],perm[split:]
    # Fit a numerically stable polynomial coordinate for large-magnitude data.
    # Keep periodic and reciprocal functions in original units.
    center=float(np.mean(p[:,0])) if np.max(np.abs(p[:,0]))>10 else 0.0
    scale=float(max(np.max(np.abs(p[:,0]-center)),1.0)) if center else 1.0
    phi=design(p[:,0],center,scale); y=p[:,1]
    selected_count=(len(NAMES) if np.min(p[:,0])>0 else 13) if basis_count is None else int(basis_count)
    if selected_count>13 and np.min(p[:,0])<=0:
        raise ValueError('Positive-only mathematical functions require all x > 0')
    if not 1<=selected_count<=len(NAMES):raise ValueError('Invalid number of declared mathematical basis functions')
    best=None
    for k in range(1,min(4,int(maximum_terms)+1)):
        for cols in combinations(range(selected_count),k):
            coef=np.linalg.lstsq(phi[it][:,cols],y[it],rcond=None)[0]
            residual=y[it]-phi[it][:,cols]@coef
            train_mse=float(np.mean(residual**2))
            bic=len(it)*np.log(max(train_mse,1e-12))+k*np.log(len(it))
            val=float(np.mean((y[iv]-phi[iv][:,cols]@coef)**2))
            row=(bic, val, cols,coef,train_mse)
            if best is None or (bic,val)<(best[0],best[1]): best=row
    bic,val,cols,coef,train_mse=best
    mean_baseline=float(np.sqrt(np.mean((y[iv]-y[it].mean())**2)))
    # Independent stress probe: contiguous high-coordinate holdout, refit on
    # the remaining 85%. Not used to select the equation family. A low error
    # here is evidence of local robustness, not an extrapolation guarantee.
    order=np.argsort(p[:,0]);cut=max(8,int(len(p)*.15))
    train_ix,stress_ix=order[:-cut],order[-cut:]
    stress_coef=np.linalg.lstsq(phi[train_ix][:,cols],y[train_ix],rcond=None)[0]
    stress_error=float(np.sqrt(np.mean((y[stress_ix]-phi[stress_ix][:,cols]@stress_coef)**2)))
    stress_baseline=float(np.sqrt(np.mean((y[stress_ix]-np.mean(y[train_ix]))**2)))
    return {'terms':[NAMES[i] for i in cols],'coefficients':[round(float(v),8) for v in coef],
        'train_rmse':float(np.sqrt(train_mse)),'validation_rmse':float(np.sqrt(val)),
        'blocked_holdout_rmse':stress_error, 'blocked_holdout_mean_rmse':stress_baseline,
        'blocked_holdout_note':'Highest-coordinate 15% withheld from refitting; stress diagnostic only, not proof of extrapolation.',
        'mean_baseline_rmse':mean_baseline,'beats_mean_baseline':bool(np.sqrt(val)<mean_baseline),
        'stored_scalars':len(cols),'basis_functions_considered':selected_count,'observation_count':len(p),
        'coordinate_center':center,'coordinate_scale':scale,
        'equation':' + '.join(f'({float(c):.4g})·{NAMES[i]}' for i,c in zip(cols,coef)),
        'equation_coordinate_note':('x in polynomial/interaction terms denotes z=(raw x - %.8g)/%.8g; periodic/log/sqrt/inverse terms use raw x.'%(center,scale) if center else 'x denotes the input coordinate'),
        'limitation':'Search over declared 13-term compositional grammar; validation and blocked-coordinate stress check do not prove a physical law.'}

def predict(model,coordinates):
    x=np.asarray(coordinates,dtype=np.float64)
    index=[NAMES.index(t) for t in model['terms']]
    return (design(x,float(model.get('coordinate_center',0.0)),float(model.get('coordinate_scale',1.0)))[...,index]@np.asarray(model['coefficients'])).tolist()
