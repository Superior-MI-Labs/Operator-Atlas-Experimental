"""Reproducible numerical experiments and host-local performance measurements."""
import json,time,platform
from pathlib import Path
import numpy as np
from .operators import Operator,factorize,quantize
from .discovery import discover,predict
from .selection import select_operator

def evaluate(seed=41):
    rng=np.random.default_rng(seed)
    n=96;r=8
    a=rng.standard_normal((n,r)).astype(np.float32);b=rng.standard_normal((r,n)).astype(np.float32)
    structured=(a@b).astype(np.float32)
    random=rng.standard_normal((n,n)).astype(np.float32)
    exceptions=structured.copy(); inds=rng.choice(n*n,12,replace=False)
    exceptions.ravel()[inds]+=rng.choice([-1,1],size=12)*4
    records=[]
    for label,w in [('structured',structured),('structured_exceptions',exceptions),('random',random)]:
        ref=Operator('test','dense',w.shape,{'w':w})
        choices=[ref,quantize('test',w)]+[factorize('test',w,rk,extra) for rk,extra in [(4,0),(8,0),(8,24),(16,0)]]
        X=rng.standard_normal((256,n)).astype(np.float32);target=ref.apply(X)
        for op in choices:
            err=float(np.linalg.norm(op.apply(X)-target)/max(np.linalg.norm(target),1e-12))
            records.append({'case':label,'operator':op.kind,'rank':op.spec()['rank'],'sparse_entries':op.spec()['nonzeros'],
                            'relative_output_error':err,'payload_bytes':op.payload_bytes()})
    x=rng.uniform(-2,2,90);y=1.2-.7*x*x+1.5*np.sin(2*x)+rng.normal(0,.03,len(x))
    fit=discover(list(zip(x,y)));pred=np.array(predict(fit,[-1,0,1]))
    # Timings use the same math paths as the runtime; they are device- and load-specific.
    X=rng.standard_normal((1,768)).astype(np.float32)
    aw=rng.standard_normal((768,48)).astype(np.float32);bw=rng.standard_normal((48,768)).astype(np.float32)
    dense=Operator('dense','dense',(768,768),{'w':aw@bw})
    compact=Operator('factor','factorized',(768,768),{'a':aw,'b':bw})
    timings={}
    for op in (dense,compact):
        for _ in range(12):op.apply(X)
        t=[]
        for _ in range(160):
            s=time.perf_counter_ns();op.apply(X);t.append((time.perf_counter_ns()-s)/1e6)
        timings[op.name]={'median_ms':round(float(np.median(t)),5),'min_ms':round(float(np.min(t)),5),'storage_bytes':op.payload_bytes()}
    chooser=[]
    for label,w in [('structured',structured),('random',random)]:
        selected,decision=select_operator(w,rng.standard_normal((200,n)).astype('float32'),.025,shifted_inputs=rng.standard_normal((200,n)).astype('float32'))
        chooser.append({'case':label,'result':decision})
    return {'selector':chooser,'schema':1,'experiment':'synthetic representation + bounded symbolic grammar + host-local microbenchmark',
        'seed':seed,'representation':records,'equation':fit,'equation_example':pred.tolist(),
        'equation_family_study':equation_family_study(),
        'timings':timings,'platform':platform.platform(),'limitations':['Synthetic source matrices intentionally include or exclude low-rank structure.',
        'No held-out language-model quality or GPU tokens-per-second results.',
        'CPU timings are host-local and do not represent production speedups.']}

def write_results(path,seed=41):
    out=evaluate(seed);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False));return out


def equation_family_study(seeds=(3, 7, 11, 17, 23)):
    """Independent sealed samples; old eight terms and new interactions share one search implementation.

    Synthetic function was deliberately constructed from a new interaction
    term. This is a capability regression, not broad symbolic invention.
    """
    results=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        train_x=rng.uniform(-2,2,110)
        held_x=rng.uniform(-2,2,80)
        def target(x,kind):
            if kind=='interaction':return .8+1.4*x*np.sin(2*x)+.45*np.cos(x)
            if kind=='polynomial':return .6-.2*x+.8*x*x
            return np.zeros_like(x)
        for kind in ('interaction','polynomial','unstructured'):
            train_y=(rng.normal(0,1,len(train_x)) if kind=='unstructured' else
                     target(train_x,kind)+rng.normal(0,.01,len(train_x)))
            held_y=(rng.normal(0,1,len(held_x)) if kind=='unstructured' else target(held_x,kind))
            for basis_count in (8,13):
                model=discover(list(zip(train_x,train_y)),basis_count=basis_count)
                out=np.asarray(predict(model,held_x))
                results.append({'seed':seed,'task':kind,'basis_count':basis_count,
                                'selected_terms':model['terms'],
                                'test_rmse':round(float(np.sqrt(np.mean((held_y-out)**2))),6),
                                'validation_rmse':round(model['validation_rmse'],6),
                                'blocked_holdout_rmse':round(model['blocked_holdout_rmse'],6),
                                'mean_test_rmse':round(float(np.sqrt(np.mean((held_y-train_y.mean())**2))),6)})
    summary=[]
    for task in ('interaction','polynomial','unstructured'):
        for basis_count in (8,13):
            rows=[r for r in results if r['task']==task and r['basis_count']==basis_count]
            summary.append({'task':task,'basis_count':basis_count,
                            'mean_test_rmse':round(float(np.mean([r['test_rmse'] for r in rows])),6),
                            'mean_blocked_holdout_rmse':round(float(np.mean([r['blocked_holdout_rmse'] for r in rows])),6),
                            'seeds':len(rows)})
    return {'schema':'atlas-equation-family-study-v1','seeds':list(seeds),'rows':results,'summary':summary,
            'method':'Held-out independent numerical coordinates sampled separately from equation fitting and selection.',
            'limitations':['Interaction dataset deliberately matches an enlarged declared grammar.',
                           'Unstructured test targets are sampled independently; neither equation family can predict them.',
                           'Selected coefficients are not proof of a physical law or broad language capability.']}


def coordinate_family_study():
    """Compare legacy equation grammar and expanded terms with independent x.

    Synthetic truth is not itself chosen from training errors; results should
    not be described as unrestricted symbolic law invention.
    """
    from .discovery import discover, predict
    results=[]
    for seed in (3,7,11,19,29):
        rng=np.random.default_rng(seed)
        x=np.linspace(.7,24.,96)
        y=1.2+2.1*np.log(x)+rng.normal(0,.008,len(x))
        unseen=np.linspace(1.,22.,31)+.075
        truth=1.2+2.1*np.log(unseen)
        for count in (13,16):
            fit=discover(list(zip(x,y)),basis_count=count)
            err=float(np.sqrt(np.mean((np.asarray(predict(fit,unseen))-truth)**2)))
            results.append({'seed':seed,'family_terms':count,'unseen_rmse':err,'selected_terms':fit['terms']})
    large_x=12000+np.linspace(0,200,90)
    large_y=1.5+2.2*((large_x-12100)/100)+.4*((large_x-12100)/100)**2
    fit=discover(list(zip(large_x,large_y)))
    large_err=float(np.max(np.abs(np.asarray(predict(fit,large_x))-large_y)))
    return {'schema':'atlas-coordinate-study-v1','seeds':5,
            'mean_unseen_rmse_13':round(float(np.mean([r['unseen_rmse'] for r in results if r['family_terms']==13])),6),
            'mean_unseen_rmse_16':round(float(np.mean([r['unseen_rmse'] for r in results if r['family_terms']==16])),6),
            'large_coordinate_max_abs_error':large_err,'large_coordinate_center':fit['coordinate_center'],
            'per_run':results,
            'limits':'Synthetic logarithm drawn from an explicitly available candidate basis. No arbitrary law discovery or OOD extrapolation demonstrated.'}


def real_world_sunspot_study(root):
    """Pre-declared, forward-chaining historical stress study.

    No measurements from the test horizon enter equation selection or fitting.
    The production equation endpoint refuses extrapolation; this offline study
    deliberately measures its dangers, not a supported prediction feature.
    """
    import csv
    import hashlib
    root=Path(root)
    path=root/'examples'/'fixtures'/'sunspots_1700_2008.csv'
    raw=path.read_bytes()
    rows=list(csv.DictReader(raw.decode('utf-8').splitlines()))
    years=np.array([int(row['year']) for row in rows], dtype=np.int32)
    values=np.array([float(row['sunspots']) for row in rows], dtype=np.float64)
    if len(rows)!=309 or years[0]!=1700 or years[-1]!=2008 or not np.all(np.diff(years)==1) or not np.isfinite(values).all():
        raise ValueError('Sunspot fixture provenance/shape contract violated')
    results=[]
    for cutoff in (1880,1900,1920,1940,1960,1980):
        tr=years<cutoff;te=(years>=cutoff)&(years<cutoff+15)
        assert tr.sum()>100 and te.sum()==15
        baseline=float(np.mean(values[tr]));last=float(values[np.where(tr)[0][-1]])
        means=float(np.sqrt(np.mean((values[te]-baseline)**2)))
        previous=float(np.sqrt(np.mean((values[te]-last)**2)))
        # Honest stronger control: fit a 2-lag linear autoregression on historical
        # values ONLY, forecast recursively without seeing any horizon targets.
        hist=values[tr]; features=np.column_stack((np.ones(len(hist)-2),hist[1:-1],hist[:-2]))
        beta=np.linalg.lstsq(features,hist[2:],rcond=None)[0]
        a,b=float(hist[-2]),float(hist[-1]); forecast=[]
        for _ in range(15):
            nxt=float(beta@np.array([1.0,b,a]))
            forecast.append(nxt);a,b=b,nxt
        ar2=float(np.sqrt(np.mean((values[te]-np.asarray(forecast))**2)))
        entry={'cutoff_year':cutoff,'train_count':int(tr.sum()),'future_count':int(te.sum()),
               'historical_mean_rmse':round(means,6),'last_training_value_rmse':round(previous,6),
               'autoregressive_2_rmse':round(ar2,6)}
        for count in (13,16):
            learned=discover(list(zip(years[tr],values[tr])),basis_count=count)
            pred=np.array(predict(learned,years[te]),dtype=np.float64)
            entry[f'equation_{count}_rmse']=round(float(np.sqrt(np.mean((values[te]-pred)**2))),6)
            entry[f'equation_{count}_terms']=learned['terms']
            entry[f'equation_{count}_validation_rmse']=round(learned['validation_rmse'],6)
        results.append(entry)
    averages={k:round(float(np.mean([r[k] for r in results])),6) for k in
              ('historical_mean_rmse','last_training_value_rmse','autoregressive_2_rmse','equation_13_rmse','equation_16_rmse')}
    return {'schema':'atlas-historical-stress-v1','dataset':'NOAA NGDC annual sunspot observations 1700-2008 via statsmodels',
            'source':'https://github.com/statsmodels/statsmodels/blob/main/statsmodels/datasets/sunspots/data.py',
            'license':'public domain as stated by statsmodels',
            'sha256':hashlib.sha256(raw).hexdigest(),'rows':len(rows),
            'evaluation':'Six fixed chronological cutoffs; fit on years before each cutoff, test on subsequent 15 years; no horizon values used for selection.',
            'note':'Off-domain stress ONLY: deployed equation predictions intentionally refuse coordinates outside observed range. Backtest is not an approved forecasting tool.',
            'averages':averages,'runs':results,
            'limitations':['One-dimensional year->activity prediction ignores solar cycle temporal dynamics and domain knowledge.',
                           'The candidate formulas do not include realistic ~11-year solar-cycle terms.',
                           'Baselines include an ordinary recursively forecast AR(2), still not a state-of-the-art forecasting method.',
                           'Six temporally overlapping historical splits are not six independent experiments.',
                           'No information from holdout years is used to choose equation terms.']}
