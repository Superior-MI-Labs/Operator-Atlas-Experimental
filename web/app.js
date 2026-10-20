'use strict';
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const S={graph:null,status:null, selected:null, scale:.72, x:25,y:70, layout:{}, active:false,lastVectors:null,lastTrace:null,routeNodes:[]};
async function call(path,payload){const opts=payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)};
 const r=await fetch('/api/'+path,opts);let data;try{data=await r.json();}catch{throw Error('Invalid API response');}if(!r.ok)throw Error(data.error||'HTTP '+r.status);return data;}
function metric(label,value){return `<div class="metric"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`}
function showErr(node,e){node.textContent='Error: '+(e?.message||e)}
function initTabs(){document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.page').forEach(n=>n.classList.remove('active'));b.classList.add('active');$('page-'+b.dataset.tab).classList.add('active');if(b.dataset.tab==='graph')requestAnimationFrame(fitGraph)});}
function center(n){const p=S.layout[n.id]||{x:n.x,y:n.y};return {x:p.x+87,y:p.y+44};}
function edgePath(edge){
 const a=S.graph.nodes.find(n=>n.id===edge.from),b=S.graph.nodes.find(n=>n.id===edge.to);
 const p=center(a),q=center(b),c=Math.max(70,Math.abs(q.x-p.x)*.38);
 if(edge.from==='sampler'&&edge.to==='input'){
   return `M${p.x+85},${p.y} C${p.x+150},${p.y} ${p.x+150},98 ${p.x+105},98 L${q.x-100},98 C${q.x-150},98 ${q.x-150},${q.y} ${q.x-85},${q.y}`;
 }
 return `M${p.x+85},${p.y} C${p.x+85+c},${p.y} ${q.x-85-c},${q.y} ${q.x-85},${q.y}`;
}
function miniActivation(values){if(!values)return '';const nums=values.slice(0,12);const mx=Math.max(1e-10,...nums.map(x=>Math.abs(x)));
 return '<div class="node-activity" title="Measured last-step vector (not decorative)">'+nums.map(v=>`<span style="opacity:${(.2+.8*Math.min(1,Math.abs(v)/mx)).toFixed(2)};background:${v>=0?'#62dec1':'#df88a6'}"></span>`).join('')+'</div>';
}
function renderGraph(){const g=S.graph;if(!g)return;
 const svg=$('graph-connections');const width=Math.max(1800,...g.nodes.map(n=>(S.layout[n.id]||n).x+240));
 const height=Math.max(880,...g.nodes.map(n=>(S.layout[n.id]||n).y+160));
 $('graph-world').style.width=width+'px';$('graph-world').style.height=height+'px';
 svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.innerHTML='';
 g.edges.forEach((edge)=>{const a=g.nodes.find(n=>n.id===edge.from),b=g.nodes.find(n=>n.id===edge.to);if(!a||!b)return;
 const path=document.createElementNS('http://www.w3.org/2000/svg','path');path.setAttribute('d',edgePath(edge));
 path.dataset.source=a.id;path.dataset.target=b.id;if((S.active&&!['research','memory'].includes(a.type)&&!['research','memory'].includes(b.type))||(S.routeNodes.includes(a.id)&&S.routeNodes.includes(b.id)))path.classList.add('active');svg.appendChild(path);});
 const layer=$('graph-nodes');layer.innerHTML='';
 g.nodes.forEach(n=>{const p=S.layout[n.id]||{x:n.x,y:n.y};const node=document.createElement('div');node.className=`graph-node ${n.type||''} ${S.selected===n.id?'selected':''} ${(S.active&&!['research','memory'].includes(n.type))||S.routeNodes.includes(n.id)?'firing':''}`;
 node.style.left=p.x+'px';node.style.top=p.y+'px';node.dataset.id=n.id;
 const op=n.operator&&S.status?.model?.operator_specs?.[n.operator];const meta=op?`${op.kind} • ${op.shape.join(' × ')}`:(n.type==='memory'?'source-backed':'inspect subsystem');
 const vectors={embedding:'embedding',input_op:'input',recurrent_op:'recurrent',state:'activation',activation:'activation',head:'logits'};
 const vec=S.lastVectors?.[n.trace_key||vectors[n.id]];
 node.innerHTML=`<span class="node-port left"></span><div class="node-kind">${esc(n.type)}</div><div class="node-title">${esc(n.name)}</div><div class="node-meta">${esc(meta)}</div>${miniActivation(vec)}<span class="node-port right"></span>`;
 node.addEventListener('pointerdown',e=>{if(e.button!==0)return;e.stopPropagation();node.setPointerCapture(e.pointerId);node.dataset.dragx=e.clientX;node.dataset.dragy=e.clientY;node.dataset.moved='0';});
 node.addEventListener('pointermove',e=>{if(!node.hasPointerCapture(e.pointerId))return;const dx=(e.clientX-Number(node.dataset.dragx))/S.scale,dy=(e.clientY-Number(node.dataset.dragy))/S.scale;
 if(Math.abs(dx)+Math.abs(dy)>.3)node.dataset.moved='1';const p=S.layout[n.id]||{x:n.x,y:n.y};S.layout[n.id]={x:p.x+dx,y:p.y+dy};node.dataset.dragx=e.clientX;node.dataset.dragy=e.clientY;
 node.style.left=S.layout[n.id].x+'px';node.style.top=S.layout[n.id].y+'px';updateLines();});
 node.addEventListener('pointerup',e=>{if(node.hasPointerCapture(e.pointerId))node.releasePointerCapture(e.pointerId);if(node.dataset.moved==='0')selectNode(n.id);});
 layer.appendChild(node);});
 updateTransform();}
function updateLines(){if(!S.graph)return;const paths=$('graph-connections').querySelectorAll('path');S.graph.edges.forEach((edge,i)=>{paths[i]?.setAttribute('d',edgePath(edge));});}
function updateTransform(){$('graph-world').style.transform=`translate(${S.x}px,${S.y}px) scale(${S.scale})`;$('zoom-label').textContent=Math.round(S.scale*100)+'%'}
function fitGraph(){const vp=$('graph-viewport');if(!vp.clientWidth||!S.graph)return;const nodes=S.graph.nodes;const xs=nodes.map(n=>(S.layout[n.id]||n).x),ys=nodes.map(n=>(S.layout[n.id]||n).y);
 const mnx=Math.min(...xs)-40,mny=Math.min(...ys)-40,maxx=Math.max(...xs)+214,maxy=Math.max(...ys)+130;
 S.scale=Math.min(1.05,Math.max(.28,Math.min((vp.clientWidth-45)/(maxx-mnx),(vp.clientHeight-65)/(maxy-mny))));
 S.x=(vp.clientWidth-(maxx-mnx)*S.scale)/2-mnx*S.scale;S.y=(vp.clientHeight-(maxy-mny)*S.scale)/2-mny*S.scale;updateTransform();}
function zoom(mult){S.scale=Math.max(.25,Math.min(2.5,S.scale*mult));updateTransform();}
function registerGraphControls(){const vp=$('graph-viewport');let pan=null;
 vp.addEventListener('wheel',e=>{e.preventDefault();const r=vp.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,old=S.scale;
 S.scale=Math.max(.25,Math.min(2.5,S.scale*(e.deltaY<0?1.12:.89)));S.x=mx-(mx-S.x)*(S.scale/old);S.y=my-(my-S.y)*(S.scale/old);updateTransform();},{passive:false});
 vp.addEventListener('pointerdown',e=>{if(e.button!==0||e.target.closest('.graph-node')||e.target.closest('button'))return;pan={x:e.clientX,y:e.clientY};vp.setPointerCapture(e.pointerId);vp.classList.add('dragging');});
 vp.addEventListener('pointermove',e=>{if(!pan)return;S.x+=e.clientX-pan.x;S.y+=e.clientY-pan.y;pan={x:e.clientX,y:e.clientY};updateTransform();});
 vp.addEventListener('pointerup',()=>{pan=null;vp.classList.remove('dragging');});
 $('zoom-in').onclick=()=>zoom(1.2);$('zoom-out').onclick=()=>zoom(1/1.2);$('zoom-fit').onclick=fitGraph;$('graph-fit').onclick=fitGraph;
 $('graph-reset').onclick=()=>{S.layout={};renderGraph();fitGraph()};}
function heatmapHTML(arr){if(!arr||!arr.length)return '';const max=Math.max(.0001,...arr.flat().map(Math.abs));let h=`<div class="heatmap" style="grid-template-columns:repeat(${arr[0].length},1fr)">`;
 for(const row of arr)for(const v of row){const alpha=Math.max(.1,Math.min(1,Math.abs(v)/max));const rgb=v>=0?`64,213,169`:`215,105,143`;
 h+=`<span title="${esc(v)}" style="background:rgba(${rgb},${alpha.toFixed(3)})"></span>`;}return h+'</div><p class="heatmap-caption">Numerically sampled matrix entries • green positive, rose negative</p>'}
async function selectNode(id){S.selected=id;renderGraph();const node=S.graph.nodes.find(n=>n.id===id);$('inspect-name').textContent=node.name;
 $('inspect-content').innerHTML=`<p>${esc(node.about)}</p>${metric('Category',node.type)}<p class="muted">Inspecting actual checkpoint metadata.</p>`;
 if(node.operator){try{const o=await call('operator/'+node.operator);if(S.selected!==id)return;
 $('inspect-content').innerHTML=`<p>${esc(node.about)}</p>${metric('Representation',o.spec.kind)}${metric('Shape',o.spec.shape.join(' × '))}${metric('Stored payload',o.spec.payload_bytes+' B')}${o.spec.rank?metric('Rank',o.spec.rank):''}${o.spec.nonzeros?metric('Sparse corrections',o.spec.nonzeros):''}<div class="panel-label" style="margin-top:18px">ACTUAL WEIGHT SAMPLES</div>${heatmapHTML(o.heatmap)}${S.lastVectors?.[node.operator]?'<div class="panel-label">LAST STEP OUTPUT VECTOR</div><div class="output-box">'+S.lastVectors[node.operator].slice(0,20).map(v=>Number(v).toFixed(3)).join(', ')+'</div>':''}`;
 }catch(e){showErr($('inspect-content'),e)}}}
async function initGraph(){S.graph=await call('graph');S.status=await call('status');$('server-status').textContent='● Local engine connected';$('server-status').classList.add('ok');
 const m=S.status.model;$('graph-metrics').innerHTML=[metric('Model',m?.kind||'Not trained'),metric('Stored weights',m?m.parameter_bytes+' B':'No checkpoint'),metric('Hidden state',m?m.hidden+' dimensions':'—'),metric('Training',m?.training?.steps+' steps' || '—'),metric('Document sources',S.status.evidence.source_count)].join('');
 renderGraph();fitGraph();registerGraphControls();selectNode('input_op');}
async function execute(prompt,resultNode,fromGraph=false){const data=await call('generate',{prompt,max_new:fromGraph?40:Number($('token-count').value),temperature:fromGraph?.65:Number($('temperature').value),seed:12});
 S.lastVectors=data.last_vectors;S.lastTrace=data.trace;renderGraph();if(S.selected)selectNode(S.selected);
 if(fromGraph){resultNode.textContent=prompt+data.response;$('graph-toast').textContent='✓ Actual recurrent state updated; output generated from trained model.';}
 else{resultNode.textContent=data.prompt+data.response;$('token-trace').innerHTML=data.trace.map(t=>`<div class="trace-row"><b>Step ${t.step} · ${esc(JSON.stringify(t.token))}</b><small>${t.top.map(k=>`${esc(JSON.stringify(k.token))}: ${(k.probability*100).toFixed(1)}%`).join(' / ')} · state norm ${t.activation_strength}</small></div>`).join('');}
 return data;}
async function resolveAtlas(){
 const b=$('answer-run');b.disabled=true;
 try{
  const result=await call('respond',{question:$('answer-question').value,mode:$('answer-mode').value,max_new:65,seed:12});
  $('answer-output').textContent=result.response;
  $('answer-metadata').innerHTML=metric('Route',result.route)+metric('Evidence supported',result.supported?'Yes (limited to source or fitted domain)':'No')+metric('Scientific disclosure',result.disclosure);
  $('answer-trace').innerHTML=result.trace.map((n,i)=>`<div class="trace-row"><b>${i+1}. ${esc(n)}</b><small>Real declared backend path; not a timing animation.</small></div>`).join('');
  $('answer-evidence').innerHTML=result.evidence.length?result.evidence.map(e=>`<div class="result-card"><b>${esc(e.title)} · Source #${e.source_id}</b><small>SHA ${esc(e.sha256.slice(0,24))}…</small><p>${esc(e.text)}</p></div>`).join(''):result.equation?`<div class="result-card">${esc(JSON.stringify(result.equation,null,2))}</div>`:'No retrieved source. No unsupported claims were generated.';
  S.routeNodes=result.trace;renderGraph();
 }catch(e){showErr($('answer-output'),e)}finally{b.disabled=false}
}
function initAnswer(){
 $('answer-run').onclick=resolveAtlas;
 $('answer-example').onclick=async()=>{const b=$('answer-example');b.disabled=true;
  try{const r=await call('load_example',{name:'research_notebook.md'});$('answer-import-status').textContent=`Imported fictional source #${r.source_id}; content identity preserved. Ask again to compare. `;await refreshSources();}
  catch(e){showErr($('answer-import-status'),e)}finally{b.disabled=false}
 };
}
function initGenerate(){document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>$('prompt').value=b.dataset.prompt);
 $('generate').onclick=async()=>{const button=$('generate');button.disabled=true;try{await execute($('prompt').value,$('generated'))}catch(e){showErr($('generated'),e)}finally{button.disabled=false}};
 $('graph-run').onclick=async()=>{const b=$('graph-run');b.disabled=true;S.active=true;renderGraph();try{await execute($('graph-prompt').value,$('graph-result'),true)}catch(e){showErr($('graph-result'),e)}finally{setTimeout(()=>{S.active=false;renderGraph()},1600);b.disabled=false}};}
function fillPoints(which){const lines=[];let seed=47;function rand(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;}
 for(let i=0;i<48;i++){const x=-2.4+i*.1;let y=which==='wave'?1.3-.5*x*x+2*Math.sin(2*x):which==='curve'?.8+1.2*x-.3*x*x:(rand()*2-1)*3;
 y+=(rand()-.5)*.04;lines.push(`${x.toFixed(3)}, ${y.toFixed(5)}`);}$('points').value=lines.join('\n');}
function plotEquation(points,preds){const svg=$('equation-chart');const W=440,H=240,P=24;
 const xs=points.map(a=>a[0]),ys=points.map(a=>a[1]).concat(preds),mnx=Math.min(...xs),mxx=Math.max(...xs),mny=Math.min(...ys),mxy=Math.max(...ys),xx=x=>P+(x-mnx)/(mxx-mnx||1)*(W-2*P),yy=y=>H-P-(y-mny)/(mxy-mny||1)*(H-2*P);
 const rows=points.map((p,i)=>({x:p[0],y:preds[i]})).sort((a,b)=>a.x-b.x);
 svg.innerHTML=`<path d="${rows.map((p,i)=>(i?'L':'M')+xx(p.x).toFixed(1)+','+yy(p.y).toFixed(1)).join(' ')}" stroke="#6ee0c3" stroke-width="2" fill="none"/>`+
 points.map(p=>`<circle cx="${xx(p[0]).toFixed(1)}" cy="${yy(p[1]).toFixed(1)}" r="2.8" fill="#edbd84"/>`).join('');}
function initEquations(){fillPoints('wave');document.querySelectorAll('[data-data]').forEach(b=>b.onclick=()=>fillPoints(b.dataset.data));$('fit-equation').onclick=async()=>{
 const pts=$('points').value.trim().split('\n').map(s=>s.split(/[\s,;]+/).map(Number));try{if(pts.some(a=>a.length!==2||a.some(x=>!Number.isFinite(x))))throw Error('Enter finite x,y pairs, one per line');
 const res=await call('fit_equation',{points:pts});$('equation-status').textContent=res.equation;plotEquation(pts.slice(0,80),res.predictions);
 $('equation-stats').innerHTML=metric('Equation terms',res.terms.join(' + '))+metric('Validation RMSE',res.validation_rmse.toFixed(5))+metric('Mean baseline RMSE',res.mean_baseline_rmse.toFixed(5))+metric('Outperforms mean',res.beats_mean_baseline?'Yes':'No')+metric('High-coordinate blocked holdout RMSE',Number(res.blocked_holdout_rmse).toFixed(5));
 }catch(e){showErr($('equation-status'),e)}};}
async function refreshSources(){const r=await call('sources',{limit:40});$('source-index').innerHTML=r.sources.length?r.sources.map(a=>`<div class="result-card"><b>${esc(a.title)}</b><small> #${a.source_id} · ${a.characters} chars · ${esc(a.media_type||'text/plain')}</small><p>SHA256 ${esc(a.sha256.slice(0,24))}…</p></div>`).join(''):'No documents yet.';}
function fileToBase64(file){return file.arrayBuffer().then(buffer=>{const bytes=new Uint8Array(buffer),chunks=[];for(let i=0;i<bytes.length;i+=8192)chunks.push(String.fromCharCode(...bytes.subarray(i,i+8192)));return btoa(chunks.join(''));});}
function initMemory(){
 $('import-file').onclick=async()=>{const b=$('import-file'),file=$('document-file').files?.[0];if(!file){$('file-status').textContent='Choose a file first.';return;}if(file.size>5000000){$('file-status').textContent='File is over the 5 MB limit.';return;}
 b.disabled=true;$('file-status').textContent='Decoding and preserving source locally…';try{const encoded=await fileToBase64(file),r=await call('ingest_file',{filename:file.name,base64:encoded});$('file-status').textContent=`Preserved ${r.filename} · source #${r.source_id} · ${r.characters} characters · SHA ${r.sha256.slice(0,16)}…`;await refreshSources();await refreshFields();}catch(e){showErr($('file-status'),e)}finally{b.disabled=false}};
 $('refresh-sources').onclick=()=>refreshSources().catch(e=>showErr($('source-index'),e));
 $('ask-sources').onclick=async()=>{try{const r=await call('ask_sources',{question:$('search-text').value,limit:3});$('evidence-warning').textContent=r.answer+' '+r.method;$('search-results').innerHTML=r.passages.map(a=>`<div class="result-card"><b>${esc(a.title)}</b><small> Source #${a.source_id}:${a.chunk_id} · SHA ${esc(a.sha256.slice(0,18))}…</small><p>${esc(a.text)}</p></div>`).join('');}catch(e){showErr($('evidence-warning'),e)}};
 $('ingest').onclick=async()=>{try{const r=await call('ingest',{title:$('doc-title').value,text:$('doc-text').value});$('ingest-result').textContent=`Preserved source #${r.source_id} · SHA256 ${r.sha256.slice(0,18)}… · ${r.chunks} chunk(s)`;}catch(e){showErr($('ingest-result'),e)}};
 $('search').onclick=async()=>{try{const r=await call('search',{query:$('search-text').value});$('search-results').innerHTML=r.results.length?r.results.map(a=>`<div class="result-card"><b>${esc(a.title)}</b><small> #${a.source_id}:${a.chunk_id} · ${a.term_matches} matches</small><p>${esc(a.text)}</p></div>`).join(''):'No matching source passages.';}catch(e){showErr($('search-results'),e)}};}
async function refreshFields(){
 const [sources,models]=await Promise.all([call('sources',{limit:100}),call('equations',{limit:100})]);
 const current=$('field-source').value,chosen=$('saved-equations').value;
 const csv=sources.sources.filter(s=>s.media_type==='text/csv');
 $('field-source').innerHTML=csv.length?csv.map(s=>`<option value="${s.source_id}">${esc(s.title)} (#${s.source_id})</option>`).join(''):'<option value="">Import a CSV in Evidence first</option>';
 if(csv.some(s=>String(s.source_id)===current))$('field-source').value=current;
 $('saved-equations').innerHTML=models.equations.length?models.equations.map(m=>`<option value="${m.equation_id}">${esc(m.title)}: ${esc(m.equation)} (#${m.equation_id})</option>`).join(''):'<option value="">No qualified equations</option>';
 if(models.equations.some(m=>String(m.equation_id)===chosen))$('saved-equations').value=chosen;
 for(const id of ['compose-first','compose-second']){
  const previous=$(id).value;
  $(id).innerHTML=models.equations.length?models.equations.map(m=>`<option value="${m.equation_id}">#${m.equation_id} ${esc(m.title)} (${esc(m.x_column)} → ${esc(m.y_column)})</option>`).join(''):'<option value="">No equations</option>';
  if(models.equations.some(m=>String(m.equation_id)===previous))$(id).value=previous;
 }
 if(models.equations.length>1&&!$('compose-second').value)$('compose-second').selectedIndex=1;
 if(models.equations.length>1&&$('compose-first').value===$('compose-second').value)$('compose-second').selectedIndex=1;
}
function initFieldLearning(){
 $('compose-run').onclick=async()=>{
  const b=$('compose-run');b.disabled=true;
  try{
   const first=Number($('compose-first').value),second=Number($('compose-second').value),x=Number($('compose-x').value);
   if(!first||!second||!Number.isFinite(x))throw Error('Select two trained equations and enter a finite x');
   const data=await call('compose_equations',{equation_ids:[first,second],x});
   $('compose-output').innerHTML=metric('Status',data.supported?'Computed':'Stopped (domain refusal)')+
    data.steps.map((step,i)=>`<div class="trace-row"><b>${i+1}. Equation #${step.equation_id} · source #${step.source_id}</b><small>Input: ${esc(step.x)} → ${step.supported?esc(step.value):esc(step.reason)} | SHA ${esc(step.source_sha256.slice(0,16))}…</small></div>`).join('')+
    (data.supported?metric('Output',data.value):metric('Refusal',data.reason))+
    `<p class="muted">${esc(data.disclosure)}</p>`;
   S.routeNodes=['input','answer_router','equation','composition','answer_output'];renderGraph();
  }catch(e){showErr($('compose-output'),e)}finally{b.disabled=false}
 };
 $('field-refresh').onclick=()=>refreshFields().catch(e=>showErr($('field-fit-result'),e));
 $('field-learn').onclick=async()=>{const b=$('field-learn');b.disabled=true;
 try{const cols=$('field-columns').value.split(',').map(v=>v.trim());if(cols.length!==2||!cols[0]||!cols[1])throw Error('Provide two column names separated by a comma');
 const res=await call('learn_equation',{source_id:Number($('field-source').value),x_column:cols[0],y_column:cols[1]});
 $('field-fit-result').textContent=`${res.reused?'Reused existing equation':'Equation learned and stored'} from source #${res.source_id}. ${res.equation} | validation RMSE ${res.validation_rmse.toFixed(5)}; mean-only baseline ${res.mean_baseline_rmse.toFixed(5)}. ${res.equation_coordinate_note||""} This is a bounded mathematical fit, not a law of nature.`;
 await refreshFields();$('saved-equations').value=String(res.equation_id);
 }catch(e){showErr($('field-fit-result'),e)}finally{b.disabled=false}};
 $('field-predict').onclick=async()=>{try{const result=await call('predict_equation',{equation_id:Number($('saved-equations').value),x:Number($('field-x').value)});
 $('field-prediction').textContent=result.supported?`Predicted ${result.value.toFixed(6)} for x=${result.x}; observed range ${result.observed_range.join(' to ')}; held-out RMSE ${result.validation_rmse.toFixed(6)}. Source #${result.source_id} SHA ${result.source_sha256.slice(0,16)}... ${result.coordinate_note||""} This is a model estimate.`:`Refused: ${result.reason}. Observed range ${result.observed_range.join(' to ')}.`;
 }catch(e){showErr($('field-prediction'),e)}};
 refreshFields().catch(e=>showErr($('field-fit-result'),e));
}
function initDiagnostics(){let recent=[];
 $('refresh-events').onclick=async()=>{try{const r=await call('events');recent=r.events;$('event-summary').textContent=`${recent.length} most recent events · local only · no raw user prompts stored`;$('event-list').innerHTML=recent.slice().reverse().map(e=>`<div class="trace-row event-line"><small>${esc(e.time)}</small><div><b>${esc(e.event)}</b> <code>${esc(JSON.stringify(Object.fromEntries(Object.entries(e).filter(([k])=>!['schema','time','event'].includes(k)))) )}</code></div></div>`).join('')||'No events yet.';}catch(e){showErr($('event-summary'),e)}};
 $('export-events').onclick=()=>{const blob=new Blob([JSON.stringify(recent,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='operator-atlas-diagnostics.json';a.click();URL.revokeObjectURL(url)};}
function initBench(){
 const historical=S.status?.historical_sunspot_stress;
 if($('v1-real-study')) $('v1-real-study').innerHTML= historical ? [
  metric('AR(2) forecast RMSE', historical.autoregressive_2_rmse.toFixed(2)),
  metric('Historical mean RMSE',historical.historical_mean_rmse.toFixed(2)),
  metric('Atlas 13-term RMSE',historical.equation_13_rmse.toFixed(2)),
  metric('Atlas 16-term RMSE',historical.equation_16_rmse.toFixed(2)),
  metric('Scientific finding','Atlas lost to the simple AR(2) and historical-mean baselines on this test')
 ].join('') : 'Reproduce with ./launch.sh study-real';
 const rq=S.status?.research_v09||{};
 if($('v09-research')){
  const e=rq.source_answers,c=rq.mathematical_coordinates;
  $('v09-research').innerHTML=e&&c?(
   metric('Synthetic questions with cited source',`${e.positive_correct}/${e.positive_total}`)+
   metric('Unsupported questions correctly refused',`${e.negative_abstained}/${e.negative_total}`)+
   metric('Conflicting sources shown separately',e.conflict_reported?'Yes':'No')+
   metric('13-term log-pattern held-out RMSE',Number(c.mean_unseen_rmse_13).toFixed(5))+
   metric('16-term log-pattern held-out RMSE',Number(c.mean_unseen_rmse_16).toFixed(5))+
   metric('Shifted-coordinate numerical error',Number(c.large_coordinate_max_abs_error).toExponential(2))
  ):'No packaged v0.9 research measurements found.';
 }
 $('run-probe').onclick=async()=>{const b=$('run-probe');b.disabled=true;$('probe-results').textContent='Evaluating held-out text against original model…';try{const r=await call('probe',{characters:Number($('probe-chars').value),allowed_increase:Number($('probe-budget').value),objective:$('probe-objective').value,max_slowdown:0.05});$('probe-results').innerHTML=metric('Original validation loss',r.base_validation.cross_entropy.toFixed(5))+metric('Selected validation loss',r.selected_validation.cross_entropy.toFixed(5))+metric('Original sealed test loss',r.base_test.cross_entropy.toFixed(5))+metric('Selected sealed test loss',r.selected_test.cross_entropy.toFixed(5))+metric('Dense CPU ms per character',r.timing.baseline_ms_per_character.toFixed(5))+metric('Selected CPU ms per character',r.timing.selected_ms_per_character.toFixed(5))+metric('Original weight payload',r.original_payload_bytes+' B')+metric('Candidate weight payload',r.selected_payload_bytes+' B')+metric('Selection decision',r.decision.reason+'; candidate '+r.timing.candidate_to_baseline_ratio.toFixed(2)+'× baseline latency')+metric('Applied representations',r.operators.map(x=>x.operator+': '+x.selected.kind+(x.selected.rank?' / rank '+x.selected.rank:'')).join('; '))+metric('Saved experiment',r.run_id); }catch(e){showErr($('probe-results'),e)}finally{b.disabled=false}};
 if(S.status?.model){const a=S.status.model,b=S.status.dense_control;const d=$('model-comparison');if(b){const abs=S.status.ablation_summary;d.innerHTML=(abs?metric('3-seed factorized mean loss',abs.factorized_rank16.mean_validation_loss.toFixed(3))+metric('3-seed dense mean loss',abs.dense_width38.mean_validation_loss.toFixed(3))+metric('Factorized payload',abs.factorized_rank16.mean_payload_bytes+' B')+metric('Dense matched payload',abs.dense_width38.mean_payload_bytes+' B'):'')+metric('Current model loss',Number(a.training.validation_loss).toFixed(3));}}$('benchmark').onclick=async()=>{const b=$('benchmark');b.disabled=true;$('bench-meta').textContent='Running on this machine…';try{const r=await call('experiments',{seed:41});$('bench-meta').textContent=r.platform+' · deterministic seed '+r.seed;
 $('bench-timing').innerHTML=Object.entries(r.timings).map(([k,v])=>metric(k+' median',v.median_ms+' ms · '+v.storage_bytes+' B')).join('');
 $('family-study').innerHTML=(r.equation_family_study?.summary||[]).map(row=>
 metric(`${row.task} · ${row.basis_count} terms`,`${row.mean_test_rmse.toFixed(5)} independent RMSE · ${row.mean_blocked_holdout_rmse.toFixed(5)} blocked RMSE`)).join('');
 $('bench-rows').innerHTML=r.representation.filter(x=>['dense','factorized'].includes(x.operator)&&[null,8,16].includes(x.rank)).slice(0,15).map(x=>metric(`${x.case} · ${x.operator} ${x.rank||''}`,`${(x.relative_output_error*100).toFixed(2)}% err · ${x.payload_bytes} B`)).join('');
 }catch(e){showErr($('bench-meta'),e)}finally{b.disabled=false}};}
window.addEventListener('DOMContentLoaded',async()=>{initTabs();initAnswer();initGenerate();initEquations();initMemory();initFieldLearning();initBench();initDiagnostics();try{await initGraph();initBench();}catch(e){$('server-status').textContent='Engine offline';$('server-status').classList.remove('ok');showErr($('graph-toast'),e)}});
