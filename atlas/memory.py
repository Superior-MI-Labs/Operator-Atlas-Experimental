"""Immutable source-backed text records. This is retrieval, not parametric model learning."""
import csv,hashlib,io,json,math,sqlite3,re,time
from pathlib import Path

class EvidenceStore:
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as con:
            con.execute('CREATE TABLE IF NOT EXISTS sources (id INTEGER PRIMARY KEY, title TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE, content TEXT NOT NULL, created_at REAL NOT NULL)')
            con.execute('CREATE TABLE IF NOT EXISTS chunks (source_id INTEGER NOT NULL, chunk_id INTEGER NOT NULL, text TEXT NOT NULL, PRIMARY KEY(source_id,chunk_id), FOREIGN KEY(source_id) REFERENCES sources(id))')
            con.execute('''CREATE TABLE IF NOT EXISTS equations (id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL REFERENCES sources(id), x_column TEXT NOT NULL, y_column TEXT NOT NULL, method_version TEXT NOT NULL, model_json TEXT NOT NULL, created_at REAL NOT NULL, UNIQUE(source_id,x_column,y_column,method_version))''')
            columns={row[1] for row in con.execute('PRAGMA table_info(sources)')}
            for name in ('source_name','media_type','original_sha256'):
                if name not in columns:
                    con.execute(f'ALTER TABLE sources ADD COLUMN {name} TEXT')
    def connect(self):
        c=sqlite3.connect(self.path,timeout=10);c.execute('PRAGMA foreign_keys=ON');c.execute('PRAGMA busy_timeout=10000');return c
    def ingest(self,title,text,source_name=None,media_type="text/plain",original_sha256=None):
        title=str(title).strip()[:200]; text=str(text).strip()
        if not title or not (20<=len(text)<=1500000):raise ValueError('Title and 20–1500000 characters required')
        digest=hashlib.sha256(text.encode()).hexdigest()
        with self.connect() as con:
            con.execute('INSERT OR IGNORE INTO sources(title,sha256,content,created_at,source_name,media_type,original_sha256) VALUES(?,?,?,?,?,?,?)',(title,digest,text,time.time(),str(source_name or '')[:220],str(media_type)[:80],original_sha256))
            sid=con.execute('SELECT id FROM sources WHERE sha256=?',(digest,)).fetchone()[0]
            count=con.execute('SELECT count(*) FROM chunks WHERE source_id=?',(sid,)).fetchone()[0]
            if not count:
                words=text.split()
                for idx in range(0,len(words),65):
                    chunk=' '.join(words[idx:idx+65])
                    con.execute('INSERT INTO chunks(source_id,chunk_id,text) VALUES(?,?,?)',(sid,idx//65,chunk))
        return {'source_id':sid,'title':title,'sha256':digest,'chunks':max(1,(len(text.split())+64)//65),'ingestion':'exact source storage and lexical retrieval; no neural training'}
    def search(self,query,limit=5):
        # Search informative words rather than rewarding ubiquitous question words.
        stop={'the','and','for','what','where','when','which','who','why','how','are','was','were','does','have','with','this','that','from','about','into','tell','give','can','you','your','please','explain','information','document','source'}
        terms=[t for t in dict.fromkeys(re.findall(r'[a-z0-9]{3,}',str(query).lower())) if t not in stop][:12]
        if not terms:return []
        results=[]
        with self.connect() as con:
            for sid,idx,chunk,title,digest in con.execute('SELECT c.source_id,c.chunk_id,c.text,s.title,s.sha256 FROM chunks c JOIN sources s ON s.id=c.source_id'):
                txt=chunk.lower();words=re.findall(r'[a-z0-9]{3,}',txt);score=sum(words.count(term) for term in terms)
                if score and (len(terms)==1 or sum(t in words for t in terms)>=2):results.append({'source_id':sid,'chunk_id':idx,'title':title,'text':chunk,'sha256':digest,'term_matches':score})
        return sorted(results,key=lambda r:(-r['term_matches'],r['source_id'],r['chunk_id']))[:max(1,min(int(limit),20))]
    def stats(self):
        with self.connect() as con:return {'source_count':con.execute('SELECT count(*) FROM sources').fetchone()[0], 'chunk_count':con.execute('SELECT count(*) FROM chunks').fetchone()[0], 'equation_count':con.execute('SELECT count(*) FROM equations').fetchone()[0]}

    def list_sources(self,limit=40):
        """Stable source index; content is never exposed by the listing."""
        with self.connect() as con:
            rows=con.execute('SELECT id,title,sha256,created_at,source_name,media_type,original_sha256,length(content) FROM sources ORDER BY id DESC LIMIT ?', (max(1,min(int(limit),200)),)).fetchall()
        return [{'source_id':r[0],'title':r[1],'sha256':r[2],'created_at':r[3],
                 'source_name':r[4],'media_type':r[5],'original_sha256':r[6],'characters':r[7]} for r in rows]

    @staticmethod
    def _evidence_terms(text):
        """Conservative, transparent lexical retrieval; NOT semantic understanding."""
        stop={'the','and','for','what','where','when','which','who','why','how','are','was','were','does',
              'have','has','had','with','this','that','from','about','into','tell','give','can','you',
              'your','please','explain','information','document','source','is','it','of','to','in',
              'a','an','on','did','do','its','me','there','any','by'}
        def normalize(word):
            if len(word)>5 and word.endswith('ing'):
                word=word[:-3]
            elif len(word)>4 and word.endswith('ed'):
                word=word[:-2]
            elif len(word)>4 and word.endswith('s'):
                word=word[:-1]
            if word in ('stored','stor','store','storage','located','locat','location','placed','plac','place','position'):
                return 'place'
            if word=='occurr':return 'occur'
            return word
        return set(normalize(w) for w in re.findall(r'[a-z0-9]{3,}',str(text).lower()) if w not in stop)

    def answer_sentences(self,question,limit=3):
        """Extract complete matching sentences from canonical source content.

        Evidence, not factual adjudication: a source may be false and lexical
        overlap does not establish entailment. Full coverage is deliberately
        required for short questions to avoid accidental confident responses.
        """
        terms=self._evidence_terms(question)
        if len(terms)<2:return []
        rows=[]
        with self.connect() as con:
            source_rows=con.execute('SELECT id,title,sha256,content FROM sources ORDER BY id').fetchall()
        for sid,title,digest,content in source_rows:
            for i,sentence in enumerate(re.split(r'(?<=[.!?])\s+|\n+',content)):
                sentence=sentence.strip().lstrip('-* ').strip()
                if not (12<=len(sentence)<=1100):continue
                present=self._evidence_terms(sentence)
                matches=len(terms & present)
                coverage=matches/len(terms)
                if matches<2 or coverage < (1.0 if len(terms)<=3 else 0.80):continue
                # Do not infer the answer to a question asking about a property
                # that is absent from an otherwise related source passage.
                rows.append({'source_id':sid,'chunk_id':i,'title':title,
                             'text':sentence,'sha256':digest,'term_matches':matches,
                             'term_coverage':round(coverage,4),'method':'lexical_sentence'})
        rows.sort(key=lambda r:(-r['term_coverage'],-r['term_matches'],len(r['text']),r['source_id'],r['chunk_id']))
        # Prefer independent sources over several near-identical sentences.
        selected=[];source_ids=set()
        for row in rows:
            if row['source_id'] not in source_ids:
                selected.append(row);source_ids.add(row['source_id'])
            if len(selected)>=max(1,min(int(limit),20)):break
        return selected

    def extractive_answer(self,question,limit=3):
        """Grounded passages, not generated claims. Abstains on no lexical evidence."""
        passages=self.answer_sentences(question,limit)
        return {'question':str(question)[:500], 'found':bool(passages), 'passages':passages,
                'answer': 'Potentially relevant source passages are shown below; verify their meaning against the original document.' if passages else 'No matching evidence. This system cannot answer from the available documents.',
                'method':'Conservative lexical sentence extraction. No semantic entailment or truth verification; source text may be incorrect.'}

    def learn_equation(self, source_id, x_column, y_column):
        """Discover a bounded equation from an imported numerical CSV source.

        Source text is authoritative; derived equation is immutable and identified
        by source, columns, and method version. No neural weights are changed.
        """
        from .discovery import discover
        source_id=int(source_id)
        x_column=str(x_column).strip(); y_column=str(y_column).strip()
        if not x_column or not y_column or x_column==y_column:
            raise ValueError('Select two distinct CSV column names')
        if len(x_column)>80 or len(y_column)>80:
            raise ValueError('Column name too long')
        version='coordinate-bic-v3'
        with self.connect() as con:
            row=con.execute('SELECT content,media_type,sha256 FROM sources WHERE id=?',(source_id,)).fetchone()
            if not row:raise ValueError('Unknown source ID')
            content,media_type,digest=row
            if media_type!='text/csv':raise ValueError('Equation learning requires a CSV source')
            existing=con.execute('SELECT id,model_json FROM equations WHERE source_id=? AND x_column=? AND y_column=? AND method_version=?', (source_id,x_column,y_column,version)).fetchone()
            if existing:return {'equation_id':existing[0],'source_id':source_id,'source_sha256':digest,**json.loads(existing[1]),'reused':True}
        reader=csv.DictReader(io.StringIO(content))
        if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)):
            raise ValueError('CSV must have unique named columns')
        if x_column not in reader.fieldnames or y_column not in reader.fieldnames:
            raise ValueError('Requested CSV columns do not exist')
        points=[]
        for row in reader:
            if len(points)>=10000:raise ValueError('CSV equation learning supports at most 10000 observations')
            try:x=float(row[x_column]);y=float(row[y_column])
            except (TypeError,ValueError,KeyError):raise ValueError('Selected columns must contain only numerical observations')
            if not(math.isfinite(x) and math.isfinite(y)):
                raise ValueError('Selected columns must be finite')
            points.append((x,y))
        if not 12<=len(points)<=10000:raise ValueError('Equation learning requires 12–10000 rows')
        if len(set(x for x,y in points))!=len(points):
            raise ValueError('Duplicate x coordinates are ambiguous; combine repeated observations first')
        fitted=discover(points)
        if not fitted['beats_mean_baseline'] or fitted['validation_rmse']>=.9*fitted['mean_baseline_rmse']:
            raise ValueError('No useful equation: held-out error did not beat baseline by at least 10%; retain explicit observations instead')
        fitted={**fitted,'x_min':min(x for x,y in points),'x_max':max(x for x,y in points),'x_column':x_column,'y_column':y_column,
                'method_version':version,'training_notice':'Fitted an equation from CSV values; no neural model weights changed. In-domain interpolation is not guaranteed truth.'}
        payload=json.dumps(fitted,allow_nan=False,sort_keys=True)
        with self.connect() as con:
            con.execute('INSERT OR IGNORE INTO equations(source_id,x_column,y_column,method_version,model_json,created_at) VALUES(?,?,?,?,?,?)',
                        (source_id,x_column,y_column,version,payload,time.time()))
            found=con.execute('SELECT id,model_json FROM equations WHERE source_id=? AND x_column=? AND y_column=? AND method_version=?',
                              (source_id,x_column,y_column,version)).fetchone()
        return {'equation_id':found[0],'source_id':source_id,'source_sha256':digest,**json.loads(found[1]),'reused':False}

    def list_equations(self,limit=40):
        with self.connect() as con:
            rows=con.execute('SELECT e.id,e.source_id,s.title,s.sha256,e.model_json FROM equations e JOIN sources s ON s.id=e.source_id ORDER BY e.id DESC LIMIT ?',
                             (max(1,min(int(limit),200)),)).fetchall()
        return [{'equation_id':i,'source_id':sid,'title':title,'source_sha256':digest,**json.loads(raw)} for i,sid,title,digest,raw in rows]

    def predict_equation(self,equation_id,x):
        from .discovery import predict
        try:x=float(x)
        except (ValueError,TypeError):raise ValueError('Prediction coordinate must be numeric')
        if not math.isfinite(x):raise ValueError('Prediction coordinate must be finite')
        with self.connect() as con:
            row=con.execute('SELECT e.source_id,s.title,s.sha256,e.model_json FROM equations e JOIN sources s ON s.id=e.source_id WHERE e.id=?',(int(equation_id),)).fetchone()
        if not row:raise ValueError('Unknown equation ID')
        sid,title,digest,raw=row;model=json.loads(raw)
        if not model['x_min']<=x<=model['x_max']:
            return {'equation_id':int(equation_id),'source_id':sid,'title':title,'source_sha256':digest,
                    'x':x,'supported':False,'value':None,'reason':'Outside observed coordinate domain; extrapolation refused',
                    'observed_range':[model['x_min'],model['x_max']]}
        return {'equation_id':int(equation_id),'source_id':sid,'title':title,'source_sha256':digest,
                'x':x,'supported':True,'value':float(predict(model,[x])[0]),'equation':model['equation'],
                'validation_rmse':model['validation_rmse'],'coordinate_note':model.get('equation_coordinate_note','Legacy raw-coordinate fit'), 'observed_range':[model['x_min'],model['x_max']],
                'warning':'Prediction from a fitted restricted equation, not an independently verified physical law'}

    def compose_equations(self, equation_ids, x):
        """Execute an explicit chain of already-qualified source-bound equations.

        A composition is an ephemeral computation, not another copy of an
        equation or a persistent second owner. Values and source hashes are
        inspected at every stage; no dimensional/unit compatibility is implied.
        """
        if not isinstance(equation_ids, (list, tuple)) or not 2 <= len(equation_ids) <= 6:
            raise ValueError('Compose between 2 and 6 equation IDs')
        try:
            ids=[int(v) for v in equation_ids if isinstance(v,int) and not isinstance(v,bool) or isinstance(v,str) and v.isascii() and v.isdecimal()]
        except (TypeError,ValueError) as exc:
            raise ValueError('Equation IDs must be positive integers') from exc
        if len(ids)!=len(equation_ids) or any(v < 1 for v in ids):
            raise ValueError('Equation IDs must be positive integers')
        if len(set(ids)) != len(ids):
            raise ValueError('Repeated equation IDs require explicit separate evaluation')
        try:value=float(x)
        except (TypeError,ValueError) as exc:raise ValueError('Input coordinate must be finite') from exc
        if not math.isfinite(value):raise ValueError('Input coordinate must be finite')
        steps=[]
        for eid in ids:
            prediction=self.predict_equation(eid,value)
            steps.append(prediction)
            if not prediction['supported']:
                return {'schema':'atlas-composition-v1','supported':False,'value':None,
                        'initial_x':float(x),'equation_ids':ids,'steps':steps,
                        'reason':f"Stopped at equation #{eid}: {prediction['reason']}",
                        'disclosure':'No unverified continuation outside an observed domain.'}
            value=prediction['value']
            if not math.isfinite(value):
                raise ValueError('Equation produced non-finite numerical output')
        return {'schema':'atlas-composition-v1','supported':True,'value':value,
                'initial_x':float(x),'equation_ids':ids,'steps':steps,
                'source_ids':[step['source_id'] for step in steps],
                'disclosure':'Numerically evaluated sequence of source-linked empirical fits. Input/output units and combined uncertainty are not validated; individual RMSE values cannot simply be added.'}

    def migrate_from(self,legacy_path):
        """Explicit offline SQLite backup from prior Atlas version, preserving source IDs.

        Refuse nonempty destination; caller should stop both local servers first.
        """
        old=Path(legacy_path).expanduser().resolve()
        if not old.is_file() or old==self.path.resolve():
            raise ValueError('Specify an existing different SQLite source database')
        with sqlite3.connect(f'file:{old}?mode=ro',uri=True) as source:
            tables={row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'sources','chunks'}.issubset(tables):
                raise ValueError('Not a compatible Operator Atlas evidence database')
            with self.connect() as dest:
                if dest.execute('SELECT count(*) FROM sources').fetchone()[0]:
                    raise ValueError('Destination already contains sources; migration would overwrite them')
                source.backup(dest)
        # Evolve copied v0.2 schema through the existing migration authority.
        self.__init__(self.path)
        return self.stats()
