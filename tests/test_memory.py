from atlas.memory import EvidenceStore

def test_source_preservation_dedup_and_citation(tmp_path):
    m=EvidenceStore(tmp_path/'evidence.db')
    a=m.ingest('Initial note','The experiment involves mathematical operators and compact neural matrix products.')
    b=m.ingest('Duplicate','The experiment involves mathematical operators and compact neural matrix products.')
    assert a['source_id']==b['source_id']
    assert m.stats()['source_count']==1
    hits=m.search('mathematical operators')
    assert hits[0]['source_id']==a['source_id']
    assert hits[0]['sha256']==a['sha256']
    assert m.search('definitelymissingword')==[]
