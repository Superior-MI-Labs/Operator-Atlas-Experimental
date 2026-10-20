"""Reproducible decoding sensitivity measurements for the shipped tiny model.

These are character statistics, not measures of factual correctness or fluency.
All prompts come from shipped public test strings; never capture private queries.
"""
import math
from pathlib import Path
from .model import CharacterModel


def evaluate_language(root):
    root=Path(root)
    model=CharacterModel(root/'models'/'starter')
    corpus=(root/'data'/'corpus.txt').read_text(encoding='utf-8')
    validation=corpus[int(len(corpus)*.88):]
    reference=model.score_text(validation[:800])
    variants=[]
    for temperature in (.25,.75,1.3):
        outputs=[model.generate('An atlas',96,temperature,seed)['response'] for seed in (3,7,11)]
        sequences=''.join(outputs)
        triples=[sequences[i:i+3] for i in range(max(0,len(sequences)-2))]
        variants.append({'temperature':temperature,'seeds':3,'generated_characters':len(sequences),
                         'unique_character_count':len(set(sequences)),
                         'distinct_trigram_fraction':round(len(set(triples))/len(triples),5) if triples else None,
                         'space_fraction':round(sequences.count(' ')/len(sequences),5)})
    assert len({row['unique_character_count'] for row in variants})>1 or len({row['distinct_trigram_fraction'] for row in variants})>1
    return {'schema':'atlas-language-probe-v1', 'checkpoint':model.manifest['checkpoint_id'],
            'held_out_corpus_characters':reference['characters'],
            'held_out_corpus_cross_entropy':round(reference['cross_entropy'],6),
            'held_out_corpus_perplexity':round(math.exp(reference['cross_entropy']),5),
            'temperature_sensitivity':variants,
            'limits':'Character statistics on narrow authored corpus, not validated language understanding. Temperature only changes sampling.'}
