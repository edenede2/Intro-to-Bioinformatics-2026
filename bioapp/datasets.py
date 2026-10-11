"""All fixtures are original synthetic teaching examples, released under MIT."""
import numpy as np
from pathlib import Path

SEARCH_DB='''>motif_exact
MKTACDEFGHIKLMNPQ
>motif_variant
MKTACDEYGHIKLMNPQ
>motif_remote
MKTACDDFGHLKLMNPQ
>unrelated
WWWWGGGGPPPPSSSS
>short_fragment
ACDEFG
'''
MSA='''>sequence_1
ACDEFGHIK
>sequence_2
ACDFGHIK
>sequence_3
ACDEYGHVK
>sequence_4
ACDEFGIK
'''


def backbone(n=24):
    t=np.arange(n,dtype=float)
    # C-alpha-like helix geometry, not an experimentally observed molecule.
    return np.column_stack([2.3*np.cos(t*1.75),2.3*np.sin(t*1.75),1.5*t])


def prediction_example():
    reference=backbone(30); rng=np.random.default_rng(19)
    uncertainty=np.r_[np.full(10,.15),np.linspace(.2,1.8,10),np.full(10,2.2)]
    predicted=reference+rng.normal(size=reference.shape)*uncertainty[:,None]
    confidence=np.clip(98-uncertainty*24,0,100)
    # Confidence-like teaching matrix, independent of measured coordinate error.
    pae=1+np.maximum.outer(uncertainty,uncertainty)*3
    pae[:10,20:]+=8; pae[20:,:10]+=8
    return reference,predicted,confidence,pae


def pdb_text(coords):
    return '\n'.join(f'ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C  ' for i,(x,y,z) in enumerate(coords,1))+'\nEND\n'
