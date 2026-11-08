"""Fixed validation candidates for comparing intensity teachers, never test data.

The first `generated_count` hypotheses alone participate in deployable ranking.
Reference hypotheses appended afterward are diagnostic only. No decoder or
classifier is updated or regenerated between teacher comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from spectral_mixture import Hypothesis, MixtureObjective


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def save_atomic(value,path):
    path=Path(path);temporary=path.with_suffix('.tmp')
    torch.save(value,temporary);temporary.replace(path)


@torch.no_grad()
def capture(root,checkpoint,cache,device='cuda'):
    from train_spectral_mixture import load_trained_mixture
    from spectral_mixture_joint import pool_from_model,candidate_sequence_scores
    root=Path(root);checkpoint=Path(checkpoint);cache=Path(cache)
    if cache.exists():raise FileExistsError('Fixed ranking cache already exists; do not overwrite its candidates.')
    cfg=json.loads((checkpoint.parent/'config.json').read_text())
    panel=cfg['panel'];data=Path(cfg['settings']['data'])
    validation=set(json.loads((data/'splits.json').read_text())['validation'])
    if len(panel)!=512 or len(set(panel))!=512 or not set(panel)<=validation:
        raise ValueError('Expected the same 512 unique validation spectra, never test spectra.')
    model,objective=load_trained_mixture(root,checkpoint,checkpoint.parent/'source_snapshot',device)
    peaks=np.load(data/'peaks.npy',mmap_mode='r');lengths=np.load(data/'lengths.npy',mmap_mode='r')
    precursors=np.load(data/'precursors.npy',mmap_mode='r')
    wanted=set(panel);labels={}
    with (data/'records.jsonl').open() as stream:
        for i,line in enumerate(stream):
            if i in wanted:labels[i]=json.loads(line)['model_labels']
    if any(sum(len(labels[i])==n for i in panel)!=256 for n in (1,2)):
        raise ValueError('Ranking panel must contain exactly 256 single and 256 double labels.')
    rows=[]
    for offset in range(0,len(panel),16):
        ids=panel[offset:offset+16];width=int(lengths[ids].max())
        x=torch.from_numpy(np.array(peaks[ids,:width],copy=True)).to(device)
        p=torch.from_numpy(np.array(precursors[ids],copy=True)).to(device)
        pools=pool_from_model(model,x,p,beam=4)
        original=model.generate(x,p,beam_size=4)
        memory,mask=model.memories(x);logits=model.classify_memory(memory,mask)
        for j,(index,pool) in enumerate(zip(ids,pools)):
            truth=Hypothesis(tuple(labels[index]))
            # Oracle entries below are used only to inspect physical costs.
            hs=list(dict.fromkeys(pool+[truth]+[Hypothesis((q,)) for q in truth.peptides]))
            seq=candidate_sequence_scores(model,p[j:j+1],memory[j:j+1],mask[j:j+1],hs)
            rows.append(dict(index=index,labels=list(truth.peptides),
                spectrum=x[j:j+1,:int(lengths[index])].cpu(),
                hypotheses=[h.peptides for h in hs],generated_count=len(pool),
                sequence_scores=seq.cpu(),count_logit=logits[j].cpu(),
                original=original[j]['peptides']))
        print(f'Fixed validation candidates: {len(rows)}/{len(panel)}',flush=True)
    save_atomic(dict(format='intensity_ranking_panel_v1',rows=rows,
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        data_sha256=digest(data/'manifest.json'),objective_config=objective_config(objective),
        reconstruction_weight=float(F.softplus(objective.raw_weight)),
        note='Generated candidates only for ranking; appended references are diagnostic only.'),cache)


def objective_config(objective):
    return dict(charges=objective.charges,complexity_penalty=objective.complexity_penalty)


def margin_stats(values):
    return dict(spectra=len(values),true_better=sum(v>1e-6 for v in values),
        ties=sum(abs(v)<=1e-6 for v in values),true_worse=sum(v< -1e-6 for v in values),
        median_advantage=float(np.median(values)) if values else None)


@torch.no_grad()
def evaluate_cached(teacher,cache,device,stopped=lambda:False,heartbeat=None):
    cached=torch.load(str(cache),map_location='cpu',weights_only=False)
    if cached['format']!='intensity_ranking_panel_v1':raise ValueError('Unknown ranking panel.')
    was_training=teacher.training
    # MixtureObjective freezes its teacher; restore trainability afterward.
    parameters=list(teacher.parameters());requires=[p.requires_grad for p in parameters]
    objective=MixtureObjective(teacher,initial_weight=cached['reconstruction_weight'],**cached['objective_config']).to(device).eval()
    counts={str(n):dict(n=0,original=0,candidate_only=0,mixture=0,truth_in_pool=0) for n in (1,2)}
    margins={str(n):[] for n in (1,2)};same_count={str(n):[] for n in (1,2)};pair_margins=[]
    try:
        for position,row in enumerate(cached['rows']):
            if stopped():return None
            if heartbeat is not None and position%32==0:heartbeat(position,len(cached['rows']))
            truth=Hypothesis(tuple(row['labels']));hs=[Hypothesis(tuple(q)) for q in row['hypotheses']]
            n=str(len(truth.peptides));c=counts[n];c['n']+=1
            c['original']+=int(set(row['original'])==set(truth.peptides))
            k=row['generated_count'];c['truth_in_pool']+=int(truth in hs[:k])
            x=row['spectrum'].to(device)
            costs=torch.stack([v['cost'] for v in objective.reconstruction_costs(x,hs)])
            sequence=row['sequence_scores'].to(device);logit=row['count_logit'].to(device)
            if k:
                # Labels and appended hypotheses NEVER enter these two argmaxes.
                base=int(objective.scores(sequence[:k],logit,torch.zeros_like(costs[:k]),hs[:k]).argmax())
                best=int(objective.scores(sequence[:k],logit,costs[:k],hs[:k]).argmax())
                c['candidate_only']+=int(hs[base]==truth);c['mixture']+=int(hs[best]==truth)
                if hs[base]!=truth:
                    difference=float(costs[base]-costs[hs.index(truth)])
                    margins[n].append(difference)
                    if len(hs[base].peptides)==len(truth.peptides):same_count[n].append(difference)
            if n=='2':
                best_single=min(float(costs[hs.index(Hypothesis((p,)))]) for p in truth.peptides)
                pair_margins.append(best_single-float(costs[hs.index(truth)]))
        return dict(counts=counts,score=sum(c['mixture']/max(1,c['n']) for c in counts.values())/2,
            true_vs_wrong={n:margin_stats(v) for n,v in margins.items()},
            same_count_true_vs_wrong={n:margin_stats(v) for n,v in same_count.items()},
            true_pair_vs_best_true_single=margin_stats(pair_margins),
            frozen_checkpoint_sha256=cached['checkpoint_sha256'],
            scope='Fixed validation candidates and sequencer; oracle margins are diagnostics, not inference accuracy.')
    finally:
        for p,flag in zip(parameters,requires):p.requires_grad_(flag)
        teacher.train(was_training)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    capture(args.root.resolve(),args.checkpoint.resolve(),args.cache.resolve())
