"""Joint sequencing integration for spectral_mixture; imports no Casanovo.

Call pool_from_model without references, then optionally insert the reference
as a training positive. Inference uses the exact same scorer on generated
candidate sets only. No oracle selection or true precursor mass enters it.
"""
import torch
from torch.nn import functional as F
from spectral_mixture import Hypothesis,parse_peptide


@torch.no_grad()
def pool_from_model(model,spectra,precursors,beam=4,max_tokens=96):
    if beam<2:raise ValueError('Candidate collection requires beam >= 2.')
    from dual_slot import _SlotView
    from evaluate_holdout import generate_beam
    # A no-grad autocast context must not cache detached decoder weights for a
    # later gradient-bearing forward in an outer mixed-precision scope.
    was_training=model.training
    try:
        model.eval()
        with torch.autocast(device_type=spectra.device.type,enabled=False):
            memory,mask=model.memories(spectra.float())
            outputs=[generate_beam(_SlotView(model,memory,mask,slot),spectra.float(),precursors.float(),
                beam_size=beam,return_candidates=True,max_tokens=max_tokens) for slot in (0,1)]
        pools=[]
        for first,second in zip(*outputs):
            slots=[]
            for result in (first,second):
                candidates=[]
                for row in result.get('candidates',[]):
                    if not row['ended_with_eos'] or len(row['peptides'])!=1:continue
                    try:
                        sequence=row['peptides'][0].replace('I','L');parse_peptide(sequence)
                    except ValueError:continue
                    if sequence not in candidates:candidates.append(sequence)
                slots.append(candidates)
            pairs=[Hypothesis((a,b)) for a in slots[0] for b in slots[1] if a!=b]
            singles=[Hypothesis((a,)) for a in slots[0]+slots[1]]
            pools.append(list(dict.fromkeys(singles+pairs)))
        return pools
    finally:
        model.train(was_training)


def add_training_positive(hypotheses,reference):
    """Supervised candidate enrichment, NEVER called by predict_mixtures.

    Include both true singles as alternatives to a true pair. This gives the
    count prior a direct pair-versus-one training signal even with a poor beam.
    """
    result=list(hypotheses)
    for h in [Hypothesis(tuple(reference))]+[Hypothesis((p,)) for p in reference]:
        if h not in result:result.append(h)
    if len(result)<2:
        raise ValueError('No negative hypothesis; generate candidates before training.')
    return result


def candidate_sequence_scores(model,precursor,memory,mask,hypotheses):
    peptides=list(dict.fromkeys(p for h in hypotheses for p in h.peptides))
    p=precursor.expand(len(peptides),-1)
    m=memory.expand(len(peptides),-1,-1);k=mask.expand(len(peptides),-1)
    # Teacher forcing here scores our own *candidate*, including at inference.
    # It does not supply a ground-truth prefix to candidate generation.
    first=model.slot_loss(peptides,p,m,k,0)
    second=model.slot_loss(peptides,p,m,k,1)
    index={x:i for i,x in enumerate(peptides)}
    scores=[]
    for h in hypotheses:
        a=index[h.peptides[0]]
        if len(h.peptides)==1:scores.append(-first[a])
        else:
            b=index[h.peptides[1]]
            scores.append(-torch.minimum(first[a]+second[b],first[b]+second[a])/2)
    return torch.stack(scores)


def mixture_training_loss(model,objective,spectra,precursors,labels,pools):
    """Gradient-bearing candidate ranking + expected reconstruction risk.

    Main model.loss_components (sequence CE/BCE/evidence/PIT) is still required
    in the caller. This term can improve candidate scores, not invent sequences
    missing from the generator's beam. Explicit positives are training only.
    """
    if not (len(pools)==len(labels)==len(spectra)):raise ValueError('Batch size mismatch.')
    memory,mask=model.memories(spectra)
    logits=model.classify_memory(memory,mask)
    losses=[];statistics=[]
    for i,(pool,truth) in enumerate(zip(pools,labels)):
        if not pool and len(truth)==1:
            # Main sequencing/BCE still trains on this spectrum. An optional
            # rank loss has no negative candidate here, so it supplies zero
            # additional gradient rather than inventing a label-based decoy.
            losses.append(logits[i]*0)
            statistics.append(dict(candidates=0,positive_was_generated=False,
                                   rank=0.,risk=0.,no_negative_candidate=True))
            continue
        hypotheses=add_training_positive(pool,truth)
        fits=objective.reconstruction_costs(spectra[i:i+1],hypotheses)
        costs=torch.stack([f['cost'] for f in fits])
        sequence=candidate_sequence_scores(model,precursors[i:i+1],memory[i:i+1],mask[i:i+1],hypotheses)
        loss,stats=objective.supervised_loss(sequence,logits[i],costs,hypotheses,truth)
        losses.append(loss)
        statistics.append(dict(candidates=len(pool),positive_was_generated=Hypothesis(tuple(truth)) in pool,
            rank=float(stats['rank']),risk=float(stats['risk'])))
    return torch.stack(losses),statistics


@torch.no_grad()
def predict_mixtures(model,objective,spectra,precursors,beam=4):
    """Deployment path: intentionally has no reference/label argument."""
    model.eval();objective.eval()
    pools=pool_from_model(model,spectra,precursors,beam)
    memory,mask=model.memories(spectra)
    logits=model.classify_memory(memory,mask)
    result=[]
    for i,pool in enumerate(pools):
        if not pool:
            result.append(dict(peptides=[],truncated=True,reason='no_completed_supported_candidate'))
            continue
        fits=objective.reconstruction_costs(spectra[i:i+1],pool)
        costs=torch.stack([f['cost'] for f in fits])
        sequence=candidate_sequence_scores(model,precursors[i:i+1],memory[i:i+1],mask[i:i+1],pool)
        scores=objective.scores(sequence,logits[i],costs,pool)
        best=int(scores.argmax());fit=fits[best]
        candidate_only=int(objective.scores(sequence,logits[i],torch.zeros_like(costs),pool).argmax())
        result.append(dict(peptides=list(pool[best].peptides),truncated=False,
            candidate_only_peptides=list(pool[candidate_only].peptides),
            chimeric_probability=float(logits[i].sigmoid()),predicted_count=len(pool[best].peptides),
            mixture_score=float(scores[best]),reconstruction_cost=float(costs[best]),
            candidate_charges=fit['charges'],contributions=fit['coefficients'].cpu().tolist(),
            candidates=len(pool),routing='sequence_count_prior_plus_frozen_intensity_fit'))
    return result
