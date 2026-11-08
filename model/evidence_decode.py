"""Re-rank free Casanovo slot beams using measured MS2 and selected precursor.

The input API deliberately accepts no reference labels or searched precursors.
All comparisons use the original token vocabulary, including terminal PTMs.
"""
from itertools import product
import math

import numpy as np
import torch

from dual_slot import _SlotView
from evaluate_holdout import generate_beam
from route import apply_route


WATER = 18.010565
PROTON = 1.007276466621
ISOTOPE = 1.00335483507


def candidate_evidence(model, tokens, mz, power, observed):
    """Capped isotope-aware mass bonus and fragment evidence for one candidate."""
    pieces = [model.decoder._idx2aa[t] for t in tokens if t != model.stop]
    if model.decoder.reverse:
        pieces.reverse()
    masses = model.decoder._peptide_mass.masses
    residue = [masses[s] for s in pieces if s[0].isalpha()]
    terminal = sum(masses[s] for s in pieces if s[0] in '+-')
    total = sum(residue) + terminal + WATER
    err = min(abs(observed - total - k * ISOTOPE) for k in (-1, 0, 1, 2))
    tolerance = max(.02, total * 20e-6)
    fit = math.exp(-.5 * (err / tolerance) ** 2)
    b = np.cumsum(residue)[:-1] + terminal
    y = total - b
    ions = np.stack([b + PROTON, y + PROTON, b/2 + PROTON, y/2 + PROTON], axis=1)
    # Mark both sides of the insertion point. Coverage counts a cleavage once,
    # even if several charge states match. Intensities are counted once per peak.
    right = np.searchsorted(mz, ions)
    matched_peaks = np.zeros(len(power), bool)
    ion_seen = np.zeros(ions.shape, bool)
    for idx in (np.clip(right-1, 0, len(mz)-1), np.clip(right, 0, len(mz)-1)):
        hit = np.abs(mz[idx] - ions) <= np.maximum(.02, 20e-6 * ions)
        ion_seen |= hit
        matched_peaks[idx[hit]] = True
    coverage = float(ion_seen.any(axis=1).mean()) if len(b) else 0.
    return fit, coverage, matched_peaks


FEATURES = (
    'double', 'count_agreement', 'average_beam', 'weak_beam', 'strong_beam',
    'precursor_fit', 'average_ladder', 'weak_ladder', 'strong_ladder',
    'explained_power', 'weak_unique_power', 'strong_unique_power',
    'beam_times_double', 'mass_times_double', 'ladder_times_double',
)


def rank_spectrum(model, spectrum, observed, p, slots, threshold, policy=None,
                  return_features=False):
    peaks = spectrum[spectrum[:, 1] > 0]
    peaks = peaks[np.argsort(peaks[:, 0])]
    mz, power = peaks[:, 0], np.square(peaks[:, 1])
    power = power / max(1e-12, power.sum())
    cache = {}
    pools = []
    for slot in slots:
        pool = []
        for c in slot.get('candidates', []):
            peptide = c['peptides'][0]
            if peptide not in cache:
                cache[peptide] = candidate_evidence(model, c['raw_tokens'], mz, power, observed)
            fit, coverage, matched = cache[peptide]
            pool.append((peptide, c['beam_score'], fit, coverage, matched))
        pools.append(pool)

    hypotheses = []
    log_odds = math.log(max(1e-6, p)) - math.log(max(1e-6, 1-p))
    for a in pools[0]:
        explained = float(power[a[4]].sum())
        score = a[1] + .65*a[2] + .4*a[3] + .3*explained
        features = [0., -log_odds, a[1], a[1], a[1], a[2],
                    a[3], a[3], a[3], explained, explained, explained, 0., 0., 0.]
        hypotheses.append((1, (a[0],), float(score), a[1], features))
    for a, b in product(*pools):
        if a[0] == b[0]:
            continue
        # One measured precursor can support either peptide, never both by fiat.
        fit = max(a[2], b[2])
        coverage = (a[3] + b[3]) / 2
        explained = power[a[4] | b[4]].sum()
        unique, stronger = sorted((power[a[4] & ~b[4]].sum(), power[b[4] & ~a[4]].sum()))
        beam = (a[1] + b[1]) / 2
        score = beam + .65*fit + .4*coverage + .3*explained + .4*unique
        features = [1., log_odds, beam, min(a[1], b[1]), max(a[1], b[1]),
                    fit, coverage, min(a[3], b[3]), max(a[3], b[3]),
                    explained, unique, stronger, beam, fit, coverage]
        features = [float(v) for v in features]
        hypotheses.append((2, (a[0], b[0]), float(score), beam, features))

    count = 1 + int(p >= threshold)
    valid = [h for h in hypotheses if h[0] == count]
    if valid:
        reranked = list(max(valid, key=lambda h: h[2])[1])
        baseline = list(max(valid, key=lambda h: h[3])[1])
    else:
        baseline = list(dict.fromkeys(sum((s['peptides'] for s in slots[:count]), [])))
        reranked = baseline.copy()
    out = dict(peptides=reranked, beam_peptides=baseline,
               predicted_count=count, chimeric_probability=float(p),
               candidate_sets=[list(h[1]) for h in hypotheses],
               count_constraint_satisfied=len(reranked) == count)
    if return_features or policy is not None:
        # Canonicalize AB/BA as one set for training/re-ranking, using the
        # most probable assignment, just as PIT does in training.
        sets = {}
        for h in hypotheses:
            key = tuple(sorted(h[1]))
            if key not in sets or h[3] > sets[key][3]:
                sets[key] = h
        keys = list(sets)
        features = np.array([sets[k][4] for k in keys], dtype=float)
        if return_features:
            out['feature_sets'] = [list(k) for k in keys]
            out['features'] = features.tolist()
        if policy is not None and keys:
            if tuple(policy['features']) != FEATURES:
                raise ValueError('Ranker feature schema mismatch')
            out['heuristic_peptides'] = reranked
            top = int(np.argmax(features @ np.asarray(policy['weights'])))
            out['peptides'] = list(keys[top])
            out['predicted_count'] = len(keys[top])
            out['count_constraint_satisfied'] = True
    return out


@torch.no_grad()
def evidence_generate(model, spectra, precursors, beam_size=4, threshold=None,
                      max_tokens=120, min_aa=6, diagnostics=False, policy=None,
                      return_features=False):
    """Decode one observed spectrum to an unordered set, retaining baseline for A/B."""
    if beam_size < 2:
        raise ValueError('Reranking requires beam_size >= 2.')
    model.eval()
    cutoff = model.threshold if threshold is None else threshold
    if not 0 < cutoff < 1:
        raise ValueError('Invalid classifier threshold.')
    memory, mask = model.memories(spectra)
    probabilities = model.classify_memory(memory, mask).sigmoid().cpu().tolist()
    chosen = [i for i, p in enumerate(probabilities)
              if p >= cutoff or diagnostics or policy is not None or return_features]
    first = generate_beam(_SlotView(model, memory, mask, 0), spectra, precursors,
                          beam_size=beam_size, max_tokens=max_tokens, min_aa=min_aa,
                          return_candidates=True)
    second = {}
    if chosen:
        ids = torch.as_tensor(chosen, device=spectra.device)
        results = generate_beam(_SlotView(model, memory[ids], mask[ids], 1),
                                spectra[ids], precursors[ids], beam_size=beam_size,
                                max_tokens=max_tokens, min_aa=min_aa, return_candidates=True)
        second = dict(zip(chosen, results))
    rows = []
    for i, (x, precursor, p) in enumerate(zip(spectra.float().cpu().numpy(),
                                               precursors.float().cpu().numpy(), probabilities)):
        slots = [first[i], second.get(i, {'peptides': []})]
        row = rank_spectrum(model, x, float(precursor[0]), p, slots, cutoff,
                            policy=policy, return_features=return_features)
        if not diagnostics:
            row.pop('candidate_sets')
        rows.append(apply_route(row, policy) if policy is not None else row)
    return rows
