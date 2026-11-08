"""Explicit user-run evaluation. Defaults to validation, never tunes on test."""
import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
import time

os.environ['MPLBACKEND'] = 'Agg'
import torch

from model import ChimeraSequenceModel
from training_common import PreparedData, digest, evaluate_sequences, write_json
from workflow import FORMAT, evaluate_components


@torch.no_grad()
def generate_beam(model, spectra, precursors, beam_size=4, max_tokens=120,
                  min_aa=6, length_penalty=1.0, routing=None, threshold=None,
                  return_candidates=False):
    """Decode without references; beam_size=1 preserves the original generator.

    Completed and active candidates are kept separately. Scores are sums of
    model log probabilities (plus the optional soft count prior), divided by
    the number of generated tokens ** length_penalty, including SEP and EOS.
    Hard masks do not renormalize the model's probabilities. Duplicate peptide
    pairs are retained only as a fallback when no distinct completion exists.
    """
    if not isinstance(beam_size, int) or beam_size < 1:
        raise ValueError('beam_size must be a positive integer.')
    if not math.isfinite(length_penalty) or length_penalty < 0:
        raise ValueError('length_penalty must be finite and nonnegative.')
    if max_tokens < 1 or min_aa < 1:
        raise ValueError('Token budget and minimum peptide length must be positive.')
    if getattr(model, 'architecture', None) in ('dual_slot_v1', 'dual_slot_v2', 'dual_slot_evidence', 'dual_slot_separation'):
        return model.generate(spectra, precursors, beam_size=beam_size,
                              max_tokens=max_tokens, min_aa=min_aa, length_penalty=length_penalty,
                              routing=routing, threshold=threshold)
    if beam_size == 1:
        return model.generate(spectra, precursors, max_tokens=max_tokens,
                              min_aa=min_aa, routing=routing, threshold=threshold)
    model.eval()
    mode = model.routing if routing is None else routing
    cutoff = model.threshold if threshold is None else threshold
    if mode not in ('unconstrained', 'soft', 'hard') or not 0 < cutoff < 1:
        raise ValueError('Invalid routing mode or threshold.')
    memory, mask = model.memories(spectra)
    probabilities = model.classify_memory(memory, mask).sigmoid().cpu().tolist()
    counts = [1 + int(p >= cutoff) for p in probabilities]
    n = len(probabilities)
    active = [[([], 0.0)] for _ in range(n)]
    completed, duplicates = [[] for _ in range(n)], [[] for _ in range(n)]
    last_active = [[] for _ in range(n)]
    terminal_ids = set(model.nterminal_ids)

    def normalized(candidate):
        tokens, score = candidate
        return score / max(1, len(tokens)) ** length_penalty

    def peptide_parts(tokens):
        parts, current = [], []
        for token in tokens:
            if token in (model.stop, model.separator):
                if current:
                    pieces = [model.decoder._idx2aa[t] for t in current]
                    parts.append(''.join(reversed(pieces) if model.decoder.reverse else pieces))
                current = []
            else:
                current.append(token)
        # Match the original generator: unfinished trailing segments are omitted.
        return parts

    for _ in range(max_tokens):
        flat = [(i, tokens, score) for i, beams in enumerate(active)
                for tokens, score in beams]
        if not flat:
            break
        owner = torch.tensor([item[0] for item in flat], device=spectra.device)
        prefixes = [item[1] for item in flat]
        sequences = ([torch.tensor(p, dtype=torch.long, device=spectra.device)
                      for p in prefixes] if prefixes[0] else None)
        logits, _ = model.decoder(sequences, precursors.index_select(0, owner),
                                  memory.index_select(0, owner), mask.index_select(0, owner))
        positions = torch.tensor([len(p) for p in prefixes], device=logits.device)
        rows = torch.arange(len(flat), device=logits.device)
        log_probs = logits[rows, positions].log_softmax(-1).cpu()
        proposals = [[] for _ in range(n)]
        for row, (i, prefix, previous_score) in enumerate(flat):
            scores = log_probs[row].clone()
            scores[0] = -torch.inf
            has_separator = model.separator in prefix
            segment = prefix[prefix.index(model.separator) + 1:] if has_separator else prefix
            aa_count = sum(t not in terminal_ids for t in segment)
            if has_separator:
                scores[model.separator] = -torch.inf
            if aa_count < min_aa:
                scores[model.stop] = scores[model.separator] = -torch.inf
                scores[model.nterminal_ids] = -torch.inf
            if segment and segment[-1] in terminal_ids and model.decoder.reverse:
                allowed = [model.stop] + ([] if has_separator else [model.separator])
                keep = torch.full_like(scores, -torch.inf)
                keep[allowed] = scores[allowed]
                scores = keep
            if mode == 'hard':
                if counts[i] == 1:
                    scores[model.separator] = -torch.inf
                elif not has_separator:
                    scores[model.stop] = -torch.inf
            elif mode == 'soft' and not has_separator and aa_count >= min_aa:
                p = min(1 - 1e-6, max(1e-6, probabilities[i]))
                scores[model.separator] += model.routing_strength * math.log(p)
                scores[model.stop] += model.routing_strength * math.log1p(-p)
            if not torch.isfinite(scores).any():
                raise RuntimeError('All decoding choices were masked; inspect the count constraints.')
            if torch.isfinite(scores[model.stop]):
                candidate = (prefix + [model.stop], previous_score + float(scores[model.stop]))
                parts = peptide_parts(candidate[0])
                bucket = duplicates[i] if len(parts) != len(set(parts)) else completed[i]
                bucket.append(candidate)
            scores[model.stop] = -torch.inf
            values, tokens = scores.topk(min(beam_size, len(scores)))
            for value, token in zip(values.tolist(), tokens.tolist()):
                if math.isfinite(value):
                    proposals[i].append((prefix + [token], previous_score + value))
        for i in range(n):
            if not active[i]:
                continue
            completed[i] = sorted(completed[i], key=normalized, reverse=True)[:beam_size]
            duplicates[i] = sorted(duplicates[i], key=normalized, reverse=True)[:beam_size]
            active[i] = sorted(proposals[i], key=lambda c: c[1], reverse=True)[:beam_size]
            if active[i]:
                last_active[i] = active[i]
            # Every future log-probability / soft-prior term is <= 0. For
            # nonnegative alpha this bounds any completion within max_tokens.
            if completed[i] and active[i]:
                upper_bound = max(c[1] for c in active[i]) / max_tokens ** length_penalty
                cutoff_candidate = completed[i][-1] if return_candidates else completed[i][0]
                enough = len(completed[i]) >= beam_size if return_candidates else True
                if enough and normalized(cutoff_candidate) >= upper_bound:
                    active[i] = []

    output = []
    for i in range(n):
        choices = completed[i] or duplicates[i] or last_active[i]
        if not choices:
            raise RuntimeError('Beam search produced no candidate.')
        tokens, score = max(choices, key=normalized)
        ended = tokens[-1] == model.stop
        parts = peptide_parts(tokens)
        unique = list(dict.fromkeys(parts))
        output.append({'peptides': unique, 'ended_with_eos': ended, 'truncated': not ended,
                       'chimeric_probability': probabilities[i], 'predicted_count': counts[i],
                       'routing': mode, 'count_constraint_satisfied': ended and len(unique) == counts[i],
                       'duplicate_segment': len(parts) != len(unique), 'raw_tokens': tokens,
                       'beam_score': normalized((tokens, score)), 'beam_size': beam_size,
                       'beam_length_penalty': length_penalty})
        if return_candidates:
            pool = []
            for candidate in sorted(choices, key=normalized, reverse=True):
                candidate_tokens = candidate[0]
                candidate_parts = peptide_parts(candidate_tokens)
                if candidate_tokens[-1] == model.stop and candidate_parts:
                    pool.append({'peptides': list(dict.fromkeys(candidate_parts)),
                                 'beam_score': normalized(candidate), 'raw_tokens': candidate_tokens,
                                 'ended_with_eos': True, 'truncated': False})
            output[-1]['candidates'] = pool
    return output


def beam_evaluation_rows(model, data, indices, device, beam_size, length_penalty):
    """Keep reference labels outside the candidate generator and its ranking."""
    spectra, precursors, labels = data.batch(indices, device)
    predictions = generate_beam(model, spectra, precursors, beam_size=beam_size,
                                length_penalty=length_penalty)
    rows = []
    for index, reference, prediction in zip(indices, labels, predictions):
        record = data.records[index]
        expected, actual = set(reference), set(prediction['peptides'])
        rows.append({'raw_file': record['raw_file'], 'scan': record['scan'],
                     'reference_pseudolabels': reference, 'prediction': prediction['peptides'],
                     'matched_peptides': len(expected & actual), 'complete_set_match': expected == actual,
                     'ended_with_eos': prediction['ended_with_eos'], 'truncated': prediction['truncated'],
                     'chimeric_probability': prediction['chimeric_probability'],
                     'predicted_count': prediction['predicted_count'], 'routing': prediction['routing'],
                     'count_constraint_satisfied': prediction['count_constraint_satisfied'],
                     'duplicate_segment': prediction['duplicate_segment'],
                     'beam_score': prediction.get('beam_score')})
    return rows


def summarize(rows):
    summary = {'spectra': len(rows)}
    for count, name in ((1, 'single'), (2, 'double')):
        selected = [r for r in rows if len(r['reference_pseudolabels']) == count]
        n = len(selected)
        summary[name] = {'spectra': n}
        for metric, predicate in (
            ('complete_set_match_rate', lambda r: r['complete_set_match']),
            ('at_least_one_match_rate', lambda r: r['matched_peptides'] >= 1),
            ('two_output_rate', lambda r: len(r['prediction']) == 2),
            ('truncation_rate', lambda r: r['truncated']),
            ('count_agreement_rate', lambda r: r['count_constraint_satisfied'])):
            summary[name][metric] = sum(predicate(r) for r in selected) / n if n else None
    return summary


@torch.no_grad()
def dual_slot_hypotheses(model, spectra, precursors, beam_size=4, length_penalty=1.):
    """Generate both count hypotheses once, without reference labels.

    Uses the existing slot decoder and distinct-pair ranking unchanged. The
    primary candidate can change when selecting a pair, so retain a separate
    single hypothesis instead of taking the first peptide from a chosen pair.
    """
    from dual_slot import _SlotView
    if getattr(model, 'architecture', '') not in ('dual_slot_v1', 'dual_slot_v2', 'dual_slot_evidence', 'dual_slot_separation'):
        raise ValueError('Count hypotheses require a dual-slot model.')
    model.eval()
    memory, mask = model.memories(spectra)
    probabilities = model.classify_memory(memory, mask).sigmoid().cpu().tolist()
    collect = model.joint_candidate_selection and beam_size > 1
    slots = [
        generate_beam(_SlotView(model, memory, mask, slot), spectra, precursors,
                      beam_size=beam_size, length_penalty=length_penalty,
                      return_candidates=collect)
        for slot in (0, 1)]
    result = []
    for probability, first, second in zip(probabilities, *slots):
        single = dict(peptides=first['peptides'][:], ended_with_eos=first['ended_with_eos'],
                      beam_score=first.get('beam_score'))
        a, b = first, second
        if collect:
            pairs = [(x, y) for x in first.get('candidates', [])
                     for y in second.get('candidates', [])
                     if len(x['peptides']) == len(y['peptides']) == 1
                     and x['peptides'][0] != y['peptides'][0]]
            if pairs:
                a, b = max(pairs, key=lambda pair: pair[0]['beam_score'] + pair[1]['beam_score'])
        double = dict(peptides=list(dict.fromkeys(a['peptides'] + b['peptides'])),
                      ended_with_eos=a['ended_with_eos'] and b['ended_with_eos'],
                      peptide_scores=[a.get('beam_score'), b.get('beam_score')])
        result.append(dict(chimeric_probability=probability, single=single, double=double))
    return result


def threshold_tradeoff(rows, thresholds, baseline_threshold, max_single_drop=.055):
    """Compare validation-only operating points; never change model weights."""
    if not 0 <= max_single_drop <= 1:
        raise ValueError('Single drop must be an absolute rate in [0,1].')
    table = []
    for threshold in sorted(set(thresholds) | {baseline_threshold}):
        if not 0 < threshold < 1:
            raise ValueError('Threshold must be in (0,1).')
        evaluated = []
        for row in rows:
            count = 2 if row['chimeric_probability'] >= threshold else 1
            prediction = row['double' if count == 2 else 'single']
            expected, actual = set(row['reference_pseudolabels']), set(prediction['peptides'])
            evaluated.append(dict(
                reference_pseudolabels=row['reference_pseudolabels'],
                prediction=prediction['peptides'], complete_set_match=expected == actual,
                matched_peptides=len(expected & actual), truncated=not prediction['ended_with_eos'],
                count_constraint_satisfied=prediction['ended_with_eos'] and len(actual) == count))
        table.append(dict(threshold=threshold, **summarize(evaluated)))
    baseline = next(x for x in table if x['threshold'] == baseline_threshold)
    if not baseline['single']['spectra'] or not baseline['double']['spectra']:
        raise ValueError('Need both label counts to choose a tradeoff.')
    single_base = baseline['single']['complete_set_match_rate']
    double_base = baseline['double']['complete_set_match_rate']
    for item in table:
        item['single_change_pp'] = 100 * (item['single']['complete_set_match_rate'] - single_base)
        item['double_change_pp'] = 100 * (item['double']['complete_set_match_rate'] - double_base)
    eligible = [x for x in table if x['single']['complete_set_match_rate'] >= single_base - max_single_drop]
    chosen = max(eligible, key=lambda x: (x['double']['complete_set_match_rate'],
                                         x['single']['complete_set_match_rate'],
                                         -abs(x['threshold'] - baseline_threshold)))
    return dict(baseline_threshold=baseline_threshold, max_single_drop_pp=100 * max_single_drop,
                recommended_threshold=chosen['threshold'], operating_points=table,
                note='Validation development comparison only; recommendation is not auto-applied. '
                     'Changes reflect output-count routing, not improved sequence weights.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--preprocessing-reference', help='Explicit original training data for a same-scan preprocessing-only comparison; identity files must match.')
    parser.add_argument('--cross-dataset-reference', help='Actual checkpoint training dataset; explicitly allow evaluation on another dataset without treating it as unseen data.')
    parser.add_argument('--count-policy', help='Optional checkpoint-specific candidate-count calibration JSON; also reports original hard routing as a comparison.')
    parser.add_argument('--output', help='Optional: save predictions here. Omit for console-only evaluation.')
    parser.add_argument('--split', choices=('validation', 'test', 'strict_validation', 'strict_test'), default='validation')
    parser.add_argument('--routing', choices=('unconstrained', 'soft', 'hard'), nargs='+', default=['hard'])
    parser.add_argument('--threshold', type=float, help='Validation-chosen cutoff; omitted = training cutoff.')
    parser.add_argument('--threshold-grid', type=float, nargs='+',
                        help='Dual slots, validation only: decode both counts once and compare thresholds.')
    parser.add_argument('--max-single-drop', type=float, default=.055,
                        help='Absolute single exact-match drop allowed when recommending a threshold (default 5.5 percentage points).')
    parser.add_argument('--monitor-indices', help='Existing validation monitor JSON list; preserves the historical development sample.')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--beam-size', type=int, default=1, help='1 keeps original greedy generation; >1 explores multiple candidates.')
    parser.add_argument('--beam-length-penalty', type=float, default=1.0, help='Nonnegative token-length exponent for beam candidate ranking.')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--per-class', type=int, default=0, help='Fixed random sample from each label count, instead of the first N spectra.')
    parser.add_argument('--seed', type=int, default=463)
    parser.add_argument('--progress-every', type=int, default=128)
    parser.add_argument('--classification-only', action='store_true')
    parser.add_argument('--generate', action='store_true', help='Also generate after classifier-only training.')
    args = parser.parse_args()
    if args.batch_size < 1 or min(args.limit, args.per_class) < 0 or args.progress_every < 1:
        parser.error('Invalid batch size / limit / progress interval.')
    if args.limit and args.per_class:
        parser.error('Choose --limit or --per-class, not both.')
    if args.beam_size < 1 or not math.isfinite(args.beam_length_penalty) or args.beam_length_penalty < 0:
        parser.error('Invalid beam size / length penalty.')
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error('Threshold must lie strictly between zero and one.')
    if args.threshold_grid and (
            args.split not in ('validation', 'strict_validation') or args.classification_only
            or any(not 0 < x < 1 for x in args.threshold_grid) or not 0 <= args.max_single_drop <= 1):
        parser.error('Threshold comparison requires validation generation and valid thresholds/drop allowance.')
    if args.monitor_indices and (args.limit or args.per_class or args.split not in ('validation', 'strict_validation')):
        parser.error('Monitor indices require validation and cannot combine with other sampling options.')
    output = Path(args.output) if args.output else None
    if output is not None and output.exists():
        raise SystemExit('Use a new output directory; previous evaluation will not be overwritten.')
    started = time.perf_counter()
    torch.set_num_threads(4)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable.')
    data = PreparedData(args.data)
    saved = torch.load(args.weights, map_location='cpu', weights_only=False)
    if saved.get('format') != FORMAT:
        raise ValueError('Use a full v4 checkpoint from this workflow.')
    config = saved['immutable_config']
    policy = None
    if args.count_policy:
        policy = json.loads(Path(args.count_policy).read_text())
        if (not policy.get('accepted') or policy['checkpoint_sha256'] != digest(args.weights)
                or policy['beam_size'] != args.beam_size or policy['length_penalty'] != args.beam_length_penalty):
            raise ValueError('Candidate policy requires its accepted checkpoint and matching beam settings.')
        if policy.get('feature_code_sha256') != digest(Path(__file__).with_name('candidate_selection.py')):
            raise ValueError('Candidate feature implementation differs from calibration.')
        if args.threshold_grid or args.routing != ['hard'] or args.beam_size <= 1 or args.classification_only:
            raise ValueError('Candidate policy needs dual-slot hard generation, beam>1, and no threshold grid.')
        if args.threshold is not None and args.threshold != policy['threshold']:
            raise ValueError('Use the saved candidate policy threshold; do not silently reuse the classifier cutoff.')
        if config.get('architecture') not in ('dual_slot_v1', 'dual_slot_v2', 'dual_slot_evidence', 'dual_slot_separation'):
            raise ValueError('Candidate policy requires dual-slot architecture.')
    for filename in ('model.py', 'separator_base.py', 'training_common.py', 'workflow.py'):
        if config['code_sha256'][filename] != digest(Path(__file__).parent / filename):
            raise ValueError('Model/evaluation implementation changed since training: ' + filename)
    if config['base_sha256'] != digest(args.base_checkpoint):
        raise ValueError('Checkpoint base hash does not match.')
    if args.cross_dataset_reference and args.preprocessing_reference:
        raise ValueError('Use either cross-dataset evaluation or preprocessing-only comparison.')
    if args.cross_dataset_reference:
        reference = Path(args.cross_dataset_reference).resolve()
        if digest(reference / 'manifest.json') != config['data_manifest_sha256']:
            raise ValueError('Cross-dataset reference must be the checkpoint training dataset.')
    elif config['data_manifest_sha256'] != digest(data.path / 'manifest.json'):
        if not args.preprocessing_reference:
            raise ValueError('Checkpoint data hash does not match; a preprocessing-only comparison needs an explicit reference.')
        reference = Path(args.preprocessing_reference).resolve()
        reference_manifest = json.loads((reference / 'manifest.json').read_text())
        reference_hash = digest(reference / 'manifest.json')
        evaluated_hash = digest(data.path / 'manifest.json')
        linked = (data.manifest.get('source_data_manifest_sha256') == reference_hash or
                  reference_manifest.get('source_data_manifest_sha256') == evaluated_hash)
        if reference_hash != config['data_manifest_sha256'] or not linked:
            raise ValueError('Preprocessing variant must derive from this checkpoint training dataset.')
        for filename in ('records.jsonl', 'splits.json', 'precursors.npy'):
            if digest(reference / filename) != digest(data.path / filename):
                raise ValueError('Preprocessing comparison changed spectrum identities, labels, split or precursors: ' + filename)
    if config['task'] == 'sequence' and any(m != 'unconstrained' for m in args.routing) and not args.classification_only:
        raise ValueError('Sequence-only control does not use the classifier for decoding. Use --routing unconstrained.')
    model_class = ChimeraSequenceModel
    if config.get('architecture') in ('dual_slot_v1', 'dual_slot_v2', 'dual_slot_evidence', 'dual_slot_separation'):
        from dual_slot import DualSlotModel, DistinctDualSlotModel, EvidenceDualSlotModel, SeparationDualSlotModel
        model_class = {'dual_slot_v1': DualSlotModel, 'dual_slot_v2': DistinctDualSlotModel,
                       'dual_slot_evidence': EvidenceDualSlotModel, 'dual_slot_separation': SeparationDualSlotModel}[config['architecture']]
        if args.routing != ['hard']:
            raise ValueError('Dual-slot evaluation currently requires --routing hard.')
        if config['code_sha256']['dual_slot.py'] != digest(Path(__file__).parent / 'dual_slot.py'):
            raise ValueError('Dual-slot implementation changed since training.')
    model = model_class(args.base_checkpoint, threshold=args.threshold if args.threshold is not None else config['threshold'],
                        routing_strength=config['routing_strength'])
    if config.get('architecture') in ('dual_slot_v1', 'dual_slot_v2', 'dual_slot_evidence', 'dual_slot_separation'):
        model.evaluation_beam_size = args.beam_size
    model.load_state_dict(saved['state_dict'], strict=True)
    epoch = saved['epoch_index']
    del saved
    model.to(args.device).eval()
    indices = data.splits[args.split]
    if not indices:
        raise SystemExit('Selected strict evaluation subset is empty; report this limitation instead of inventing a score.')
    if args.monitor_indices:
        selected = json.loads(Path(args.monitor_indices).read_text())
        if (not isinstance(selected, list) or not selected or
                any(type(i) is not int for i in selected) or len(selected) != len(set(selected))
                or not set(selected) <= set(indices)):
            raise ValueError('Monitor must contain unique indices entirely inside the selected validation split.')
        indices = selected
    if args.limit:
        indices = indices[:args.limit]
    if args.per_class:
        rng = random.Random(args.seed)
        selected = []
        for count in (1, 2):
            pool = [i for i in indices if len(data.records[i]['model_labels']) == count]
            selected.extend(rng.sample(pool, min(args.per_class, len(pool))))
        indices = selected
    if args.threshold_grid:
        rows = []
        print(f'Count hypotheses: 0/{len(indices)} (0.0%)', flush=True)
        for start in range(0, len(indices), args.batch_size):
            group = indices[start:start + args.batch_size]
            spectra, precursors, labels = data.batch(group, args.device)
            hypotheses = dual_slot_hypotheses(model, spectra, precursors, args.beam_size, args.beam_length_penalty)
            for index, reference, hypothesis in zip(group, labels, hypotheses):
                rows.append(dict(index=int(index), reference_pseudolabels=reference, **hypothesis))
            done = start + len(group)
            if done % args.progress_every < len(group) or done == len(indices):
                print(f'Count hypotheses: {done}/{len(indices)} ({100*done/len(indices):.1f}%)', flush=True)
        result = threshold_tradeoff(rows, args.threshold_grid, config['threshold'], args.max_single_drop)
        result.update(split=args.split, preview_only=bool(args.limit or args.per_class or args.monitor_indices),
                      checkpoint=str(Path(args.weights).resolve()), checkpoint_sha256=digest(args.weights),
                      checkpoint_epoch=epoch, data_manifest_sha256=digest(data.path / 'manifest.json'),
                      evaluation_code_sha256=digest(__file__), beam_size=args.beam_size,
                      length_penalty=args.beam_length_penalty, device=args.device,
                      elapsed_minutes=(time.perf_counter() - started) / 60,
                      label_status=data.manifest.get('label_source', 'DDA+ search pseudo-labels'),
                      limitations=data.manifest.get('limitations', []))
        print(json.dumps(result, indent=2), flush=True)
        if output is not None:
            output.mkdir(parents=True)
            write_json(output / 'metrics.json', result)
            write_json(output / 'count_hypotheses.json', rows)
        return
    losses, classifier_rows = evaluate_components(model, data, indices, args.device, args.batch_size, config['task'],
                                                  progress_every=args.progress_every)
    predictions, summaries = {}, {}
    policy_rows, policy_probabilities = [], []
    should_generate = not args.classification_only and (config['task'] != 'classifier' or args.generate)
    if should_generate:
        for mode in dict.fromkeys(args.routing):
            model.routing = mode
            rows, last_report = [], 0
            print(f'Generation [{mode}]: 0/{len(indices)} (0.0%)', flush=True)
            for start in range(0, len(indices), args.batch_size):
                group = indices[start:start + args.batch_size]
                if policy is not None:
                    from candidate_selection import features, probabilities
                    spectra, precursors, labels = data.batch(group, args.device)
                    hypotheses = dual_slot_hypotheses(model, spectra, precursors, args.beam_size, args.beam_length_penalty)
                    values = [features(h, s.detach().cpu().numpy()) for h, s in zip(hypotheses, spectra)]
                    calibrated = probabilities(policy, values)
                    batch_rows = []
                    for index, reference, h, probability in zip(group, labels, hypotheses, calibrated):
                        for target, p, cutoff in ((batch_rows, h['chimeric_probability'], config['threshold']),
                                                  (policy_rows, float(probability), policy['threshold'])):
                            count = 2 if p >= cutoff else 1
                            prediction = h['double' if count == 2 else 'single']
                            record = data.records[index]
                            target.append(dict(raw_file=record['raw_file'], scan=record['scan'],
                                               reference_pseudolabels=reference, prediction=prediction['peptides'],
                                               matched_peptides=len(set(reference) & set(prediction['peptides'])),
                                               complete_set_match=set(reference) == set(prediction['peptides']),
                                               ended_with_eos=prediction['ended_with_eos'], truncated=not prediction['ended_with_eos'],
                                               chimeric_probability=h['chimeric_probability'], selection_probability=float(p),
                                               predicted_count=count, routing='hard' if target is batch_rows else 'candidate_policy',
                                               count_constraint_satisfied=prediction['ended_with_eos'] and len(prediction['peptides']) == count))
                        policy_probabilities.append(float(probability))
                elif args.beam_size == 1:
                    _, batch_rows = evaluate_sequences(model, data, group, args.device, args.batch_size)
                else:
                    batch_rows = beam_evaluation_rows(model, data, group, args.device, args.beam_size,
                                                      args.beam_length_penalty)
                rows.extend(batch_rows)
                done = start + len(group)
                if done-last_report >= args.progress_every or done == len(indices):
                    print(f'Generation [{mode}]: {done}/{len(indices)} ({100*done/len(indices):.1f}%)', flush=True)
                    last_report = done
            predictions[mode], summaries[mode] = rows, summarize(rows)
    if policy is not None:
        predictions['candidate_policy'], summaries['candidate_policy'] = policy_rows, summarize(policy_rows)
    result = {'split': args.split, 'preview_only': bool(args.limit or args.per_class or args.monitor_indices), 'checkpoint_epoch': epoch,
              'sampling': {'per_class': args.per_class, 'limit': args.limit, 'seed': args.seed,
                           'note': 'Stratified previews change class prevalence; precision is not full-population precision.' if args.per_class else ''},
              'checkpoint': str(Path(args.weights).resolve()), 'checkpoint_sha256': digest(args.weights),
              'data_manifest_sha256': digest(data.path / 'manifest.json'), 'task': config['task'],
              'training_data_manifest_sha256': config['data_manifest_sha256'],
              'cross_dataset_evaluation': bool(args.cross_dataset_reference),
              'cross_dataset_reference': str(Path(args.cross_dataset_reference).resolve()) if args.cross_dataset_reference else None,
              'preprocessing_reference': args.preprocessing_reference,
              'preprocessing': data.manifest.get('preprocessing'),
              'head_updated_in_this_task': config['task'] != 'sequence', 'validation_or_test': losses,
              'generation': summaries,
              'decoding': {'beam_size': args.beam_size, 'length_penalty': args.beam_length_penalty,
                           'evaluation_code_sha256': digest(Path(__file__))},
              'elapsed_minutes': (time.perf_counter()-started)/60,
              'label_status': data.manifest.get('label_source', 'DDA+ search pseudo-labels; one search label does not prove a truly pure spectrum.'),
              'limitations': data.manifest.get('limitations', [])}
    if policy is not None:
        from workflow import classification_metrics
        truth = [int(len(data.records[i]['model_labels']) == 2) for i in indices]
        selected_classification = classification_metrics(truth, policy_probabilities, policy['threshold'])
        result['classification_by_decision'] = {'original_head': losses['classification'], 'candidate_policy': selected_classification}
        result['candidate_policy'] = {'path': str(Path(args.count_policy).resolve()), 'sha256': digest(args.count_policy),
                                      'checkpoint_sha256': policy['checkpoint_sha256'], 'threshold': policy['threshold']}
        result['limitations'] = list(result['limitations']) + ['Policy fitted on development validation samples; arbitrary previews may overlap calibration samples.']
    if args.cross_dataset_reference:
        result['limitations'] = list(result['limitations']) + [
            'Back-evaluation: this checkpoint lineage may already have used this dataset; not a new independent test.']
    print(json.dumps(result, indent=2), flush=True)
    if output is None:
        print('Console-only evaluation: no report or prediction files written.', flush=True)
        return
    output.mkdir(parents=True)
    write_json(output / 'metrics.json', result)
    write_json(output / 'classification.json', classifier_rows)
    for mode, rows in predictions.items():
        write_json(output / f'predictions_{mode}.json', rows)
        with (output / f'predictions_{mode}.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(['raw_file', 'scan', 'chimeric_probability', 'predicted_count', 'reference_count',
                             'reference_pseudolabels', 'prediction_count', 'prediction', 'complete_set_match',
                             'matched_peptides', 'count_constraint_satisfied', 'truncated'])
            for row in rows:
                writer.writerow([row['raw_file'], row['scan'], row['chimeric_probability'], row['predicted_count'],
                                 len(row['reference_pseudolabels']), ' | '.join(row['reference_pseudolabels']),
                                 len(row['prediction']), ' | '.join(row['prediction']), row['complete_set_match'],
                                 row['matched_peptides'], row['count_constraint_satisfied'], row['truncated']])
    print('Saved: '+str(output.resolve()), flush=True)


if __name__ == '__main__':
    main()
