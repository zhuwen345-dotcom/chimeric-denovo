"""User-launched classifier warmup, joint training, and sequence-only control.

No training on import. New task/data means a new run and fresh optimizer.
--resume restores THIS run's model, optimizer, RNG and next batch exactly.
--rebatch-from explicitly branches a paused run, retaining optimizer/RNG and
the exact next spectrum in the epoch shuffle while changing the batch size.
"""
import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import time

os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import torch

from model import ChimeraSequenceModel
from separator_base import optimizer_step_with_new_token_rate
from training_common import LossMeter, PreparedData, digest, evaluate_sequences, write_json
from workflow import FORMAT, evaluate_components, initialize, validate_splits

SLOT_ARCHITECTURES = ('dual_slot_v1', 'dual_slot_v2', 'dual_slot_evidence', 'dual_slot_separation')
COMPACT_ARCHITECTURES = ('dual_slot_v2', 'dual_slot_evidence', 'dual_slot_separation')


def cap_training_repeats(records, indices, seed, sequence_cap=0, chimeric_cap=0):
    """Deterministic train-only thinning; never change validation/test labels.

    Sequence keys are a singleton or unordered pair of prepared peptidoforms.
    Peptides retain the preprocessing's I/L and modification conventions.
    """
    if not sequence_cap and not chimeric_cap:
        return list(indices)
    candidates = list(indices)
    random.Random(seed).shuffle(candidates)
    groups, peptides, kept = {}, {}, set()
    for index in candidates:
        key = tuple(sorted(records[index]['model_labels']))
        if sequence_cap and groups.get(key, 0) >= sequence_cap:
            continue
        if len(key) == 2 and chimeric_cap and any(peptides.get(p, 0) >= chimeric_cap for p in key):
            continue
        kept.add(index)
        groups[key] = groups.get(key, 0) + 1
        if len(key) == 2:
            for peptide in key:
                peptides[peptide] = peptides.get(peptide, 0) + 1
    return [i for i in indices if i in kept]


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def save_reports(run, history):
    write_json(run / 'epochs.json', history)
    if history:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
        temporary = run / 'epochs.csv.tmp'
        temporary.write_text(stream.getvalue(), encoding='utf-8')
        temporary.replace(run / 'epochs.csv')


def selection_values(task, validation, monitor_metrics, double_priority=1., single_floor=None):
    """Use identical checkpoint selection for the initializer and trained epochs."""
    if task == 'classifier':
        ce = validation['classification_bce']['all']
        primary = (validation['classification']['auroc'],)
    else:
        single = monitor_metrics['single']['complete_set_match_rate']
        double = monitor_metrics['double']['complete_set_match_rate']
        ce = (validation['sequence_ce']['single'] + validation['sequence_ce']['double']) / 2
        primary = ((single + double_priority * double) / (1 + double_priority), double)
        if single_floor is not None:
            primary = (single >= single_floor,) + primary
    return primary, primary + (-ce,), ce


def training_objective(sequence, classification, regularizer, labels, task,
                       double_weight, classification_weight,
                       classification_double_weight=None):
    """Separate sequencing emphasis from cardinality calibration when requested.

    None preserves the historical objective and its arithmetic. An explicit
    classification weight changes only the classification term in joint runs.
    """
    weights = classification.new_tensor([double_weight if len(p) == 2 else 1.
                                        for p in labels])
    if task == 'classifier':
        return (classification * weights).sum() / weights.sum()
    if classification_double_weight is None:
        per_spectrum = sequence + (classification_weight * classification if task == 'joint' else 0)
        if regularizer is not None:
            per_spectrum = per_spectrum + regularizer
        return (per_spectrum * weights).sum() / weights.sum()
    if task != 'joint':
        raise ValueError('Separate classification weighting requires joint training.')
    sequence_terms = sequence if regularizer is None else sequence + regularizer
    classification_weights = classification.new_tensor([
        classification_double_weight if len(p) == 2 else 1. for p in labels])
    return ((sequence_terms * weights).sum() / weights.sum() +
            classification_weight * (classification * classification_weights).sum() /
            classification_weights.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--initialize-from', help='New experiment: inherit trusted full project weights only.')
    parser.add_argument('--baseline-before-training', action='store_true',
                        help='Evaluate and retain the initializer before the first optimizer update; resumable if interrupted.')
    parser.add_argument('--task', choices=('classifier', 'joint', 'sequence'), default='classifier')
    parser.add_argument('--architecture', choices=('separator',) + SLOT_ARCHITECTURES, default='separator')
    parser.add_argument('--decode-beam-size', type=int, default=1)
    parser.add_argument('--slot-lr', type=float, default=1e-3)
    parser.add_argument('--diversity-weight', type=float, default=.5)
    parser.add_argument('--diversity-margin', type=float, default=.2)
    parser.add_argument('--evidence-weight', type=float, default=.1)
    parser.add_argument('--hard-peptide-weight', type=float, default=0.,
                        help='Evidence dual-slot training only: extra CE of the harder matched peptide; .25 gives 1.5x its local CE gradient. Not an intensity estimate.')
    parser.add_argument('--cap-sequence-repeats', type=int, default=0,
                        help='Train-only cap per singleton sequence or unordered peptide pair; 0 disables. Conference-inspired starting value: 2.')
    parser.add_argument('--cap-chimeric-peptide', type=int, default=0,
                        help='Train-only cap on double-label spectra containing each peptide; 0 disables. Conference-inspired starting value: 10.')
    parser.add_argument('--double-weight', type=float, default=1., help='Relative training loss weight for double-label spectra; validation remains unweighted.')
    parser.add_argument('--classification-double-weight', type=float, default=None,
                        help='Joint only: independent double-label BCE weight. Omit for legacy coupling; 1 leaves classification unweighted.')
    parser.add_argument('--selection-double-priority', type=float, default=1.,
                        help='Relative double exact-match priority for best checkpoint and early stopping. 1 preserves equal-class selection; does not change the loss.')
    parser.add_argument('--selection-max-single-drop', type=float, default=None,
                        help='Optional absolute single exact-match drop budget relative to this stage initializer. Requires baseline-before-training.')
    parser.add_argument('--encoder-last-layers', type=int, choices=(0, 1), default=0)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--classification-lr', type=float, default=None,
                        help='Classification head only; omitted uses head-lr. Peak evidence retains head-lr.')
    parser.add_argument('--freeze-classifier', action='store_true',
                        help='Joint sequencing adaptation with fixed encoder and classifier; requires encoder-last-layers=0.')
    parser.add_argument('--separation-lr', type=float, default=1e-4)
    parser.add_argument('--separation-only', action='store_true',
                        help='Freeze all inherited parameters and train only spectral_separation.')
    parser.add_argument('--rebaseline-on-resume', action='store_true',
                        help='Evaluate a changed architecture at the preserved cumulative epoch/sample cursor.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Decoder learning rate.')
    parser.add_argument('--encoder-lr', type=float, default=1e-6)
    parser.add_argument('--classification-weight', type=float, default=0.1)
    parser.add_argument('--new-token-lr-factor', type=float, default=100.0)
    parser.add_argument('--routing', choices=('unconstrained', 'soft', 'hard'), default='hard')
    parser.add_argument('--threshold', type=float, default=0.5, help='Choose on validation only, never on test.')
    parser.add_argument('--routing-strength', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--monitor-per-class', type=int, default=64)
    parser.add_argument('--seed', type=int, default=454)
    parser.add_argument('--resume', nargs='?', const='auto')
    parser.add_argument('--rebatch-from', help='Paused v4 checkpoint; new run, same task/optimizer/data, different batch size.')
    parser.add_argument('--prepare-only', action='store_true', help='Rebatch checkpoint and save paused, without any optimizer update.')
    parser.add_argument('--save-every', type=int, default=200)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--session-minutes', type=float, default=120)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.separation_only and (args.architecture != 'dual_slot_separation' or args.encoder_last_layers != 0):
        parser.error('Separation-only requires dual_slot_separation and a frozen encoder.')
    if args.freeze_classifier and (args.task != 'joint' or args.encoder_last_layers != 0):
        parser.error('Freeze-classifier requires joint with a fully frozen encoder.')
    if not math.isfinite(args.separation_lr) or args.separation_lr <= 0:
        parser.error('Separation learning rate must be positive and finite.')
    if args.resume and args.rebatch_from:
        parser.error('Use either --resume or --rebatch-from.')
    if args.prepare_only and not args.rebatch_from:
        parser.error('--prepare-only requires --rebatch-from.')
    if min(args.epochs, args.batch_size, args.patience, args.monitor_per_class,
           args.save_every, args.log_every, args.threads) < 1:
        parser.error('Counts must be positive.')
    if min(args.lr, args.head_lr, args.encoder_lr, args.new_token_lr_factor) <= 0:
        parser.error('Learning rates / token factor must be positive.')
    if args.classification_lr is not None and (not math.isfinite(args.classification_lr) or args.classification_lr <= 0):
        parser.error('Classification learning rate must be positive and finite.')
    if args.decode_beam_size < 1 or args.slot_lr <= 0:
        parser.error('Beam size and slot learning rate must be positive.')
    if args.diversity_weight < 0 or args.diversity_margin < 0 or args.evidence_weight < 0 or not math.isfinite(args.double_weight) or args.double_weight <= 0:
        parser.error('Diversity weight / margin must be nonnegative.')
    if args.classification_double_weight is not None and (
            args.task != 'joint' or not math.isfinite(args.classification_double_weight) or
            args.classification_double_weight <= 0):
        parser.error('Independent classification weighting requires joint and a finite positive value.')
    if not 0 <= args.hard_peptide_weight <= 1 or (args.hard_peptide_weight and args.architecture != 'dual_slot_evidence'):
        parser.error('Hard-peptide weight requires dual_slot_evidence and a finite value in [0,1].')
    if min(args.cap_sequence_repeats, args.cap_chimeric_peptide) < 0:
        parser.error('Training repeat caps must be nonnegative integers.')
    if not math.isfinite(args.selection_double_priority) or args.selection_double_priority <= 0:
        parser.error('Selection priority must be finite and positive.')
    if args.task == 'classifier' and args.selection_double_priority != 1:
        parser.error('Exact-match selection priority applies only to sequencing tasks.')
    if args.selection_max_single_drop is not None and (
            args.task == 'classifier' or not args.baseline_before_training
            or not 0 <= args.selection_max_single_drop <= 1):
        parser.error('Single-drop guard needs sequencing, baseline-before-training, and a finite budget in [0,1].')
    if args.architecture in SLOT_ARCHITECTURES and (args.task != 'joint' or args.routing != 'hard' or args.new_token_lr_factor != 1):
        parser.error('Dual slots require joint, hard routing and new-token-lr-factor=1.')
    if args.session_minutes < 0 or args.classification_weight < 0 or args.routing_strength < 0 or not 0 < args.threshold < 1:
        parser.error('Invalid loss weight / session duration / routing setting.')
    if args.task == 'classifier' and args.encoder_last_layers:
        parser.error('Classifier warmup freezes the encoder; use joint for encoder adaptation.')
    if args.task == 'sequence' and args.routing != 'unconstrained':
        parser.error('Sequence control must use unconstrained decoding, without classifier routing.')
    if args.task == 'joint' and args.classification_weight == 0:
        parser.error('Use --task sequence for the zero-classification-loss control.')
    if not (Path(args.data) / 'manifest.json').exists():
        raise SystemExit('Prepared DATA directory is absent. Expand/search/split/cache data before training.')
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / '.training.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This run is already training.')
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; use the migrated casanovo4 environment.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    print('Reading prepared data, hashes and split overlap audit...', flush=True)
    data = PreparedData(args.data)
    counts = validate_splits(data)
    sampling = None
    if args.cap_sequence_repeats or args.cap_chimeric_peptide:
        original_count = len(data.splits['train'])
        data.splits['train'] = cap_training_repeats(
            data.records, data.splits['train'], args.seed,
            args.cap_sequence_repeats, args.cap_chimeric_peptide)
        counts = validate_splits(data)
        sampling = {'original_train_spectra': original_count,
                    'retained_train_spectra': len(data.splits['train']),
                    'sequence_cap': args.cap_sequence_repeats,
                    'chimeric_peptide_cap': args.cap_chimeric_peptide,
                    'indices_sha256': __import__('hashlib').sha256(
                        json.dumps(data.splits['train']).encode()).hexdigest()}
        print('Train-only repeat caps: ' + json.dumps(sampling), flush=True)
    base = Path(args.base_checkpoint).resolve()
    sources = ('train_resumable.py', 'model.py', 'separator_base.py', 'training_common.py', 'workflow.py')
    if args.architecture in SLOT_ARCHITECTURES:
        sources += ('dual_slot.py', 'evaluate_holdout.py')
    immutable = {
        'data_manifest_sha256': digest(data.path / 'manifest.json'), 'base_sha256': digest(base),
        'initializer_sha256': digest(args.initialize_from) if args.initialize_from else None,
        'baseline_before_training': args.baseline_before_training,
        'task': args.task, 'encoder_last_layers': args.encoder_last_layers,
        'batch_size': args.batch_size, 'head_lr': args.head_lr, 'lr': args.lr, 'encoder_lr': args.encoder_lr,
        'classification_weight': args.classification_weight, 'new_token_lr_factor': args.new_token_lr_factor,
        'double_weight': args.double_weight,
        'routing': args.routing, 'threshold': args.threshold, 'routing_strength': args.routing_strength,
        'seed': args.seed, 'patience': args.patience, 'monitor_per_class': args.monitor_per_class,
        'pooling': 'mean of real peaks, excluding padding and global token',
        'objective': 'per-spectrum minimum-order sequence CE + lambda * binary logit CE; no class weighting',
        'optimizer': 'AdamW; head weight decay 0, encoder/decoder 0.01',
        'precision': 'float32', 'scheduler': 'none; constant group learning rates',
        'code_sha256': {name: digest(Path(__file__).parent / name) for name in sources},
        'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__)}
    if args.architecture in SLOT_ARCHITECTURES:
        immutable.update(architecture=args.architecture, decode_beam_size=args.decode_beam_size,
                         slot_lr=args.slot_lr,
                         objective='mean independent-slot peptide CE, minimum over label permutations + lambda * classification BCE')
    if args.architecture in COMPACT_ARCHITECTURES:
        immutable.update(diversity_weight=args.diversity_weight, diversity_margin=args.diversity_margin,
                         objective='permutation-invariant peptide CE + duplicate-set contrastive hinge + lambda * classification BCE',
                         checkpoint_policy='last: full resume; best: model and provenance; best_ce: not written')
    if args.architecture in ('dual_slot_evidence', 'dual_slot_separation'):
        immutable.update(evidence_weight=args.evidence_weight,
                         objective='double-weighted spectrum loss: PIT peptide CE + duplicate contrast + weak b/y peak supervision + classification BCE',
                         peak_annotation='b/y z1,z2; max(20ppm,0.02Da); shared peaks allowed; no annotations at inference')
    if args.classification_double_weight is not None:
        immutable.update(classification_double_weight=args.classification_double_weight,
                         objective='separately normalized double-weighted sequence/regularizer loss + lambda * independently weighted classification BCE')
    if args.classification_lr is not None:
        immutable['classification_lr'] = args.classification_lr
    if args.freeze_classifier:
        immutable['freeze_classifier'] = True
    if args.architecture == 'dual_slot_separation':
        immutable.update(separation_lr=args.separation_lr, allocation_weight=.05,
                         separation='four-state weak peak allocation + one contextual Transformer residual; neutral initialization')
    if args.separation_only:
        immutable['separation_only'] = True
    if args.rebaseline_on_resume:
        immutable['rebaseline_on_resume'] = True
    if args.selection_double_priority != 1.:
        immutable['selection_double_priority'] = args.selection_double_priority
    if args.selection_max_single_drop is not None:
        immutable['selection_max_single_drop'] = args.selection_max_single_drop
    if args.hard_peptide_weight:
        immutable['hard_peptide_weight'] = args.hard_peptide_weight
    if sampling is not None:
        immutable['training_repeat_sampling'] = sampling
    resuming = bool(args.resume or args.rebatch_from)
    migration = None
    saved = None
    if resuming:
        path = (Path(args.rebatch_from).resolve() if args.rebatch_from else
                run / 'last.pt' if args.resume == 'auto' else Path(args.resume).resolve())
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if args.rebatch_from:
            parent = Path(saved['run_directory']).resolve()
            if saved.get('format') != FORMAT or path.parent != parent or parent == run:
                raise ValueError('Rebatch requires a full v4 checkpoint and a distinct new run.')
            if (run / 'last.pt').exists() or (run / 'config.json').exists():
                raise ValueError('Rebatch target exists: use --resume for its next session.')
            parent_lock = (parent / '.training.lock').open('a')
            fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if json.loads((parent / 'status.json').read_text()).get('state') != 'paused':
                raise ValueError('Pause the parent and wait for its durable checkpoint first.')
            previous = saved['immutable_config']
            allowed_changes = {'batch_size', 'code_sha256'}
            if ({k: v for k, v in previous.items() if k not in allowed_changes} !=
                    {k: v for k, v in immutable.items() if k not in allowed_changes}):
                raise ValueError('Rebatch may change only batch size and the audited trainer implementation.')
            for name in sources:
                if digest(parent / 'source_snapshot' / name) != previous['code_sha256'][name]:
                    raise ValueError('Parent snapshot changed: ' + name)
                if name != 'train_resumable.py' and previous['code_sha256'][name] != immutable['code_sha256'][name]:
                    raise ValueError('Rebatch may not change model, objective, data or evaluation: ' + name)
            migration = {'kind': 'explicit batch-size change with sample cursor',
                         'parent_run': str(parent), 'parent_checkpoint': str(path),
                         'parent_checkpoint_sha256': digest(path),
                         'parent_global_step': saved['global_step'],
                         'old_batch_size': previous['batch_size'], 'new_batch_size': args.batch_size,
                         'epoch_sample_cursor': min(saved.get('next_sample', saved['next_batch'] * previous['batch_size']),
                                                    len(data.splits['train'])),
                         'optimizer_and_rng_retained': True,
                         'note': 'Same shuffled spectrum stream; different update grouping, not numerically identical continuation.'}
            write_json(run / 'batch_transition.json', migration)
        else:
            if saved.get('format') != FORMAT or saved['immutable_config'] != immutable:
                raise ValueError('Resume task/data/code/config mismatch. Use a NEW run for a changed experiment.')
            if Path(saved['run_directory']).resolve() != run:
                raise ValueError('Resume into the same run directory.')
            migration = saved.get('batch_transition')
    elif (run / 'config.json').exists() or (run / 'last.pt').exists():
        raise SystemExit('Run exists; use --resume or a new RUN name.')
    if args.architecture in SLOT_ARCHITECTURES:
        from dual_slot import DualSlotModel, DistinctDualSlotModel, EvidenceDualSlotModel, SeparationDualSlotModel
        model_class = {'dual_slot_v1': DualSlotModel, 'dual_slot_v2': DistinctDualSlotModel,
                       'dual_slot_evidence': EvidenceDualSlotModel, 'dual_slot_separation': SeparationDualSlotModel}[args.architecture]
        extra = ({'diversity_weight': args.diversity_weight, 'diversity_margin': args.diversity_margin}
                 if args.architecture in COMPACT_ARCHITECTURES else {})
        if args.architecture in ('dual_slot_evidence', 'dual_slot_separation'):
            extra['evidence_weight'] = args.evidence_weight
            extra['hard_peptide_weight'] = args.hard_peptide_weight
        model = model_class(base, routing=args.routing, threshold=args.threshold,
                            routing_strength=args.routing_strength,
                            evaluation_beam_size=args.decode_beam_size, **extra)
    else:
        model = ChimeraSequenceModel(base, routing=args.routing, threshold=args.threshold,
                                    routing_strength=args.routing_strength)
    if resuming:
        model.load_state_dict(saved['state_dict'], strict=True)
        training_exposure = saved['training_exposure']
    else:
        training_exposure = initialize(model, args.initialize_from, immutable['base_sha256'], data)
    model.configure_task(args.task, args.encoder_last_layers)
    if args.separation_only:
        model.requires_grad_(False)
        model.spectral_separation.requires_grad_(True)
    if args.freeze_classifier:
        model.classification_head.requires_grad_(False)
        if any(p.requires_grad for p in model.encoder.parameters()):
            raise RuntimeError('Classifier stability requires a frozen encoder.')
    model.to('cuda')
    groups = []
    for name, module, lr, wd in (('decoder', model.decoder, args.lr, 0.01),
                                ('classification_head', model.classification_head, args.classification_lr if args.classification_lr is not None else args.head_lr, 0.0),
                                ('encoder_last_layer', model.encoder, args.encoder_lr, 0.01)):
        parameters = [p for p in module.parameters() if p.requires_grad]
        if parameters:
            groups.append({'params': parameters, 'lr': lr, 'weight_decay': wd, 'group_name': name})
    if args.architecture in SLOT_ARCHITECTURES and not args.separation_only:
        groups.append({'params': [model.slot_embeddings, model.precursor_scales],
                       'lr': args.slot_lr, 'weight_decay': 0., 'group_name': 'slot_conditioning'})
    if args.architecture in ('dual_slot_evidence', 'dual_slot_separation') and not args.separation_only:
        groups.append({'params': list(model.peak_evidence.parameters()), 'lr': args.head_lr,
                       'weight_decay': .01, 'group_name': 'peak_evidence'})
    if args.architecture == 'dual_slot_separation':
        groups.append({'params': list(model.spectral_separation.parameters()), 'lr': args.separation_lr,
                       'weight_decay': .01, 'group_name': 'spectral_separation'})
    optimizer = torch.optim.AdamW(groups)
    epoch = next_batch = next_sample = global_step = no_improvement = 0
    epoch_active_seconds = 0.0
    history, sequence_meter, classification_meter = [], LossMeter(), LossMeter()
    best_key = best_primary = None
    best_ce = float('inf')
    baseline = {'state': 'pending', 'active_seconds': 0.0} if args.baseline_before_training else None
    rng = random.Random(args.seed + 9)
    monitor = []
    for label_count in (1, 2):
        pool = [i for i in data.splits['validation'] if len(data.records[i]['model_labels']) == label_count]
        monitor.extend(rng.sample(pool, min(args.monitor_per_class, len(pool))))
    if resuming:
        optimizer.load_state_dict(saved['optimizer'])
        epoch, next_batch, global_step = saved['epoch_index'], saved['next_batch'], saved['global_step']
        next_sample = min(saved.get('next_sample', next_batch * saved['immutable_config']['batch_size']),
                          len(data.splits['train']))
        next_batch = math.ceil(next_sample / args.batch_size)
        history, epoch_active_seconds = saved['history'], saved['epoch_active_seconds']
        sequence_meter = LossMeter(saved['sequence_meter'])
        classification_meter = LossMeter(saved['classification_meter'])
        best_key, best_primary = saved['best_key'], saved['best_primary']
        best_ce, no_improvement = saved['best_ce'], saved['epochs_without_improvement']
        baseline = saved.get('baseline_before_training')
        if args.baseline_before_training:
            if not isinstance(baseline, dict) or baseline.get('state') not in ('pending', 'completed'):
                raise ValueError('Baseline state is absent or invalid; do not skip initializer evaluation on resume.')
            if baseline['state'] == 'pending' and (epoch or next_sample or global_step or best_key is not None) and not args.rebaseline_on_resume:
                raise ValueError('Pending baseline cannot contain optimizer updates or a selected checkpoint.')
        random.setstate(saved['rng_python'])
        np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch'])
        torch.cuda.set_rng_state_all(saved['rng_cuda'])
        del saved
        steps_path = run / 'steps.jsonl'
        if steps_path.exists():
            valid = []
            for line in steps_path.read_text().splitlines():
                try:
                    if json.loads(line)['global_step'] <= global_step:
                        valid.append(line)
                except (ValueError, KeyError):
                    continue
            steps_path.write_text('\n'.join(valid) + ('\n' if valid else ''))
    if no_improvement >= args.patience:
        print('This run already early-stopped; retain its results and use a new experiment.', flush=True)
        return
    if epoch >= args.epochs:
        print('Requested epochs already completed.', flush=True)
        return
    config = {'immutable': immutable, 'arguments': vars(args),
               'batch_transition': migration,
               'baseline_before_training': baseline,
               'data_split_counts': counts, 'initialization': ('retain full optimizer/RNG and sample cursor across explicit rebatching' if migration else
                                                               'inherit model weights; fresh optimizer for a NEW experiment'),
               'best_selection': f'classifier: AUROC then BCE; joint/sequence: (single exact + {args.selection_double_priority:g} * double exact) / {1 + args.selection_double_priority:g}, then double exact, balanced sequence CE; optional single-drop budget={args.selection_max_single_drop} relative to initializer',
               'limitations': data.manifest.get('limitations', [])}
    write_json(run / 'config.json', config)
    write_json(run / 'validation_monitor.json', monitor)
    for filename in ('manifest.json', 'splits.json'):
        (run / ('data_' + filename)).write_bytes((data.path / filename).read_bytes())
    snapshots = run / 'source_snapshot'
    snapshots.mkdir(exist_ok=True)
    for name in sources:
        (snapshots / name).write_bytes((Path(__file__).parent / name).read_bytes())
    save_reports(run, history)
    stop = {'requested': False, 'reason': ''}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: stop.update(requested=True, reason=signal.Signals(signum).name))
    session_started = last_account = time.perf_counter()
    steps_per_epoch = math.ceil(len(data.splits['train']) / args.batch_size)

    def requested():
        if (run / 'STOP').exists():
            stop.update(requested=True, reason='STOP file')
        if args.session_minutes and time.perf_counter() - session_started >= args.session_minutes * 60:
            stop.update(requested=True, reason='session time limit')
        if args.architecture in COMPACT_ARCHITECTURES and shutil.disk_usage(run).free < 1024**3:
            stop.update(requested=True, reason='disk safety margin below 1 GiB')
        return stop['requested']

    def account_time():
        nonlocal last_account, epoch_active_seconds
        now = time.perf_counter()
        if baseline is not None and baseline['state'] == 'pending':
            baseline['active_seconds'] += now - last_account
        else:
            epoch_active_seconds += now - last_account
        last_account = now

    def status(state, **extra):
        write_json(run / 'status.json', {'state': state, 'task': args.task, 'epoch': epoch + 1,
                   'completed_epochs': epoch, 'next_batch': next_batch, 'global_step': global_step,
                   'next_sample': next_sample, 'epoch_spectra': len(data.splits['train']),
                   'epoch_progress_percent': 100 * next_sample / len(data.splits['train']),
                   'batch_size': args.batch_size,
                   'baseline_state': baseline['state'] if baseline is not None else 'disabled',
                   'steps_per_epoch': steps_per_epoch, 'pid': os.getpid(), **extra})

    def persist(best=False, ce_best=False, publish_baseline=False):
        account_time()
        checkpoint = {'format': FORMAT, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'immutable_config': immutable, 'run_directory': str(run),
                      'epoch_index': epoch, 'next_batch': next_batch, 'global_step': global_step,
                      'next_sample': next_sample, 'batch_transition': migration,
                      'history': history, 'epoch_active_seconds': epoch_active_seconds,
                      'sequence_meter': sequence_meter.state(), 'classification_meter': classification_meter.state(),
                      'best_key': best_key, 'best_primary': best_primary, 'best_ce': best_ce,
                      'baseline_before_training': baseline,
                      'epochs_without_improvement': no_improvement, 'training_exposure': training_exposure,
                      'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                      'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()}
        # Commit a completed baseline in last.pt only after both initializer
        # selection files are durable. An interrupted publication can safely
        # resume from the prior pending-baseline checkpoint and reevaluate.
        if not publish_baseline:
            atomic_checkpoint(run / 'last.pt', checkpoint)
        if best:
            selected = checkpoint
            if args.architecture in COMPACT_ARCHITECTURES:
                selected = {k: v for k, v in checkpoint.items() if k not in
                            ('optimizer', 'rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda')}
                selected['checkpoint_kind'] = 'model_and_provenance'
            atomic_checkpoint(run / 'best.pt', selected)
        if ce_best and args.architecture not in COMPACT_ARCHITECTURES:
            atomic_checkpoint(run / 'best_ce.pt', checkpoint)
        if publish_baseline:
            atomic_checkpoint(run / 'last.pt', checkpoint)
        if baseline is not None:
            config['baseline_before_training'] = baseline
            write_json(run / 'config.json', config)

    def evaluate_baseline():
        """Return only aggregate metrics; partial evaluation is restarted on resume."""
        status('validating_baseline', phase='full validation components')
        print('BASELINE: evaluating the initializer before any optimizer update...', flush=True)
        result = evaluate_components(model, data, data.splits['validation'], 'cuda',
                                     args.batch_size, args.task, requested)
        if result is None:
            return None
        validation, _ = result
        monitor_metrics = None
        if args.task != 'classifier':
            status('validating_baseline', phase='fixed validation free generation')
            print(f'Baseline free generation on fixed validation monitor ({len(monitor)} spectra)...', flush=True)
            generated = evaluate_sequences(model, data, monitor, 'cuda', min(args.batch_size, 8), requested)
            if generated is None:
                return None
            monitor_metrics, _ = generated
        return validation, monitor_metrics

    print(f'TASK={args.task}; data={data.path}; train={counts["train"]}; validation={counts["validation"]}', flush=True)
    print('Optimizer groups: ' + json.dumps([{'name': g['group_name'], 'lr': g['lr'],
          'parameters': sum(p.numel() for p in g['params'])} for g in optimizer.param_groups]), flush=True)
    print(f'Batches/epoch={steps_per_epoch}; routing={args.routing}; threshold={args.threshold}; '
          f'lambda={args.classification_weight}. No test performance is evaluated.', flush=True)
    if not resuming or args.rebatch_from:
        persist()
    if args.rebatch_from:
        parent_lock.close()
        print('BATCH TRANSITION: ' + json.dumps(migration), flush=True)
    status('training')
    if args.prepare_only:
        status('paused', reason='Batch transition prepared; no optimizer update performed.')
        print('PREPARED: complete state retained; ready for --resume.', flush=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        return
    try:
        if baseline is not None and baseline['state'] == 'pending':
            baseline_result = evaluate_baseline()
            if baseline_result is not None:
                validation, monitor_metrics = baseline_result
                single_floor = (monitor_metrics['single']['complete_set_match_rate'] - args.selection_max_single_drop
                                if args.selection_max_single_drop is not None else None)
                best_primary, best_key, best_ce = selection_values(
                    args.task, validation, monitor_metrics, args.selection_double_priority, single_floor)
                account_time()
                baseline.update(state='completed', epoch_index=epoch, global_step=global_step,
                                validation=validation, monitor_metrics=monitor_metrics,
                                selection_key=list(best_key), selection_primary=list(best_primary),
                                selection_ce=best_ce,
                                note='Initializer evaluated before updates; same selection rule and monitor as trained epochs.')
                persist(best=True, ce_best=True, publish_baseline=True)
                single_text = (f'{monitor_metrics["single"]["complete_set_match_rate"]:.6f}'
                               if monitor_metrics is not None else 'n/a')
                double_text = (f'{monitor_metrics["double"]["complete_set_match_rate"]:.6f}'
                               if monitor_metrics is not None else 'n/a')
                print(f'BASELINE SAVED | single exact={single_text} | double exact={double_text} | '
                      f'AUROC={validation["classification"]["auroc"]:.6f} | selection CE={best_ce:.6f} | '
                      + ('best.pt / last.pt at stage epoch 0' if args.architecture in COMPACT_ARCHITECTURES
                       else 'best.pt / best_ce.pt / last.pt at stage epoch 0'), flush=True)
                status('training')
            elif not requested():
                raise RuntimeError('Baseline evaluation did not finish; optimizer updates remain blocked.')
        while epoch < args.epochs and (baseline is None or baseline['state'] == 'completed'):
            order = list(data.splits['train'])
            random.Random(args.seed + epoch).shuffle(order)
            print(f'\nEpoch {epoch+1}/{args.epochs}, resume batch {next_batch+1}/{steps_per_epoch}', flush=True)
            model.train()
            if args.separation_only:
                model.eval()
                model.spectral_separation.train()
            if args.freeze_classifier:
                model.encoder.eval()
                model.classification_head.eval()
            while next_sample < len(order):
                if requested():
                    break
                started = time.perf_counter()
                batch_indices = order[next_sample:next_sample+args.batch_size]
                spectra, precursors, labels = data.batch(batch_indices, 'cuda')
                optimizer.zero_grad(set_to_none=True)
                sequence, classification, logits = model.loss_components(spectra, precursors, labels, args.task)
                loss = training_objective(
                    sequence, classification, getattr(model, 'sequence_regularizer', None),
                    labels, args.task, args.double_weight, args.classification_weight,
                    args.classification_double_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss.')
                loss.backward()
                if global_step == 0:
                    for group in optimizer.param_groups:
                        gradients = [p.grad for p in group['params'] if p.grad is not None]
                        if not gradients:
                            raise RuntimeError('No gradient in trainable group ' + group['group_name'])
                        norm = torch.stack([g.detach().norm() for g in gradients]).norm()
                        print(f'First-batch {group["group_name"]} gradient norm={float(norm):.6g}', flush=True)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError('Non-finite gradient; optimizer not updated.')
                if args.task == 'classifier':
                    optimizer.step()
                else:
                    optimizer_step_with_new_token_rate(optimizer, model, args.new_token_lr_factor)
                classification_meter.add(classification, labels)
                if sequence is not None:
                    sequence_meter.add(sequence, labels)
                next_sample += len(batch_indices)
                next_batch = math.ceil(next_sample / args.batch_size)
                global_step += 1
                torch.cuda.synchronize()
                if global_step % args.log_every == 0 or next_batch == steps_per_epoch:
                    item = {'epoch': epoch+1, 'batch': next_batch, 'global_step': global_step,
                            'batch_loss': float(loss.detach()),
                            'sequence_ce': float(sequence.mean().detach()) if sequence is not None else None,
                            'classification_bce': float(classification.mean().detach()),
                            'running_train_loss': (classification_meter if args.task == 'classifier' else sequence_meter).means()['all'],
                            'batch_seconds': time.perf_counter()-started, 'session_seconds': time.perf_counter()-session_started}
                    item.update(epoch_spectra_done=next_sample, epoch_spectra_total=len(order),
                                spectra_per_second=len(batch_indices) / item['batch_seconds'])
                    if args.architecture in COMPACT_ARCHITECTURES:
                        item['diversity_penalty'] = float(model.sequence_regularizer.mean().detach())
                    if args.architecture in ('dual_slot_evidence', 'dual_slot_separation'):
                        item['evidence_loss'] = float(model.evidence_loss.mean().detach())
                        item['diversity_penalty'] -= args.evidence_weight * item['evidence_loss']
                        if args.hard_peptide_weight:
                            item['hard_peptide_loss'] = float(model.hard_peptide_loss.mean().detach())
                            item['diversity_penalty'] -= item['hard_peptide_loss']
                    if args.architecture == 'dual_slot_separation':
                        item['allocation_loss'] = float(model.allocation_loss.mean().detach())
                        item['diversity_penalty'] -= .05 * item['allocation_loss']
                    with (run / 'steps.jsonl').open('a') as handle:
                        handle.write(json.dumps(item)+'\n')
                    print(f'Epoch {epoch+1}/{args.epochs} | {next_batch}/{steps_per_epoch} batches | '
                          f'step {global_step} | total={item["batch_loss"]:.4f} | seq={item["sequence_ce"]} | '
                          f'cls={item["classification_bce"]:.4f} | '
                          + (f'diversity={item["diversity_penalty"]:.4f} | ' if 'diversity_penalty' in item else '')
                          + (f'peak={item["evidence_loss"]:.4f} | ' if 'evidence_loss' in item else '')
                          + (f'allocation={item["allocation_loss"]:.4f} | ' if 'allocation_loss' in item else '')
                          + (f'hard_peptide={item["hard_peptide_loss"]:.4f} | ' if 'hard_peptide_loss' in item else '')
                          + f'{next_sample}/{len(order)} spectra '
                          f'({100*next_sample/len(order):.1f}%) | {item["spectra_per_second"]:.1f} spectra/s | '
                          f'session={item["session_seconds"]/60:.1f}min', flush=True)
                    status('training', **item)
                if global_step % args.save_every == 0:
                    persist()
            if requested():
                break
            status('validating')
            validation_result = evaluate_components(model, data, data.splits['validation'], 'cuda',
                                                    args.batch_size, args.task, requested)
            if validation_result is None:
                break
            validation, classification_rows = validation_result
            monitor_metrics = None
            if args.task != 'classifier':
                print(f'Free generation on fixed validation monitor ({len(monitor)} spectra)...', flush=True)
                generated = evaluate_sequences(model, data, monitor, 'cuda', min(args.batch_size, 8), requested)
                if generated is None:
                    break
                monitor_metrics, monitor_rows = generated
                write_json(run / f'validation_epoch_{epoch+1:03d}.json',
                           {'scope': 'fixed validation subset; not full validation sequence accuracy',
                            'metrics': monitor_metrics, 'rows': monitor_rows})
                single = monitor_metrics['single']['complete_set_match_rate']
                double = monitor_metrics['double']['complete_set_match_rate']
            single_floor = (baseline['monitor_metrics']['single']['complete_set_match_rate'] - args.selection_max_single_drop
                            if args.selection_max_single_drop is not None else None)
            primary, key, ce = selection_values(
                args.task, validation, monitor_metrics, args.selection_double_priority, single_floor)
            improved = best_key is None or key > best_key
            primary_improved = best_primary is None or primary > best_primary
            if improved:
                best_key = key
            if primary_improved:
                best_primary = primary
            no_improvement = 0 if primary_improved else no_improvement + 1
            ce_improved = ce < best_ce
            if ce_improved:
                best_ce = ce
            account_time()
            train_seq, train_cls = sequence_meter.means(), classification_meter.means()
            item = {'epoch': epoch+1, 'global_step': global_step,
                    'train_loss': train_cls['all'] if args.task == 'classifier' else train_seq['all'],
                    'validation_loss': validation['classification_bce']['all'] if args.task == 'classifier' else validation['sequence_ce']['all'],
                    'train_sequence_ce': train_seq['all'], 'train_classification_bce': train_cls['all'],
                    'validation_sequence_ce': validation['sequence_ce']['all'],
                    'validation_single_loss': validation['sequence_ce']['single'],
                    'validation_double_loss': validation['sequence_ce']['double'],
                    'validation_classification_bce': validation['classification_bce']['all'],
                    'validation_auroc': validation['classification']['auroc'],
                    'validation_double_recall': validation['classification']['double_recall'],
                    'validation_single_specificity': validation['classification']['single_specificity'],
                    'validation_balanced_accuracy': validation['classification']['balanced_accuracy'],
                    'validation_monitor_single_exact': single if args.task != 'classifier' else None,
                    'validation_monitor_double_exact': double if args.task != 'classifier' else None,
                    'epoch_minutes': epoch_active_seconds / 60, 'epochs_without_improvement': no_improvement}
            history.append(item)
            write_json(run / f'classification_epoch_{epoch+1:03d}.json',
                       {'scope': 'full validation classification', 'metrics': validation['classification'],
                        'spectra': len(classification_rows), 'rows_preview': classification_rows[:128],
                        'note': 'All spectra contribute to metrics; only 128 example rows retained per epoch to bound disk usage.'})
            print(f'EPOCH {epoch+1} DONE | train={item["train_loss"]:.5f} | val={item["validation_loss"]:.5f} | '
                  f'AUROC={item["validation_auroc"]:.4f} | minutes={item["epoch_minutes"]:.2f} | best={improved}', flush=True)
            if monitor_metrics:
                print('Validation generation: '+json.dumps(monitor_metrics), flush=True)
            epoch += 1
            next_batch, next_sample, epoch_active_seconds = 0, 0, 0.0
            sequence_meter, classification_meter = LossMeter(), LossMeter()
            persist(best=improved, ce_best=ce_improved)
            save_reports(run, history)
            last_account = time.perf_counter()
            if no_improvement >= args.patience:
                status('early_stopped', reason='Validation selection metric plateaued; full state saved.')
                print('EARLY STOPPED: checkpoints saved. Instance remains powered on.', flush=True)
                break
            status('training' if epoch < args.epochs else 'completed')
        if requested():
            persist()
            status('paused', reason=stop['reason'])
            print('PAUSED: full state saved to '+str(run / 'last.pt'), flush=True)
        elif no_improvement < args.patience:
            status('completed')
            print('COMPLETED: best.pt / best_ce.pt / last.pt saved.', flush=True)
    except Exception as error:
        status('failed', error=repr(error), recovery='Resume the last durable checkpoint; unsaved batches may repeat.')
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
