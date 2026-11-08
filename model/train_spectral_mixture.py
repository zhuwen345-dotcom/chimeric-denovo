"""Explicit curriculum: intensity teacher, single adaptation, joint mixture.

Importing this file never trains. It does not alter 05/06, existing checkpoints,
or current shutdown policies. No external MS2 model is silently downloaded.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import sys
import time

os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from spectral_mixture import FragmentIntensityModel,MixtureObjective,intensity_training_loss,peptide_mass


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def active_workers(root):
    """Local copy of the process check; inten never imports dual_slot helpers."""
    root=str(Path(root).resolve()).encode()
    names=(b'train_resumable.py',b'correction_pilot.py',b'train_real_correction.py',
           b'train_joint_correction.py',b'joint_input_trials.py',b'train_spectral_mixture.py',
           b'draft_fit.py',b'train_cond.py',b'train_precursor.py',b'eval_cond',b'dump_candidates.py')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command=path.read_bytes()
            if b'input_trials_guard.py' in command:continue
            if root in command and b'python' in command and any(n in command for n in names):yield path.parent.name
        except OSError:pass


def record_peptide_keys(record):
    return set(record.get('split_peptide_keys') or [re.sub('[^A-Z]','',p).replace('I','L') for p in record['model_labels']])


def collect_exposure(records,indices):
    peptides,spectra,raws=set(),set(),set()
    for i in indices:
        record=records[i];peptides.update(record_peptide_keys(record));raws.add(record['raw_file'])
        spectra.add(json.dumps([record['raw_file'],str(record['scan'])]))
    return dict(peptides=sorted(peptides),spectra=sorted(spectra),raw_files=sorted(raws))


BASE_FILES=('dual_slot.py','model.py','separator_base.py','evaluate_holdout.py',
            'workflow.py','training_common.py','train_resumable.py')


def capture_sources(code,baseline,destination):
    """Every experiment snapshots its own complete model code, including the
    actual encoder/decoder package. No sibling architecture is imported.
    """
    files=[(code/name,Path(name)) for name in ('spectral_mixture.py','spectral_mixture_joint.py','train_spectral_mixture.py','intensity_comparison.py','peak_attention.py')]
    files += [(baseline/name,Path(name)) for name in BASE_FILES]
    for source in (code/'backbone').rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts and source.suffix!='.pyc':
            files.append((source,Path('backbone')/source.relative_to(code/'backbone')))
    if not (code/'backbone/casanovo/denovo/model.py').is_file() or not (code/'backbone/depthcharge').is_dir():
        raise RuntimeError('Missing inten backbone source bundle; do not silently import the other architecture.')
    for source,relative in files:
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(source,target)
        elif sha(source)!=sha(target):raise RuntimeError('Source changed: '+str(relative))
    expected={relative.as_posix():sha(destination/relative) for _,relative in files}
    actual={p.relative_to(destination).as_posix():sha(p) for p in destination.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if actual!=expected:raise RuntimeError('Unexpected files in the source snapshot.')
    return expected


class ArrayData:
    def __init__(self,path):
        self.path=Path(path).resolve()
        self.manifest=json.loads((self.path/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('ready',True) is not True:raise ValueError('Data not ready.')
        for name,digest in self.manifest['files_sha256'].items():
            if sha(self.path/name)!=digest:raise ValueError('Data changed: '+name)
        self.splits=json.loads((self.path/'splits.json').read_text(encoding='utf-8'))
        with (self.path/'records.jsonl').open(encoding='utf-8') as f:self.records=[json.loads(row) for row in f]
        self.peaks=np.load(self.path/'peaks.npy',mmap_mode='r')
        self.lengths=np.load(self.path/'lengths.npy',mmap_mode='r')
        self.precursors=np.load(self.path/'precursors.npy',mmap_mode='r')
        if not len(self.records)==len(self.peaks)==len(self.lengths)==len(self.precursors):raise ValueError('Data array mismatch.')
        groups=[set(self.splits[k]) for k in ('train','validation','test')]
        if any(groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2))):raise ValueError('Split index leakage.')
        policy=self.manifest.get('split_policy','raw_disjoint')
        if policy == 'peptide_disjoint':
            units=[{key for i in group for key in record_peptide_keys(self.records[i])} for group in groups]
        elif policy == 'raw_disjoint' or policy.startswith('Seeded within-species RAW split'):
            units=[{self.records[i]['raw_file'] for i in group} for group in groups]
        else:raise ValueError('Unknown explicit split policy: '+policy)
        if any(units[a]&units[b] for a,b in ((0,1),(0,2),(1,2))):raise ValueError(policy+' split leakage.')

    def batch(self,ids,device):
        ids=np.asarray(ids,dtype=np.int64);length=int(self.lengths[ids].max())
        return (torch.from_numpy(np.array(self.peaks[ids,:length],copy=True)).to(device),
            torch.from_numpy(np.array(self.precursors[ids],copy=True)).to(device),
            [self.records[i]['model_labels'] for i in ids])


def load_joint(root,path,source_dir,*,peak_routing=None,attention_initial_strength=.1,attention_max_bias=4.):
    for name in ('dual_slot','model','separator_base','peak_attention'):
        loaded=sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve()!=(source_dir/(name+'.py')).resolve():
            raise RuntimeError('A different architecture is already imported; use a fresh Python process/kernel for inten.')
    sys.path[:0]=[str(source_dir),str(source_dir/'backbone'),str(Path(__file__).resolve().parent/'backbone')]
    from dual_slot import EvidenceDualSlotModel
    ck=torch.load(str(path),map_location='cpu',weights_only=False)
    cfg=ck.get('model_config',ck.get('joint_immutable_config',ck.get('immutable_config')))
    if cfg is None:raise ValueError('Missing model config in initializer.')
    saved_routing=cfg.get('peak_routing','feature')
    selected_routing=saved_routing if peak_routing is None else peak_routing
    if selected_routing not in ('feature','attention'):raise ValueError('Unknown peak routing.')
    if saved_routing=='attention' and selected_routing!='attention':
        raise ValueError('An attention checkpoint cannot silently resume with feature scaling.')
    model_class=EvidenceDualSlotModel;extra={}
    if selected_routing=='attention':
        from peak_attention import AttentionEvidenceDualSlotModel
        model_class=AttentionEvidenceDualSlotModel
        if cfg.get('fragment_suffix_v1'):
            from fragment_attention import FragmentAttentionModel
            model_class=FragmentAttentionModel
        if cfg.get('conditional_v1'):
            from conditional_residual import ConditionalResidualModel
            model_class=ConditionalResidualModel
        extra=dict(attention_initial_strength=cfg.get('attention_initial_strength',attention_initial_strength),
                   attention_max_bias=cfg.get('attention_max_bias',attention_max_bias))
        if cfg.get('conditional_v1'):
            extra['conditional_weight']=cfg.get('conditional_weight',.75)
    base=root/'model/architecture.pt'
    if not base.exists():base=root/'data/base/source/Model Checkpoints/casanovo_massivekb.ckpt'
    model=model_class(base,routing='hard',threshold=.5,evaluation_beam_size=4,
        diversity_weight=cfg.get('diversity_weight',.5),diversity_margin=cfg.get('diversity_margin',.2),
        evidence_weight=cfg.get('evidence_weight',.1),hard_peptide_weight=cfg.get('hard_peptide_weight',0.),**extra)
    state=ck.get('joint_state_dict',ck.get('state_dict',ck.get('model_state')))
    if state is None:raise ValueError('Missing sequencing weights.')
    if selected_routing=='attention' and saved_routing!='attention':
        # Explicit new-stage warm start: inherit every old tensor; initialize
        # exactly one new scalar. Actual attention resumes remain fully strict.
        if any(k in state for k in ('peak_attention_log_scale','peak_attention_progress')):
            raise ValueError('Unidentified attention parameters.')
        state=dict(state,peak_attention_log_scale=model.peak_attention_log_scale.detach().clone(),
                   peak_attention_progress=model.peak_attention_progress.detach().clone())
    if selected_routing=='attention':
        cfg=dict(cfg,peak_routing='attention',**extra)
    if hasattr(model, 'rt_head'):
        state = dict(state)
        for key, value in model.rt_head.state_dict().items():
            state.setdefault('rt_head.' + key, value)
    model.load_state_dict(state,strict=True)
    if cfg.get('fragment_suffix_v1'):
        model.output_policy=cfg.get('output_policy')
    model.requires_grad_(True);model.freeze_encoder=False
    return model,cfg,ck.get('training_exposure',{})


def load_trained_mixture(root,checkpoint,source_dir,device='cpu'):
    """Standalone inference from a trained full checkpoint, without labels or
    the old teacher checkpoint. Retain the official Casanovo construction file
    and source snapshot, as for our other models.
    """
    ck=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
    if ck.get('format')!='mixture_joint_v1':raise ValueError('Not a mixture-joint checkpoint.')
    model,_,_=load_joint(Path(root),checkpoint,Path(source_dir))
    model.threshold=ck['immutable']['settings'].get('threshold',.5)
    teacher=FragmentIntensityModel(**ck['intensity_model_config'])
    objective=MixtureObjective(teacher,**ck['objective_config'])
    objective.load_state_dict(ck['objective_state'],strict=True)
    return model.to(device).eval(),objective.to(device).eval()


def choose_indices(data,args):
    train=list(data.splits['train'])
    if args.panel_from:
        panel_cfg=json.loads(args.panel_from.read_text(encoding='utf-8'))
        panel=panel_cfg.get('panel',panel_cfg.get('settings',{}).get('panel'))
        if panel is None:raise ValueError('Panel configuration does not contain a fixed validation panel.')
        if not set(panel)<=set(data.splits['validation']):raise ValueError('Panel belongs to a different dataset/split.')
    else:
        groups=[[i for i in data.splits['validation'] if len(data.records[i]['model_labels'])==n] for n in (1,2)]
        panel=sum([random.Random(args.seed+n).sample(g,min(256,len(g))) for n,g in enumerate(groups)],[])
    if args.mode=='pretrain':
        train=[i for i in train if len(data.records[i]['model_labels'])==1]
    elif args.mode in ('finetune','joint'):
        train=double_replay_indices(data.records,train,args.single_replay_fraction,args.seed)
    if args.mode=='intensity':
        def eligible(i):
            labels=data.records[i]['model_labels'];z=float(data.precursors[i,1])
            if len(labels)!=1 or int(z)!=z or not 2<=z<=5:return False
            mass=peptide_mass(labels[0]);observed=float(data.precursors[i,0])
            return min(abs(observed-mass-k*1.00335483507) for k in (0,1,2))<=max(.02,mass*20e-6)
        train=[i for i in train if eligible(i)]
        # Use different validation spectra from the joint model's fixed panel.
        excluded=set(panel)
        panel=[i for i in data.splits['validation'] if i not in excluded and eligible(i)]
        if len(panel)<128:raise ValueError('Need >=128 separate single-label validation spectra for the intensity teacher.')
        panel=random.Random(args.seed).sample(panel,min(args.teacher_validation,len(panel)))
    if args.max_train:
        train=random.Random(args.seed).sample(train,min(args.max_train,len(train)))
    if not train or not panel:raise ValueError('Empty training/validation selection.')
    return train,panel


def double_replay_indices(records,indices,fraction,seed):
    """All training doubles plus a fixed, unique single-label replay subset.

    The subset is locked into provenance; per-epoch shuffling changes order,
    not membership. No validation/test spectrum is borrowed or duplicated.
    """
    if not 0<fraction<.5:raise ValueError('Single replay fraction must be in (0,0.5).')
    doubles=[i for i in indices if len(records[i]['model_labels'])==2]
    singles=[i for i in indices if len(records[i]['model_labels'])==1]
    if not doubles:raise ValueError('Need double-label training spectra.')
    count=round(len(doubles)*fraction/(1-fraction))
    if count<1 or count>len(singles):raise ValueError('Not enough unique single-label training spectra for replay.')
    return doubles+random.Random(seed+901).sample(singles,count)


def augment_spectra(x,generator,probability=0.,jitter=.2,dropout=.02,min_keep=16):
    """Training-only, independent RNG; no added peaks, labels or precursors.

    Mild intensity jitter and optional uniform peak dropout. Preserve at least
    min_keep real peaks; these are configurable hypotheses, not required noise.
    Padding stays exactly zero. Returned intensities remain sqrt/L2 space.
    """
    if probability==0 or (jitter==0 and dropout==0):return x
    out=x.clone();valid=x[...,1]>0
    apply=(torch.rand((len(x),1),generator=generator)<probability).to(x.device)
    factors=1+jitter*(2*torch.rand(valid.shape,generator=generator)-1).to(x.device)
    out[...,1]*=torch.where(apply & valid,factors,torch.ones_like(factors))
    if dropout:
        draws=torch.rand(valid.shape,generator=generator).to(x.device)
        remove=(draws<dropout)&valid&apply
        for i in range(len(x)):
            maximum=max(0,int(valid[i].sum())-min_keep)
            chosen=torch.nonzero(remove[i],as_tuple=False).flatten()
            if len(chosen)>maximum:
                remove[i]=False
                remove[i,chosen[torch.argsort(draws[i,chosen])[:maximum]]]=True
        out[remove]=0
    norms=out[...,1].norm(dim=1,keepdim=True)
    if torch.any(norms<=0):raise RuntimeError('Augmentation erased a spectrum; no update')
    out[...,1]/=norms
    return out


@torch.no_grad()
def evaluate_intensity(model,data,panel,device,batch,stopped=lambda:False):
    model.eval();values=[];baseline=[]
    for a in range(0,len(panel),batch):
        if stopped():return None
        x,p,labels=data.batch(panel[a:a+batch],device)
        _,stats=intensity_training_loss(model,x,[v[0] for v in labels],p[:,1].tolist())
        take=stats['eligible'];values+=stats['cosine'][take].cpu().tolist();baseline+=stats['uniform_cosine'][take].cpu().tolist()
    median=float(np.median(values)) if values else 0.;uniform=float(np.median(baseline)) if baseline else 0.
    # Exploratory acceptance gate, not proof of physically calibrated intensities.
    return dict(median_cosine=median,uniform_median_cosine=uniform,spectra=len(values),
        scope='sqrt-intensity cosine on retained theoretically matched peaks, not full-spectrum accuracy',
        qualified_for_experiment=len(values)>=128 and median>=.7 and median>=uniform+.01)


@torch.no_grad()
def evaluate_joint(model,objective,data,panel,device,run,heartbeat,stopped=lambda:False,batch=16,balanced_score=False):
    from spectral_mixture_joint import predict_mixtures
    from workflow import classification_metrics
    model.eval()
    if objective is not None:objective.eval()
    counts={str(n):dict(n=0,original=0,candidate_only=0,mixture=0) for n in (1,2)}
    rows=[];ce=[];truth=[];probabilities=[]
    for a in range(0,len(panel),batch):
        if stopped():return None
        heartbeat('validating',validation_done=a,validation_total=len(panel))
        ids=panel[a:a+batch];x,p,labels=data.batch(ids,device)
        sequence,_,logits=model.loss_components(x,p,labels,'joint')
        ce+=sequence.detach().cpu().tolist()
        probabilities+=torch.sigmoid(logits).detach().cpu().reshape(-1).tolist()
        truth += [int(len(y)==2) for y in labels]
        original=model.generate(x,p,beam_size=4)
        predicted=(predict_mixtures(model,objective,x,p,beam=4) if objective is not None else
                   [dict(v,candidate_only_peptides=v['peptides']) for v in original])
        for i,y,b,v in zip(ids,labels,original,predicted):
            c=counts[str(len(y))];c['n']+=1
            c['original']+=int(set(b['peptides'])==set(y))
            c['candidate_only']+=int(set(v.get('candidate_only_peptides',[]))==set(y))
            c['mixture']+=int(set(v['peptides'])==set(y))
            rows.append(dict(index=i,labels=y,original=b['peptides'],**v))
    write(run/'validation.json',dict(counts=counts,rows=rows,scope='Validation only; no labels in generation/fitting/ranking.'))
    score=(counts['1']['original']/max(1,counts['1']['n']) if objective is None and not balanced_score else
           sum(c['mixture']/max(1,c['n']) for c in counts.values())/2)
    return dict(counts=counts,score=score,
                scoring=('mixed single/double mean; no intensity scoring' if balanced_score else
                         'single adaptation; no intensity scoring' if objective is None else 'mixed single/double mean'),
                sequence_ce=float(np.mean(ce)),classification=classification_metrics(truth,probabilities,model.threshold))


def tune_microbatch(model,data,train,args,device,optimizer,heartbeat):
    """Time real forward/backward without updating parameters or consuming RNG.

    Reserve Adam's two FP32 moments and use the longest training spectra.
    No validation/test labels are used to select throughput settings.
    """
    if device.type!='cuda':return min(args.batch,8)
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    training=model.training;model.train();results=[]
    array=np.asarray(train,dtype=np.int64)
    positions=np.argsort(np.asarray(data.lengths[array]))[-min(args.batch,len(array)):]
    ids=array[positions].tolist()
    reserve=torch.empty(8*sum(p.numel() for g in optimizer.param_groups for p in g['params']),
                        dtype=torch.uint8,device=device)
    candidates=[n for n in (8,16,32,64,96,128) if n<=args.batch]
    if args.batch not in candidates:candidates.append(args.batch)
    def clear():
        optimizer.zero_grad(set_to_none=True)
        for name in ('sequence_regularizer','evidence_loss','hard_peptide_loss','matched_direct_assignment','conditional_sequence_ce'):
            if hasattr(model,name):setattr(model,name,None)
        import gc
        gc.collect();torch.cuda.empty_cache()
    def step(width):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0,len(ids),width):
            x,p,labels=data.batch(ids[start:start+width],device)
            with torch.autocast('cuda',dtype=torch.float16):
                if args.mode=='intensity':
                    value,stats=intensity_training_loss(model,x,[y[0] for y in labels],p[:,1].tolist())
                    loss=(value*stats['eligible']).sum()/len(ids)
                else:
                    seq,bce,_=model.loss_components(x,p,labels,'joint')
                    loss=(seq+model.sequence_regularizer).sum()/len(ids)
                    if args.classifier_epoch_samples or args.mode in ('finetune','joint'):
                        loss=loss+args.classification_weight*bce.sum()/len(ids)
            (loss*128).backward()
        torch.cuda.synchronize()
    try:
        for width in candidates:
            try:
                clear();torch.cuda.reset_peak_memory_stats();step(width)
                began=time.perf_counter();step(width)
                elapsed=time.perf_counter()-began;peak=torch.cuda.max_memory_allocated()
                if peak>torch.cuda.get_device_properties(device).total_memory*.9:
                    results.append(dict(microbatch=width,rejected='less than 10 percent memory reserve'));break
                results.append(dict(microbatch=width,spectra_per_second=len(ids)/elapsed,peak_gib=peak/1024**3))
                heartbeat('tuning',benchmark=results)
                print('MICROBATCH_BENCHMARK '+json.dumps(results[-1]),flush=True)
            except torch.cuda.OutOfMemoryError:
                clear();results.append(dict(microbatch=width,rejected='CUDA OOM; no updates performed'));break
        accepted=[x for x in results if 'spectra_per_second' in x]
        if not accepted:raise RuntimeError('Even microbatch8 failed; lower effective/microbatch explicitly.')
        best=max(accepted,key=lambda x:x['spectra_per_second'])
        print('MICROBATCH_SELECTED '+json.dumps(best),flush=True)
        return best['microbatch']
    finally:
        clear();del reserve;torch.cuda.empty_cache()
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng);model.train(training)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('intensity','pretrain','finetune','joint'))
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--data',type=Path,default=Path('data/multi/800'))
    ap.add_argument('--initial',type=Path)
    ap.add_argument('--source-dir',type=Path)
    ap.add_argument('--teacher',type=Path)
    ap.add_argument('--panel-from',type=Path)
    ap.add_argument('--run',type=Path,default=Path('runs/inten'))
    ap.add_argument('--stage-name',choices=('intensity_full','joint_full','joint_attention'))
    ap.add_argument('--peak-routing',choices=('feature','attention'),default='feature')
    ap.add_argument('--attention-initial-strength',type=float,default=.1)
    ap.add_argument('--attention-max-bias',type=float,default=4.)
    ap.add_argument('--attention-transition-steps',type=int,default=2000)
    ap.add_argument('--ranking-cache',type=Path)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--minutes',type=float,default=120)
    ap.add_argument('--patience',type=int,default=5)
    ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--microbatch',type=int,default=32)
    ap.add_argument('--lr',type=float,default=1e-5)
    ap.add_argument('--encoder-lr',type=float,default=1e-6)
    ap.add_argument('--teacher-lr',type=float,default=1e-3)
    ap.add_argument('--classifier-lr',type=float,default=3e-5)
    ap.add_argument('--classifier-epoch-samples',type=int,default=0)
    ap.add_argument('--classification-weight',type=float,default=.1)
    ap.add_argument('--validation-batch',type=int,default=16)
    ap.add_argument('--length-bucketing',type=int,choices=(0,1),default=1)
    ap.add_argument('--fused-adamw',type=int,choices=(0,1),default=1)
    ap.add_argument('--teacher-validation',type=int,default=2048)
    ap.add_argument('--max-train',type=int,default=0)
    ap.add_argument('--single-replay-fraction',type=float,default=.15)
    ap.add_argument('--rank-every',type=int,default=16)
    ap.add_argument('--rank-samples',type=int,default=4)
    ap.add_argument('--mixture-weight',type=float,default=.1)
    ap.add_argument('--seed',type=int,default=454)
    ap.add_argument('--threshold',type=float,default=.5)
    ap.add_argument('--augment-probability',type=float,default=0.)
    ap.add_argument('--intensity-jitter',type=float,default=.2)
    ap.add_argument('--peak-dropout',type=float,default=.02)
    ap.add_argument('--min-keep-peaks',type=int,default=16)
    args=ap.parse_args();root=args.root.resolve()
    for key in ('data','initial','source_dir','teacher','panel_from','run','ranking_cache'):
        value=getattr(args,key)
        if value is not None:setattr(args,key,value.resolve() if value.is_absolute() else (root/value).resolve())
    if args.source_dir is None:args.source_dir=Path(__file__).resolve().parent
    if args.mode!='intensity' and args.initial is None:
        args.initial=args.run/('pretrain/best.pt' if args.mode in ('finetune','joint') else 'initialization.pt')
        if args.mode in ('pretrain','joint') and (args.run/'finetune/best.pt').exists():args.initial=args.run/'finetune/best.pt'
        if args.mode=='joint':
            lineage=args.run/'pretrain/curve_history.json'
            if lineage.exists() and any(s.get('stage')=='finetune' for s in json.loads(lineage.read_text())['inherited_stages']):
                args.initial=args.run/'pretrain/best.pt'
    if min(args.epochs,args.patience,args.microbatch,args.classifier_epoch_samples)<0 or min(args.batch,args.rank_every,args.rank_samples,args.validation_batch)<1:raise ValueError('Invalid counts; epochs/patience=0 disables limits; microbatch=0 tunes automatically.')
    if (not math.isfinite(args.minutes) or args.minutes<=0 or args.max_train<0
            or not math.isfinite(args.mixture_weight) or args.mixture_weight<0):raise ValueError('Invalid budget/weight.')
    if not all(math.isfinite(v) and v>0 for v in (args.lr,args.encoder_lr,args.teacher_lr,args.classifier_lr)):
        raise ValueError('Learning rates must be positive and finite.')
    if (not 0<args.threshold<1 or not 0<=args.augment_probability<=1 or
            not 0<=args.intensity_jitter<1 or not 0<=args.peak_dropout<1 or args.min_keep_peaks<1):
        raise ValueError('Invalid threshold or augmentation settings')
    if not math.isfinite(args.classification_weight) or args.classification_weight<0:raise ValueError('Invalid classification weight.')
    if not 0<args.single_replay_fraction<.5:raise ValueError('Single replay fraction must be in (0,0.5).')
    if args.stage_name and args.mode!=dict(intensity_full='intensity',joint_full='joint',joint_attention='joint')[args.stage_name]:
        raise ValueError('Stage name does not match its training mode.')
    if (not math.isfinite(args.attention_initial_strength) or args.attention_initial_strength<=0 or
            not math.isfinite(args.attention_max_bias) or args.attention_max_bias<=0 or args.attention_transition_steps<1):
        raise ValueError('Attention settings must be positive and finite.')
    if args.peak_routing=='attention' and args.stage_name!='joint_attention':
        raise ValueError('Use joint_attention for the changed decoder route; never overwrite an old stage.')
    if args.stage_name=='joint_attention' and args.peak_routing!='attention':
        raise ValueError('joint_attention requires --peak-routing attention.')
    if args.ranking_cache and args.mode!='intensity':
        raise ValueError('Ranking cache is only for the intensity predictor.')
    run=args.run/(args.stage_name or args.mode);run.mkdir(parents=True,exist_ok=True)
    if (run/'config.json').exists():
        old_config=json.loads((run/'config.json').read_text(encoding='utf-8'))
        if old_config.get('settings',{}).get('mode')!=args.mode:
            raise RuntimeError('This directory belongs to another experiment; no files overwritten.')
        if args.microbatch==0:args.microbatch=old_config['settings']['microbatch']
    elif any((run/name).exists() for name in ('best.pt','last.pt')):
        raise RuntimeError('Unidentified existing weights; choose a new experiment directory.')
    if (run/'last.pt').exists() and not args.resume:raise RuntimeError('Use --resume; no overwrite.')
    if (run/'status.json').exists() and json.loads((run/'status.json').read_text(encoding='utf-8')).get('state')=='early_stopped':
        raise RuntimeError('This stage early-stopped; inspect its validation before repeating the same recipe.')
    if [p for p in active_workers(root) if int(p)!=os.getpid()]:raise RuntimeError('Pause other training first. This script does not stop it.')
    if os.name!='nt':
        import fcntl
        lock=(run/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (run/'STOP').unlink(missing_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:(run/'STOP').touch())
    signal.signal(signal.SIGINT,lambda *_:(run/'STOP').touch())
    torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed)
    augmentation_rng=torch.Generator(device='cpu').manual_seed(args.seed+7311)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU unavailable; CPU requires explicit --device cpu.')
    started=time.monotonic();elapsed_base=0.;epoch=cursor=step=0;history=[];best=-math.inf;stall=0
    epoch_loss_sum=0.;epoch_samples=0
    classifier_seen_epoch=classifier_seen_total=last_epoch_classifier_count=0
    def status(state,**extra):write(run/'status.json',dict(state=state,pid=os.getpid(),epoch=epoch+1,cursor=cursor,
        step=step,elapsed_minutes=elapsed_base+(time.monotonic()-started)/60,
        session_elapsed_minutes=(time.monotonic()-started)/60,batch=args.batch,microbatch=args.microbatch,
        classifier_seen_epoch=classifier_seen_epoch,classifier_seen_total=classifier_seen_total,
        classifier_epoch_limit=args.classifier_epoch_samples if args.mode=='pretrain' else None,**extra))
    def stopped():return (run/'STOP').exists() or time.monotonic()-started>=args.minutes*60
    def failed(kind,value,tb):status('failed',error=str(value));sys.__excepthook__(kind,value,tb)
    sys.excepthook=failed
    status('preparing')
    data=ArrayData(args.data);train,panel=choose_indices(data,args)
    class_counts={n:sum(len(data.records[i]['model_labels'])==n for i in train) for n in (1,2)}
    if args.mode in ('finetune','joint') and (not all(class_counts.values()) or
            not all(any(len(data.records[i]['model_labels'])==n for i in panel) for n in (1,2))):
        raise ValueError('Joint experiment requires both single/double training and validation samples.')
    if args.mode=='pretrain' and class_counts[2]:raise ValueError('Single adaptation must not contain doubles.')
    sources=run/'source_snapshot';sources.mkdir(exist_ok=True)
    source_hashes=capture_sources(Path(__file__).resolve().parent,args.source_dir,sources)
    model_cfg={};exposure={};teacher_model=None;objective=None
    if args.mode=='intensity':
        model=FragmentIntensityModel().to(device);objective=None
        if args.initial is not None:
            captured=run/'initialization.pt'
            if not captured.exists():shutil.copy2(args.initial,captured)
            ck=torch.load(str(captured),map_location='cpu',weights_only=False,mmap=True)
            if ck.get('format')!='mixture_intensity_v1' or ck.get('model_config')!=model.config:
                raise ValueError('Intensity warm start must be a compatible intensity checkpoint.')
            if ck.get('training_data_sha256')!=sha(data.path/'manifest.json'):
                raise ValueError('Intensity warm start belongs to another dataset.')
            heldout=collect_exposure(data.records,data.splits['validation']+data.splits['test'])
            previous_exposure=ck.get('training_exposure',{})
            if any(set(previous_exposure.get(k,[]))&set(heldout[k]) for k in ('spectra','peptides')):
                raise ValueError('Intensity initializer overlaps heldout data.')
            model.load_state_dict(ck['model_state'],strict=True)
            del ck
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.teacher_lr,weight_decay=.01,fused=bool(args.fused_adamw and device.type=='cuda'))
        exposure=collect_exposure(data.records,train)
    else:
        if args.initial is None or args.source_dir is None:raise ValueError('joint requires --initial and --source-dir.')
        captured=run/'initialization.pt'
        if not captured.exists():shutil.copy2(args.initial,captured)
        model,model_cfg,exposure=load_joint(root,captured,sources,peak_routing=args.peak_routing,
            attention_initial_strength=args.attention_initial_strength,attention_max_bias=args.attention_max_bias)
        model.to(device)
        model.threshold=args.threshold
        model_cfg=dict(model_cfg,threshold=args.threshold)
        checked=('spectra','peptides') if data.manifest.get('split_policy')=='peptide_disjoint' else ('spectra','raw_files')
        heldout=collect_exposure(data.records,data.splits['validation']+data.splits['test'])
        if any(set(exposure.get(k,[]))&set(heldout[k]) for k in checked):raise ValueError('Initializer overlaps heldout '+str(checked))
        exposure=dict(exposure)
        train_exposure=collect_exposure(data.records,train)
        for k,v in train_exposure.items():exposure[k]=sorted(set(exposure.get(k,[]))|set(v))
        teacher_path=args.teacher or args.run/'intensity/best.pt'
        if args.mode=='joint':
            ck=torch.load(str(teacher_path),map_location='cpu',weights_only=False)
            if ck.get('format')!='mixture_intensity_v1' or not ck.get('metrics',{}).get('qualified_for_experiment'):
                raise RuntimeError('Intensity teacher has not passed held-out validation; random/unvalidated intensities are not accepted.')
            if ck.get('training_data_sha256')!=sha(data.path/'manifest.json'):
                raise ValueError('Teacher belongs to another dataset/profile; retrain it on this input profile.')
            teacher_exposure=ck.get('training_exposure') or {'raw_files':ck.get('training_raw_files',[])}
            if any(set(teacher_exposure.get(k,[]))&set(heldout[k]) for k in checked):raise ValueError('Teacher overlaps heldout '+str(checked))
            teacher_model=FragmentIntensityModel(**ck['model_config']);teacher_model.load_state_dict(ck['model_state'],strict=True)
            objective=MixtureObjective(teacher_model).to(device)
            if args.stage_name in ('joint_full','joint_attention'):
                previous=torch.load(str(captured),map_location='cpu',weights_only=False,mmap=True)
                if previous.get('format')!='mixture_joint_v1':raise ValueError('joint_full must inherit a trained joint checkpoint.')
                with torch.no_grad():objective.raw_weight.copy_(previous['objective_state']['raw_weight'].to(device))
                del previous
        elif args.mode=='pretrain':
            model.classifier_features_detached=True
            model.classification_head.requires_grad_(args.classifier_epoch_samples>0)
            objective=None
        encoder=list(model.encoder.parameters());other=[p for n,p in model.named_parameters() if not n.startswith(('encoder.','classification_head.'))]
        other=[p for p in other if p.requires_grad]
        groups=[dict(params=encoder,lr=args.encoder_lr,name='encoder'),dict(params=other,lr=args.lr,name='sequencer')]
        if args.mode in ('finetune','joint') or args.classifier_epoch_samples>0:
            groups.append(dict(params=list(model.classification_head.parameters()),lr=args.classifier_lr,name='classifier',weight_decay=0.))
        if objective is not None:groups.append(dict(params=[objective.raw_weight],lr=args.lr))
        optimizer=torch.optim.AdamW(groups,weight_decay=.01,fused=bool(args.fused_adamw and device.type=='cuda'))
    for group in optimizer.param_groups:group.setdefault('target_lr',group['lr'])
    if args.microbatch==0:
        transition=getattr(model,'peak_attention_progress',None)
        if transition is not None:
            original_transition=float(transition)
            model.set_attention_progress(1.)
        try:args.microbatch=tune_microbatch(model,data,train,args,device,optimizer,status)
        finally:
            if transition is not None:model.set_attention_progress(original_transition)
    if args.microbatch>args.batch:raise ValueError('Microbatch exceeds effective batch')
    settings={k:v for k,v in vars(args).items() if k not in ('epochs','minutes','patience','resume')}
    settings={k:str(v) if isinstance(v,Path) else v for k,v in settings.items()}
    immutable=dict(settings=settings,data_sha=sha(data.path/'manifest.json'),train_indices=train,panel=panel,
        sources=source_hashes,intensity_space='sqrt_l2_then_squared_to_linear',
        initial_sha=sha(run/'initialization.pt') if (run/'initialization.pt').exists() else None,
        teacher_sha=sha(args.teacher or args.run/'intensity/best.pt') if args.mode=='joint' else None)
    if args.ranking_cache is not None:immutable['ranking_cache_sha256']=sha(args.ranking_cache)
    immutable['classifier_policy']=('balanced single/double BCE on every training spectrum; encoder and sequencing train together; no intensity teacher'
        if args.mode=='finetune' else 'first N shuffled singles per pretrain epoch; BCE head only via detached encoder; frozen batches have no head gradients or optimizer momentum/decay')
    if (run/'config.json').exists() and json.loads((run/'config.json').read_text(encoding='utf-8'))!=immutable:raise RuntimeError('Configuration/data/source mismatch.')
    write(run/'config.json',immutable)
    amp=device.type=='cuda';dtype=torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.cuda.amp.GradScaler(enabled=amp and dtype==torch.float16,init_scale=1024.)
    if args.resume:
        ck=torch.load(str(run/'last.pt'),map_location='cpu',weights_only=False)
        if ck['immutable']!=immutable:raise RuntimeError('Checkpoint provenance mismatch.')
        model.load_state_dict(ck['model_state'],strict=True)
        if objective is not None:objective.load_state_dict(ck['objective_state'],strict=True)
        optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        epoch,cursor,step=ck['epoch'],ck['cursor'],ck['step'];history,best,stall=ck['history'],ck['best'],ck['stall']
        elapsed_base=ck.get('elapsed_minutes_total',history[-1].get('elapsed_minutes',0) if history else 0)
        epoch_loss_sum,epoch_samples=ck['epoch_loss_sum'],ck['epoch_samples']
        classifier_seen_epoch=ck.get('classifier_seen_epoch',0);classifier_seen_total=ck.get('classifier_seen_total',0)
        last_epoch_classifier_count=ck.get('last_epoch_classifier_count',0)
        torch.set_rng_state(ck['rng']);random.setstate(ck['python_rng'])
        augmentation_rng.set_state(ck['augmentation_rng'])
        if amp:torch.cuda.set_rng_state_all(ck['cuda_rng'])
        if args.stage_name=='joint_attention':
            expected=min(1.,step/args.attention_transition_steps)
            if not math.isclose(float(model.peak_attention_progress),expected,abs_tol=1e-6):
                raise RuntimeError('Attention transition does not match the saved update cursor.')
    for group in optimizer.param_groups:
        group.setdefault('target_lr',group['lr'])
    def save(state,metrics=None):
        temp=run/'last.tmp'
        torch.save(dict(format='mixture_'+args.mode+'_v1',model_state=model.state_dict(),
            objective_state=objective.state_dict() if objective is not None else None,
            intensity_model_config=objective.intensity.config if objective is not None else None,
            objective_config=dict(charges=objective.charges,complexity_penalty=objective.complexity_penalty) if objective is not None else None,
            model_config=model.config if args.mode=='intensity' else model_cfg,training_exposure=exposure,
            training_data_sha256=sha(data.path/'manifest.json'),augmentation_rng=augmentation_rng.get_state(),
            training_raw_files=sorted({data.records[i]['raw_file'] for i in train}),
            optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),immutable=immutable,
            epoch=epoch,cursor=cursor,step=step,history=history,best=best,stall=stall,
            classifier_seen_epoch=classifier_seen_epoch,classifier_seen_total=classifier_seen_total,
            last_epoch_classifier_count=last_epoch_classifier_count,
            elapsed_minutes_total=elapsed_base+(time.monotonic()-started)/60,
            epoch_loss_sum=epoch_loss_sum,epoch_samples=epoch_samples,
            metrics=metrics or (history[-1]['metrics'] if history else {}),rng=torch.get_rng_state(),
            python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if amp else []),temp)
        temp.replace(run/'last.pt');status(state,train_size=len(train))
    def evaluate():
        if args.mode=='intensity':
            result=evaluate_intensity(model,data,panel,device,args.batch,stopped)
            if result is not None and args.ranking_cache is not None:
                from intensity_comparison import evaluate_cached
                ranking=evaluate_cached(model,args.ranking_cache,device,stopped,
                    lambda done,total:status('validating',validation_done=done,validation_total=total))
                if ranking is None:return None
                result['candidate_ranking']=ranking
                write(run/'validation.json',result)
            return result
        return evaluate_joint(model,objective,data,panel,device,run,status,stopped,args.validation_batch,
                              balanced_score=args.mode=='finetune')
    def record(metrics,initial=False):
        nonlocal best,stall,epoch_loss_sum,epoch_samples
        score=metrics['median_cosine'] if args.mode=='intensity' else metrics['score']
        if args.mode=='intensity' and args.ranking_cache is not None:
            # A high matched-peak cosine alone is not a useful ranking gate.
            # Select using the fixed sequencer/candidates, never a test split.
            score=metrics['candidate_ranking']['score'] if metrics['qualified_for_experiment'] else -1.
        improved=score>best;best=max(best,score)
        if not initial:stall=0 if improved else stall+1
        history.append(dict(epoch=epoch,step=step,metrics=metrics,classifier_train_spectra=last_epoch_classifier_count,
            train_loss=epoch_loss_sum/max(1,epoch_samples) if epoch_samples else None,
            elapsed_minutes=elapsed_base+(time.monotonic()-started)/60));write(run/'epochs.json',history)
        if not initial:epoch_loss_sum=0.;epoch_samples=0
        save('training',metrics)
        if improved:shutil.copy2(run/'last.pt',run/'best.pt')
    save('preparing')
    if not history:
        result=evaluate()
        if result is None:save('paused');return
        record(result,initial=True)
    from spectral_mixture_joint import pool_from_model,mixture_training_loss
    parameters=[p for g in optimizer.param_groups for p in g['params']]
    while args.epochs==0 or epoch<args.epochs:
        order=train.copy();random.Random(args.seed+epoch).shuffle(order)
        classifier_limit=min(args.classifier_epoch_samples,len(order)) if args.mode=='pretrain' else 0
        if args.mode=='pretrain' and classifier_seen_epoch!=min(cursor,classifier_limit):raise RuntimeError('Classifier sample cursor mismatch on resume.')
        classifier_members=set(order[:classifier_limit]) if classifier_limit else set()
        offset,retries=cursor,0
        while offset<len(order):
            if stopped():
                save('paused');return
            ids=order[offset:offset+args.batch];batch_start=time.monotonic()
            selected_count=(len(ids) if args.mode in ('finetune','joint') else sum(i in classifier_members for i in ids))
            if args.mode=='pretrain':model.classification_head.requires_grad_(selected_count>0)
            if args.length_bucketing:ids=sorted(ids,key=lambda i:int(data.lengths[i]))
            if retries==0:
                rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if amp else []
                augmentation_state=augmentation_rng.get_state()
            else:
                torch.set_rng_state(rng)
                augmentation_rng.set_state(augmentation_state)
                if amp:torch.cuda.set_rng_state_all(cuda_rng)
            model.train();optimizer.zero_grad(set_to_none=True);total=0.
            for a in range(0,len(ids),args.microbatch):
                subset=ids[a:a+args.microbatch];x,p,labels=data.batch(subset,device)
                x=augment_spectra(x,augmentation_rng,args.augment_probability,args.intensity_jitter,args.peak_dropout,args.min_keep_peaks)
                with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
                    if args.mode=='intensity':
                        values,stats=intensity_training_loss(model,x,[v[0] for v in labels],p[:,1].tolist())
                        loss=(values*stats['eligible']).sum()/len(ids)
                    else:
                        sequence,bce,_=model.loss_components(x,p,labels,'joint')
                        # Sequencing follows the actual 85/15 sampling. Only
                        # classification is balanced, avoiding an all-double head.
                        if args.mode=='pretrain':
                            loss=(sequence+model.sequence_regularizer).sum()/len(ids)
                            if selected_count:
                                selected=bce.new_tensor([i in classifier_members for i in subset])
                                loss=loss+args.classification_weight*(bce*selected).sum()/selected_count
                        else:
                            balance=sequence.new_tensor([len(train)/(2*class_counts[len(y)]) for y in labels])
                            loss=(sequence+model.sequence_regularizer+args.classification_weight*bce*balance).sum()/len(ids)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite base loss; no update.')
                scaler.scale(loss).backward();total+=float(loss.detach())
            if objective is not None and step%args.rank_every==0 and args.mixture_weight:
                selected=random.Random(args.seed+step).sample(ids,min(args.rank_samples,len(ids)))
                for index in selected:
                    x,p,labels=data.batch([index],device)
                    pools=pool_from_model(model,x,p,beam=4)
                    with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
                        values,_=mixture_training_loss(model,objective,x,p,labels,pools)
                        extra=args.mixture_weight*values.sum()/len(selected)
                    if not torch.isfinite(extra):raise RuntimeError('Nonfinite mixture loss; no update.')
                    scaler.scale(extra).backward();total+=float(extra.detach())
            scaler.unscale_(optimizer)
            norm=nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):
                if not scaler.is_enabled() or retries>=8:raise RuntimeError('Nonfinite gradients; no samples skipped.')
                scaler.update(new_scale=scaler.get_scale()/2);retries+=1;continue
            if args.mode=='pretrain' and not selected_count:
                if any(p.grad is not None for p in model.classification_head.parameters()):
                    raise RuntimeError('Frozen-head batch unexpectedly has classifier gradients.')
            if step==0 and args.mode!='intensity':
                early=(model.encoder.transformer_encoder.layers[0] if hasattr(model.encoder,'transformer_encoder') else model.encoder)
                audit={}
                for name,module in [('encoder_first_layer',early),('decoder',model.decoder),('classifier',model.classification_head)]:
                    grads=[p.grad for p in module.parameters() if p.grad is not None]
                    audit[name]=sum(float(g.float().norm()) for g in grads)
                if audit['encoder_first_layer']<=0 or audit['decoder']<=0 or (selected_count and audit['classifier']<=0):
                    raise RuntimeError('Missing expected first-update gradient: '+repr(audit))
                if objective is not None:
                    if objective.raw_weight.grad is None or not torch.isfinite(objective.raw_weight.grad):
                        raise RuntimeError('Mixture-score weight has no finite first-update gradient.')
                    if any(p.grad is not None for p in objective.intensity.parameters()):
                        raise RuntimeError('Validated intensity teacher must remain frozen in joint training.')
                    audit['mixture_score_weight']=float(objective.raw_weight.grad.abs())
                    audit['intensity_teacher_frozen']=True
                print('FIRST_UPDATE_GRADIENT_AUDIT '+json.dumps(audit),flush=True)
            for group in optimizer.param_groups:group['lr']=group['target_lr']*min(1.,(step+1)/200)
            scaler.step(optimizer);scaler.update();step+=1;cursor=offset+len(ids);offset=cursor;retries=0
            if args.stage_name=='joint_attention':
                model.set_attention_progress(min(1.,step/args.attention_transition_steps))
            classifier_seen_epoch+=selected_count;classifier_seen_total+=selected_count
            epoch_loss_sum+=total*len(ids);epoch_samples+=len(ids)
            status('training',train_size=len(train),loss=total,spectra_per_second=len(ids)/max(1e-9,time.monotonic()-batch_start),
                mean_loss=epoch_loss_sum/epoch_samples,learning_rates=[g['lr'] for g in optimizer.param_groups],
                attention_progress=float(model.peak_attention_progress) if args.stage_name=='joint_attention' else None,
                reconstruction_weight=float(F.softplus(objective.raw_weight)) if objective is not None else None)
            if step==1 or step%250==0:save('training')
        if args.mode=='pretrain' and classifier_seen_epoch!=classifier_limit:raise RuntimeError('Classifier epoch count mismatch.')
        last_epoch_classifier_count=classifier_seen_epoch;classifier_seen_epoch=0
        epoch+=1;cursor=0
        result=evaluate()
        if result is None:save('paused');return
        record(result)
        if args.patience and stall>=args.patience:save('early_stopped');return
    save('completed')


if __name__=='__main__':
    main()
