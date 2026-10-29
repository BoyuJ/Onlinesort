from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scripts import pretrain, run_online
from src.nn.global_identity_loss import global_identity_connection_loss
from src.nn.losses import local_proto_nce_loss, stage2_total_loss
from src.nn.model import SpikeEmbeddingModel
from src.model_io import build_model_config
from src.paper_evaluation import score_recording, aggregate_records, paired_comparisons
from src.eval import EventSet
from src.reproducibility import load_recording_manifest, resolve_data_records
from src.sort.online_assign import PrototypeAssignConfig, UnitPrototypeSet, assign_to_unit_prototypes, update_nearest_unit_prototype
from src.utils import Event, load_yaml

ROOT=Path(__file__).resolve().parents[1]


def auxiliary(z, labels, records, **settings):
    config={'stage2':{'losses':{'temporal_connection':dict(enabled=True,weight=.2,relative_margin=.08,negative_events=16,**settings)}}}
    batch={'unit_id':torch.tensor(labels),'recording_id':torch.tensor(records)}
    return global_identity_connection_loss(z,batch,{'all_idx':torch.arange(len(labels))},config)


def test_identity_loss_matches_explicit_formula_and_gradients():
    torch.manual_seed(2)
    raw=torch.randn(12,5,dtype=torch.float64,requires_grad=True)
    z=F.normalize(raw,dim=1)
    labels=[0,0,0,1,1,1];records=[0]*6
    actual,_=auxiliary(z,labels,records)
    views=[]
    for view in [z[:6],z[6:]]:
        identities=[]
        d=1-view@view.T
        for identity in [0,1]:
            events=[]
            for i in [n for n,y in enumerate(labels) if y==identity]:
                positive=[j for j,y in enumerate(labels) if y==identity and j!=i]
                negative=[j for j,y in enumerate(labels) if y!=identity]
                compact=d[i,positive].square().mean()
                margin=torch.stack([F.relu(d[i,j]-d[i,k]+.08).square() for j in positive for k in negative]).mean()
                events.append(compact+margin)
            identities.append(torch.stack(events).mean())
        views.append(torch.stack(identities).mean())
    expected=torch.stack(views).mean()
    torch.testing.assert_close(actual,expected)
    ga=torch.autograd.grad(actual,raw,retain_graph=True)[0]
    ge=torch.autograd.grad(expected,raw)[0]
    torch.testing.assert_close(ga,ge)
    assert torch.all(ga.norm(dim=1)>0)


@pytest.mark.parametrize('labels,records',[([0,1],[0,0]),([0,0],[0,0]),([0,0,1,1],[0,0,1,1])])
def test_auxiliary_omits_ineligible_and_never_pairs_across_recordings(labels,records):
    z=F.normalize(torch.randn(len(labels),3),dim=1).requires_grad_()
    loss,stats=auxiliary(z,labels,records)
    assert loss.item()==0 and stats['eligible_anchors']==0
    loss.backward()
    assert torch.count_nonzero(z.grad)==0


def test_nce_support_is_detached_and_uses_identity_means():
    anchor=torch.tensor([[1.,0.],[.99,.1],[0.,1.]],requires_grad=True)
    query=torch.tensor([[.8,.6],[.9,.1],[.5,.8]],requires_grad=True)
    y=torch.tensor([0,0,1])
    loss,_=local_proto_nce_loss(anchor,y,query,y,max_proto_per_cluster=1,top_pos=1,hard_neg=16,tau=.08,delta=.08)
    loss.backward()
    assert anchor.grad is None or torch.count_nonzero(anchor.grad)==0
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_encoder_update_keeps_gate_parameters_and_running_statistics_fixed():
    torch.manual_seed(42)
    cfg=load_yaml(ROOT/'configs/train.yaml')
    model=SpikeEmbeddingModel(build_model_config(cfg['model']))
    pretrain.apply_stage_trainability(model,stage_name='stage2_encoder')
    model.train();pretrain.enforce_stage_modules(model,stage_name='stage2_encoder')
    before={k:v.clone() for k,v in model.gate_net.state_dict().items()}
    encoder_before={k:v.clone() for k,v in model.encoder_net.state_dict().items()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=7e-4)
    x=torch.randn(12,64,9)
    meta={'gate_spatial':torch.zeros(12,5)}
    output=model(x,meta,apply_encode_gate=False)
    labels=torch.tensor([0,0,0,1,1,1]);batch={'unit_id':labels,'recording_id':torch.zeros(6,dtype=torch.long),'valid_mask':torch.ones(6,dtype=torch.bool)}
    episode={'all_idx':torch.arange(6),'anchor_idx':torch.tensor([0,1,3,4]),'loss_idx':torch.tensor([2,5])}
    nce,_=stage2_total_loss(output,None,batch,episode,cfg)
    identity,stats=pretrain.temporal_connection_loss(output.embedding,batch,episode,cfg)
    (nce+stats['lambda_temporal_conn_eff']*identity).backward();optimizer.step()
    assert all(torch.equal(v,model.gate_net.state_dict()[k]) for k,v in before.items())
    assert any(not torch.equal(v,model.encoder_net.state_dict()[k]) for k,v in encoder_before.items())


def test_schedule_and_validation_rank():
    cfg=load_yaml(ROOT/'configs/train.yaml')
    assert cfg['train']['stage2_encoder']['epochs']==30
    assert cfg['train']['stage2_encoder']['schedule_epochs']==60
    p=torch.nn.Parameter(torch.ones(()));opt=torch.optim.AdamW([p],lr=.0007)
    _,info=pretrain.build_scheduler_from_config(opt,scheduler_cfg=cfg['train']['scheduler'],base_lr=.0007,total_steps=60*288,steps_per_epoch=288)
    assert info['total_steps']==17280
    a={'val_lsm10_operational_gt':.8,'val_knn_purity10_operational_gt':.9,'val_ber002_operational_gt':.1}
    b=dict(a,val_knn_purity10_operational_gt=.8)
    c=dict(a,val_ber002_operational_gt=.2)
    assert pretrain.stage2_encoder_model_rank(a)>pretrain.stage2_encoder_model_rank(b)
    assert pretrain.stage2_encoder_model_rank(a)>pretrain.stage2_encoder_model_rank(c)


def event(t=100,c=0):
    return Event(t0=t,c0=c,amp_norm=5.,spatial_score=1.,neighbors=np.array([c]),qc_flags=0)


def test_assignment_uses_distinct_identity_gap_and_single_candidate():
    prototypes={0:UnitPrototypeSet(0,np.array([[1.,0.],[.999,.01]],np.float32),0),1:UnitPrototypeSet(1,np.array([[0.,1.]],np.float32),0)}
    result=assign_to_unit_prototypes([event()],np.array([[1.,0.]],np.float32),prototypes,cfg=PrototypeAssignConfig(.45,.01,-1))
    assert result.unit_id.tolist()==[0] and result.confidence[0]>.5
    result=assign_to_unit_prototypes([event()],np.array([[1.,0.]],np.float32),{0:prototypes[0]},cfg=PrototypeAssignConfig(.45,.01,-1))
    assert result.unit_id.tolist()==[0] and result.confidence[0]==pytest.approx(1.)
    update_nearest_unit_prototype(prototypes,0,np.array([.95,.05]),alpha_mu=.05)
    np.testing.assert_allclose(np.linalg.norm(prototypes[0].centers,axis=1),1.,atol=1e-6)


@pytest.mark.parametrize('embedding,reason',[([0.,0.],2),([float('nan'),0.],2),([-1.,0.],4)])
def test_invalid_or_distant_assignment(embedding,reason):
    proto={0:UnitPrototypeSet(0,np.array([[1.,0.]],np.float32),0)}
    result=assign_to_unit_prototypes([event()],np.array([embedding],np.float32),proto,cfg=PrototypeAssignConfig(.45,.01,-1))
    assert result.unit_id.tolist()==[-1] and result.reason_code.tolist()==[reason]


def benchmark_config():
    cfg=load_yaml(ROOT/'configs/infer.yaml')
    return run_online.ro.build_benchmark_config('kilo',bench_cfg_raw=cfg['benchmark'],label_cfg=cfg['labeling'],det_cfg=cfg['detection'])


def test_scoring_clips_both_bounds_and_retains_missed_truth():
    gt=EventSet(np.array([4999,5000,6000,99999,100000]),np.zeros(5,int),np.zeros(5,int))
    pred=EventSet(np.array([4999,5000,100000]),np.zeros(3,int),np.zeros(3,int))
    score=score_recording(pred,gt,fs=1000,config=benchmark_config())
    assert score['tp']==1 and score['fn']==2 and score['fp']==0
    assert score['recall']==pytest.approx(1/3)
    empty=EventSet(np.array([],dtype=int),np.array([],dtype=int),np.array([],dtype=int))
    score=score_recording(empty,gt,fs=1000,config=benchmark_config())
    assert score['f1']==score['nmi']==score['ari']==0 and score['fn']==3


def test_duplicate_events_and_fragmented_identities_do_not_inflate_recall():
    gt=EventSet(np.array([5000,6000,7000,8000]),np.zeros(4,int),np.zeros(4,int))
    pred=EventSet(np.array([5000,5000,6000,7000,8000]),np.array([0,0,0,1,1]),np.zeros(5,int))
    score=score_recording(pred,gt,fs=1000,config=benchmark_config())
    assert score['tp']==2 and score['fn']==2 and score['fp']==3


def test_recording_weighted_aggregation_and_matched_comparisons():
    rows=[dict(name=str(i),method='OnlineSort',condition='static',start_s=5.,stop_s=100.,precision=v,recall=v,f1=v,nmi=v,ari=v) for i,v in enumerate([1.,0.])]
    overall=next(r for r in aggregate_records(rows) if r['condition']=='overall')
    assert overall['f1']==.5 and paired_comparisons(rows)==[]
    rows.append(dict(rows[0],method='MountainSort4',precision=.1))
    assert paired_comparisons(rows)==[] # missing recording, no paired claim


def test_manifests_and_missing_data(tmp_path):
    training=load_recording_manifest(ROOT/'configs/manifests/train.json')
    benchmark=load_recording_manifest(ROOT/'configs/manifests/benchmark.json')
    assert len(training['records'])==36 and len(training['validation_indices'])==7
    assert len(benchmark['records'])==384
    assert sum(r['condition']=='drift' for r in benchmark['records'])==192
    with pytest.raises(FileNotFoundError):resolve_data_records(training,tmp_path,ROOT)


def test_pure_inference_never_loads_gt_or_full_offline_baseline(monkeypatch,tmp_path):
    from scripts import run_online as online
    def forbidden(*a,**k):raise AssertionError('GT or full-recording baseline was requested')
    monkeypatch.setattr(online,'build_benchmark_context',forbidden)
    monkeypatch.setattr(online,'run_full_offline_baseline',forbidden)
    ctx=SimpleNamespace(online_cfg=SimpleNamespace(init_window_sec=(5.,)),save_embeddings=True,fs=32000,weights_path=tmp_path/'weights')
    ctx.weights_path.write_bytes(b'test')
    monkeypatch.setattr(online,'prepare_context',lambda *a:ctx)
    monkeypatch.setattr(online,'collect_embedding_outputs',lambda *a:({'event_time_sample':np.array([160000]),'unit_id':np.array([-1])},{}))
    monkeypatch.setattr(online,'infer_identity_arrays',lambda arr,*a:(arr,{}))
    case=online.Case('case',None,None,None,tmp_path/'case.npz',tmp_path/'case.log')
    assert online.run_case(case,{'online':{'inference_only':True}},SimpleNamespace(force=False,init_window_sec=None))==0
    assert (tmp_path/'case_online_init5s.npz').exists()


def test_checkpoint_import_validates_hash_stage_and_architecture(tmp_path,monkeypatch):
    from scripts import import_weights
    from src.reproducibility import sha256
    cfg=load_yaml(ROOT/'configs/train.yaml')
    fake=tmp_path/'release';(fake/'configs').mkdir(parents=True)
    (fake/'configs/train.yaml').write_text((ROOT/'configs/train.yaml').read_text())
    model=SpikeEmbeddingModel(build_model_config(cfg['model']))
    source=tmp_path/'source.ckpt'
    torch.save({'model_state_dict':model.state_dict(),'metadata':{'stage':'stage2_encoder'}},source)
    reference={'checkpoints':{'encoder':{'sha256':sha256(source),'stage':'stage2_encoder','destination':'weights/encoder.ckpt'}}}
    (fake/'configs/paper_reference.json').write_text(json.dumps(reference))
    monkeypatch.setattr(import_weights,'ROOT',fake)
    output=import_weights.import_checkpoint(source,'encoder')
    assert output.read_bytes()==source.read_bytes()
    wrong=tmp_path/'wrong.ckpt';wrong.write_bytes(b'wrong')
    with pytest.raises(ValueError):import_weights.import_checkpoint(wrong,'encoder')
    assert output.read_bytes()==source.read_bytes()


def test_restore_preserves_optimizer_scheduler_and_rng():
    first=torch.nn.Linear(2,2);second=torch.nn.Linear(2,2)
    opt=torch.optim.AdamW(first.parameters(),lr=.001)
    schedule,_=pretrain.build_scheduler_from_config(opt,scheduler_cfg={'type':'warmup_cosine'},base_lr=.001,total_steps=10,steps_per_epoch=1)
    first(torch.ones(1,2)).sum().backward();opt.step();schedule.step()
    payload={'model_state_dict':first.state_dict(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':schedule.state_dict(),'torch_rng_state':torch.get_rng_state(),'epoch':3,'global_step':7}
    restored_opt=torch.optim.AdamW(second.parameters(),lr=.001)
    restored_schedule,_=pretrain.build_scheduler_from_config(restored_opt,scheduler_cfg={'type':'warmup_cosine'},base_lr=.001,total_steps=10,steps_per_epoch=1)
    assert pretrain.restore_training_checkpoint_payload(payload,model=second,optimizer=restored_opt,device=torch.device('cpu'),scheduler=restored_schedule)==(3,7,True)
    assert restored_opt.param_groups[0]['lr']==opt.param_groups[0]['lr']
    assert len(restored_opt.state)==len(opt.state)
    assert restored_schedule.last_epoch==schedule.last_epoch


def test_empty_candidates_and_insufficient_identity_gap():
    prototypes={0:UnitPrototypeSet(0,np.array([[1.,0.]],np.float32),4)}
    result=assign_to_unit_prototypes([event(c=0)],np.array([[1.,0.]],np.float32),prototypes,cfg=PrototypeAssignConfig(.45,.01,0),neighbors_lookup=np.arange(5)[:,None])
    assert result.unit_id[0]==-1 and result.reason_code[0]==3
    prototypes={0:UnitPrototypeSet(0,np.array([[1.,0.]],np.float32),0),1:UnitPrototypeSet(1,np.array([[1.,0.]],np.float32),0)}
    result=assign_to_unit_prototypes([event()],np.array([[1.,0.]],np.float32),prototypes,cfg=PrototypeAssignConfig(.45,.01,-1))
    assert result.unit_id[0]==-1 and result.reason_code[0]==5


def test_pure_assignment_accepts_without_updating_low_confidence_reference():
    cfg=load_yaml(ROOT/'configs/infer.yaml');online_cfg=run_online.resolve_online_config(cfg,None)
    ctx=SimpleNamespace(fs=1000,offline_cluster_cfg={},online_cfg=online_cfg,pipeline=SimpleNamespace(neighbors=None),assignment_gate_thresh=.5,candidate_filtering_disabled=False)
    unit=lambda k,z:run_online.InitializedUnit(k,np.array(z,np.float32),0,0.,5.,1.,0,10,8,np.arange(k*8,(k+1)*8))
    units={0:unit(0,[1.,0.]),1:unit(1,[.98,.2])}
    # Two established nearby identities; the new spike wins by enough for a label,
    # but not by enough for a reference update.
    emb=np.vstack([np.tile([1.,0.],(8,1)),np.tile([.98,.2],(8,1)),[1.,0.]]).astype(np.float32)
    arr={'event_time_sample':np.array(list(range(16))+[6000]),'center_channel':np.zeros(17,np.int32),'gate_prob':np.ones(17,np.float32),'encoded_mask':np.ones(17,np.int8),'confidence':np.ones(17,np.float32),'embedding':emb}
    labels=np.repeat([0,1],8).astype(np.int32)
    out,diag=run_online.apply_online_assignment(arr,init_idx=np.arange(16),init_labels=labels,units=units,ctx=ctx,init_window_sec=5.)
    assert out['unit_id'][-1]==0
    assert diag['updates']==0
