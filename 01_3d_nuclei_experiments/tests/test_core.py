import numpy as np
import pytest
import torch
from experiment.data import transform_zyx, load_case, list_cases, make_splits, source_group
from experiment.losses import components
from experiment.metrics import match_instances
from experiment.reconstruction import propagate, cluster_endpoints, reconstruct, _advect_numpy
from experiment.models import build_model, MODELS
from experiment.settings import ROOT, settings

torch.set_num_threads(2)


def test_supplied_patches_are_consistent():
    paths=list_cases(settings()['manual_dir'])
    if not paths: pytest.skip('No externally mounted annotated patches')
    for path in paths:
        c=load_case(path)
        assert c['image'].ndim == 3
        assert c['flow'].shape == (3,*c['image'].shape)
        assert np.isfinite(c['flow']).all()
        assert 0 <= c['image'].min() <= c['image'].max() <= 1


def test_group_split_prevents_overlap():
    paths=list_cases(settings()['manual_dir'])
    if not paths: pytest.skip('No externally mounted annotated patches')
    tr,val=make_splits(paths,seed=37)
    assert len(tr) and len(val)
    assert not ({source_group(p) for p in tr} & {source_group(p) for p in val})


def test_flow_loss_masks_are_not_broadcast_across_batches():
    b,z=2,4
    img=torch.ones((b,1,z,z,z))
    img[1] = 0 # all-zero sample must not contribute
    labels=torch.ones((b,z,z,z),dtype=torch.long)
    true=torch.zeros((b,3,z,z,z));true[:,0]=1
    pred=true.clone();pred[0,0]=0;pred[0,1]=1 # perpendicular, directional loss=1
    seg=torch.zeros((b,2,z,z,z))
    result=components(seg,pred,img,labels,true)
    assert result['flow'].item()==pytest.approx(1.)
    assert result['valid_n'].item()==z**3
    # All-zero mask must return finite exactly zero losses.
    empty=components(seg,pred,torch.zeros_like(img),labels,true)
    assert empty['total'].item()==0.


def test_masked_cross_entropy_ignores_zero_intensities():
    img=torch.tensor([[[[[1.,0.]]]]])
    labels=torch.tensor([[[[1,1]]]])
    seg=torch.tensor([[[[[0.,100.]]],[[[3.,-100.]]]]])
    fl=torch.zeros(1,3,1,1,2)
    got=components(seg,fl,img,labels,fl)
    expected=torch.nn.functional.cross_entropy(seg[:,:,:,:,:1],labels[:,:,:,:1],reduction='mean')
    assert got['foreground'].item()==pytest.approx(expected.item())


def test_rotation_four_times_identity_and_flow_signs():
    img=np.arange(4*5*5).reshape(4,5,5)
    labels=(img>6).astype(np.int32)
    flow=np.zeros((3,*img.shape),np.float32);flow[1]=1
    a,b,c,_=transform_zyx(img,labels,flow,k=1)
    assert np.all(c[1]==0) and np.all(c[2]==1)
    _,_,fl,_=transform_zyx(img,labels,flow,k=0,flip_z=True)
    assert np.all(fl[0]==0)
    _,_,fl,_=transform_zyx(img,labels,flow,k=0,flip_x=True)
    assert np.all(fl[1]==1)
    a,b,c,_=transform_zyx(img,labels,flow,k=0)
    for _ in range(4):a,b,c,_=transform_zyx(a,b,c,k=1)
    np.testing.assert_array_equal(a,img)
    np.testing.assert_array_equal(b,labels)
    np.testing.assert_allclose(c,flow)


def test_z_flip_flips_z_vector_only():
    image=np.ones((4,4,4));labels=np.ones_like(image)
    flow=np.ones((3,4,4,4),np.float32)
    _,_,f,_=transform_zyx(image,labels,flow,flip_z=True)
    assert np.all(f[0]==-1) and np.all(f[1:]==1)


def test_propagation_nearest_and_zero_steps():
    mask=np.zeros((8,8,8),bool);mask[2,3,4]=1
    flow=np.zeros((3,8,8,8),np.float32);flow[2]=1
    a,b=propagate(mask,flow,steps=5,rate=.2)
    np.testing.assert_allclose(b[2,0],5.,atol=1e-5)
    a,c=propagate(mask,flow,steps=0)
    np.testing.assert_allclose(c[:,0],[2,3,4])


def test_numpy_fallback_propagation_matches_nearest_voxel_sampling():
    # Checks the path used when Numba/llvmlite cannot be imported.
    rng = np.random.default_rng(123)
    flow = rng.normal(size=(3, 7, 8, 9)).astype(np.float32)
    pts = np.array([[0., 2., 6.], [3., 4., 7.], [1., 6., 8.]], dtype=np.float32)
    ref = pts.copy()
    for _ in range(5):
        for i in range(ref.shape[1]):
            z, y, x = ref[:, i].astype(int)
            v = flow[:, z, y, x].copy()
            ref[:, i] += v * np.float32(.2)
            ref[:, i] = np.clip(ref[:, i], 0, np.array(flow.shape[1:]) - 1)
    got = _advect_numpy(pts.copy(), flow, 5, np.float32(.2))
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_dbscan_collision_and_noise_preservation():
    mask=np.zeros((9,15,16),bool)
    # One original CC containing two distinct endpoint clouds.
    mask[2:7,2:7,2:9]=True
    # A second small source CC becomes noise; should remain a separate instance.
    mask[2:4,11:13,12:14]=True
    starts=np.stack(np.nonzero(mask)).astype(np.int32)
    endpoints=starts.astype(np.float32).copy()
    a=(starts[1]<7)
    b=a & (starts[2]>=5)
    endpoints[:,a & ~b]=np.array([[3],[4],[3]])
    endpoints[:,b]=np.array([[3],[4],[8]])
    labels=cluster_endpoints(mask,starts,endpoints,eps=1,min_samples=10)
    assert len(np.unique(labels[labels>0]))==3
    assert labels[2,11,12] not in (labels[2,3,2],labels[2,3,8])
    assert np.all(labels[mask]>0)


def test_dbscan_empty_and_no_clusters():
    mask=np.zeros((4,4,4),bool)
    assert not reconstruct(mask.astype(float),mask.astype(float),np.zeros((3,4,4,4))).any()
    mask[1,1,1]=1
    a,b=propagate(mask,np.zeros((3,4,4,4),np.float32),steps=0)
    labels=cluster_endpoints(mask,a,b,eps=3,min_samples=27)
    assert labels[1,1,1]==1


def test_instance_matching_perfect_and_merges():
    g=np.zeros((5,5,5),np.int32);g[0:2,0:2,0:2]=1;g[3:5,3:5,3:5]=2
    p=g.copy()
    exact=match_instances(g,p)
    assert exact['instance_f1_05']==1.
    assert exact['count_error']==0
    p[p==2]=1
    merged=match_instances(g,p)
    assert merged['instance_f1_05'] < 1.


@pytest.mark.parametrize('name',MODELS[:5])
def test_cnn_forward_and_backprop(name):
    net=build_model(name,base=8)
    image=torch.randn(1,1,16,16,16)
    prof=torch.zeros(1,8)
    seg,flow=net(image,prof)
    assert seg.shape==(1,2,16,16,16)
    assert flow.shape==(1,3,16,16,16)
    (seg.square().mean()+flow.square().mean()).backward()
    assert any(p.grad is not None for p in net.parameters())


def test_pretrained_requires_real_cache():
    for name in MODELS[5:]:
        with pytest.raises(ValueError,match='requires real cached'):
            build_model(name,8)(torch.ones(1,1,16,16,16),torch.zeros(1,8))


def test_pretrained_forward_with_precomputed_feature_tensor():
    net=build_model(MODELS[5],8)
    seg,flow=net(torch.rand(1,1,16,16,16),torch.zeros(1,8),torch.rand(1,256,4,4,4))
    assert seg.shape==(1,2,16,16,16) and flow.shape==(1,3,16,16,16)


def test_exact_original_64cube_compatibility():
    """Same state keys, parameter shapes and numerics as source original."""
    import importlib.util
    from pathlib import Path
    src=ROOT/'tests/fixtures/original_unet3d.py'
    spec=importlib.util.spec_from_file_location('original_3d_unet',src)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    reference=mod.UNet3D(1,8)
    reproduced=build_model('N0_original')
    assert list(reference.state_dict())==list(reproduced.state_dict())
    reproduced.load_state_dict(reference.state_dict())
    x=torch.randn(1,1,64,64,64);prof=torch.randn(1,8)
    with torch.no_grad():
        a,b=reference.eval()(x,prof)
        c,d=reproduced.eval()(x,prof)
    torch.testing.assert_close(a,c)
    torch.testing.assert_close(b,d)


def test_dino_feature_channel_contract():
    net=build_model('N7_cpdino_xy',8)
    seg,flow=net(torch.ones(1,1,16,16,16),torch.zeros(1,8),torch.zeros(1,768,4,4,4))
    assert seg.shape==(1,2,16,16,16)


def test_paths_are_outside_repo_and_override(monkeypatch, tmp_path):
    monkeypatch.setenv('VSEG_DATA_ROOT',str(tmp_path))
    cfg=settings()
    assert cfg['manual_dir']==tmp_path/'examples_gt'
    assert cfg['output_dir']==tmp_path/'experiments'/'01_3d_nuclei_experiments'
    assert cfg['feature_dir'].is_relative_to(tmp_path)
    monkeypatch.setenv('VSEG_DATA_ROOT',str(ROOT))
    with pytest.raises(ValueError,match='outside'):
        settings()


def test_explicit_case_groups(tmp_path):
    import json
    from experiment.data import source_group
    data_dir = tmp_path/'examples_gt'
    sample = data_dir/'some_case'
    sample.mkdir(parents=True)
    (data_dir/'groups.json').write_text(json.dumps({'some_case':'actual_specimen_1'}))
    assert source_group(sample)=='actual_specimen_1'


def test_all_eight_models_selected_by_default():
    # A finished model is reused; this test prevents silently comparing 3 only.
    cfg = settings()
    assert tuple(cfg['models']) == MODELS
    assert cfg['reuse_completed'] is True


def test_diagnostic_plots_and_sweep_figures(tmp_path):
    from experiment.plotting import debug_images, debug_motion, compare_summary, reconstruction_summary
    image = np.ones((8,8,8), np.float32)
    labels = np.zeros((8,8,8), np.int32)
    labels[2:6,2:6,2:6] = 1
    prob = (labels > 0).astype(np.float32)
    flow = np.zeros((3,8,8,8), np.float32)
    assert debug_images(tmp_path/'xyz.png',image,labels,prob,flow,labels).is_file()
    assert debug_motion(tmp_path/'motion.png',image,prob,flow,steps=3,rate=.2,threshold=.5).is_file()
    rows = [{'model':'N1_residual','status':'complete','instance_f1_05':.8,'instance_f1_075':.5,
             'foreground_dice':.9,'val_total':.3,'val_foreground':.1,'val_flow':.2}]
    assert compare_summary(rows,tmp_path/'comparison.png').is_file()
    sweep = [{'model':'N1_residual','steps':s,'eps':eps,'min_samples':27,
              'instance_f1_05':.5,'instance_f1_075':.2} for s in (20,50) for eps in (2.5,3.)]
    assert all(p.is_file() for p in reconstruction_summary(sweep,tmp_path))


def test_incomplete_best_checkpoint_can_be_finalized_without_retraining(tmp_path, monkeypatch):
    """A NaN-interrupted run must not delete its best checkpoint on restart."""
    import tifffile
    from experiment import training
    case = tmp_path / 'case'; case.mkdir()
    image = np.ones((16, 16, 16), np.float32)
    labels = np.zeros((16, 16, 16), np.uint16)
    labels[4:9, 4:9, 4:9] = 1
    flows = np.zeros((3, 16, 16, 16), np.float32)
    flows[0] = 1
    tifffile.imwrite(case / 'nuclei_patch.tif', image)
    tifffile.imwrite(case / 'masks_patch.tif', labels)
    np.savez(case / 'flow_patch.npz', flows)
    np.save(case / 'profile.npy', np.zeros(8, np.float32))
    cfg = {'seed': 37, 'batch_size': 1, 'num_workers': 0,
           'output_dir': tmp_path / 'output', 'feature_dir': tmp_path / 'cache',
           'base_channels': 8, 'learning_rate': .0001, 'weight_decay': .00001,
           'foreground_weight': 1., 'flow_weight': 1.,
           'incomplete_checkpoint': 'finalize', 'max_epochs': 1,
           'max_minutes_per_model': 1, 'validation_every': 1}
    model_dir = cfg['output_dir'] / 'N1_residual'
    model_dir.mkdir(parents=True)
    path = model_dir / 'best.pt'
    weights = build_model('N1_residual', 8).state_dict()
    torch.save({'model': weights, 'model_name': 'N1_residual', 'base_channels': 8,
                'epoch': 9, 'validation': {'total': .3, 'foreground': .2, 'flow': .1, 'dice': .8}}, path)
    before = path.read_bytes()
    monkeypatch.setattr(training, 'generate_validation_predictions',
                        lambda *args, **kwargs: dict(instance_f1_05=.7, instance_f1_075=.4,
                                                    pq_05=.5, pq_075=.2, count_error=1.,
                                                    foreground_dice=.8))
    result = training.run_model('N1_residual', [case], [case], cfg, torch.device('cpu'))
    assert result['status'] == 'complete' and result['best_epoch'] == 9
    assert result['recovered_existing_checkpoint'] is True
    assert path.read_bytes() == before


def test_nonfinite_loss_does_not_update_weights():
    from experiment.training import train_epoch
    class BrokenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.tensor(float('nan')))
        def forward(self, image, profile, features=None):
            shape = (image.shape[0], *image.shape[2:])
            seg = self.parameter * torch.ones((shape[0], 2, *shape[1:]))
            flow = torch.zeros((shape[0], 3, *shape[1:]))
            return seg, flow
    model = BrokenModel()
    before = model.parameter.data.clone()
    batch = {'image': torch.ones(1, 1, 4, 4, 4),
             'profile': torch.zeros(1, 8),
             'labels': torch.zeros(1, 4, 4, 4, dtype=torch.long),
             'flow': torch.zeros(1, 3, 4, 4, 4), 'case_id': ['broken_case']}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with pytest.raises(FloatingPointError, match='broken_case'):
        train_epoch(model, [batch], optimizer,
                    {'foreground_weight': 1., 'flow_weight': 1.},
                    torch.device('cpu'), False)
    assert torch.isnan(model.parameter.data) and torch.isnan(before)
    assert not optimizer.state


def test_early_stopping_and_epoch_ticker(tmp_path, monkeypatch, capsys):
    """Stop when validation worsens, retaining the earliest best checkpoint."""
    import tifffile
    from experiment import training
    sample = tmp_path / 'case'
    sample.mkdir()
    tifffile.imwrite(sample / 'nuclei_patch.tif', np.ones((16, 16, 16), np.float32))
    tifffile.imwrite(sample / 'masks_patch.tif', np.zeros((16, 16, 16), np.uint16))
    np.savez(sample / 'flow_patch.npz', np.zeros((3, 16, 16, 16), np.float32))
    np.save(sample / 'profile.npy', np.zeros(8, np.float32))
    cfg = dict(seed=37, batch_size=1, num_workers=0, output_dir=tmp_path/'output',
               feature_dir=tmp_path/'cache', base_channels=8, learning_rate=1e-4,
               weight_decay=1e-5, foreground_weight=1., flow_weight=1.,
               max_epochs=20, max_minutes_per_model=1, validation_every=1,
               early_stopping_patience=2, propagation_steps=0, propagation_rate=.2,
               dbscan_eps=3., dbscan_min_samples=27, foreground_threshold=.5)
    iterations = [0]
    def fake_train(*args, **kwargs):
        return {'total':.3, 'foreground':.2, 'flow':.1}, 1
    def fake_val(*args, **kwargs):
        iterations[0] += 1
        total = .2 if iterations[0] == 1 else .2 + iterations[0]*.1
        return dict(total=total, foreground=total, flow=0., dice=.8)
    monkeypatch.setattr(training, 'train_epoch', fake_train)
    monkeypatch.setattr(training, 'validate', fake_val)
    monkeypatch.setattr(training, 'generate_validation_predictions', lambda *a, **kw: dict(
        instance_f1_05=.7, instance_f1_075=.4, pq_05=.5, pq_075=.2,
        count_error=1., foreground_dice=.8))
    results = training.run_model('N1_residual', [sample], [sample], cfg, torch.device('cpu'))
    assert results['best_epoch'] == 1
    assert results['trained_epochs'] == 3
    assert results['val_total'] == pytest.approx(.2)
    output = capsys.readouterr().out
    assert 'epoch 2/20' in output and 'EARLY STOP:' in output
    state = torch.load(cfg['output_dir']/'N1_residual'/'best.pt', weights_only=True)
    assert state['epoch'] == 1
