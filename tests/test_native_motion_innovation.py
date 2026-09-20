import numpy as np
import torch

from hac.native_motion_data import aligned_actor_crop, estimate_camera_translation, phase_inputs
from hac.native_motion_innovation import ARMS, NativeMotionInnovation, bounded_motion_candidate, motion_loss


def test_camera_translation_sign_and_alignment():
    import cv2
    rng=np.random.default_rng(9);base=rng.integers(0,256,(180,320,3),dtype=np.uint8)
    transform=np.asarray([[1,0,7],[0,1,-4]],np.float32)
    moved=cv2.warpAffine(base,transform,(320,180),borderMode=cv2.BORDER_CONSTANT,borderValue=(124,116,104))
    shift,valid=estimate_camera_translation(base,moved,(135,65,185,115),source_size=(320,180))
    assert valid and np.allclose(shift[:2],[7,-4],atol=.35)
    box=(120,50,200,130)
    a=aligned_actor_crop(base,box,source_size=(320,180),source_sampling_shift=(0,0))
    b=aligned_actor_crop(moved,box,source_size=(320,180),source_sampling_shift=shift[:2])
    assert np.abs(a.astype(float)-b.astype(float))[8:-8,8:-8].mean()<8


def test_phase_controls_are_matched_and_signed_is_antisymmetric():
    crops=np.zeros((2,3,64,64,3),np.uint8);crops[:,0]=20;crops[:,1]=100;crops[:,2]=180
    values={arm:phase_inputs(crops,arm) for arm in ARMS}
    assert all(x.shape==(2,2,6,64,64) for x in values.values())
    assert np.allclose(values["appearance"][:,:,0:3],values["signed_motion"][:,:,0:3])
    assert np.all(values["signed_motion"][:,:,3:]>0)
    reversed_crops=crops[:,::-1].copy()
    assert np.allclose(phase_inputs(reversed_crops,"signed_motion")[:,:,3:],-values["signed_motion"][:,::-1,3:])
    assert np.allclose(phase_inputs(reversed_crops,"phase_destroyed")[:,:,3:],values["phase_destroyed"][:,::-1,3:])


def test_exact_retain_and_bounded_pair_mass():
    p=np.asarray([[.2,.5,.3],[.7,.2,.1]],np.float64);delta=np.asarray([.5,.5]);available=np.ones((2,2),bool)
    candidate,eligible=bounded_motion_candidate(p,delta,available)
    assert eligible.tolist()==[True,False]
    assert np.array_equal(candidate[1],p[1])
    assert candidate[0,0]==p[0,0] and np.isclose(candidate[0,1:].sum(),p[0,1:].sum())


def test_model_all_parameters_active_and_loss_finite():
    torch.manual_seed(2);model=NativeMotionInnovation();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    initial={k:v.detach().clone() for k,v in model.named_parameters()}
    for _ in range(3):
        phases=torch.randn(16,2,6,64,64);geometry=torch.randn(16,12)
        anchor=torch.softmax(torch.randn(16,3),1);anchor[:,0]*=.1;anchor=anchor/anchor.sum(1,keepdim=True)
        labels=torch.randint(0,3,(16,));available=torch.ones(16,2,dtype=torch.bool)
        optimizer.zero_grad();delta=model(phases,geometry);loss,_=motion_loss(delta,anchor,labels,available)
        assert torch.isfinite(loss);loss.backward();optimizer.step()
    assert all(not torch.equal(initial[k],v) for k,v in model.named_parameters())
    assert model.trainable_parameters==sum(v.numel() for v in model.parameters())
