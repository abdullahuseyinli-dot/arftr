"""Matched shallow residual for camera-compensated physical-time motion."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS=("appearance","signed_motion","phase_destroyed")


class NativeMotionInnovation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(6,16,5,2,2),nn.SiLU(),
            nn.Conv2d(16,32,3,2,1),nn.SiLU(),
            nn.Conv2d(32,48,3,2,1),nn.SiLU(),
            nn.AdaptiveAvgPool2d((4,4)),
        )
        self.head=nn.Sequential(nn.Linear(2*48*4*4+12,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, phases: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        if phases.ndim!=5 or phases.shape[1:3]!=(2,6) or geometry.shape!=(len(phases),12):
            raise ValueError("Malformed native-motion model inputs")
        encoded=self.encoder(phases.flatten(0,1)).flatten(1).reshape(len(phases),-1)
        return .5*torch.tanh(self.head(torch.cat((encoded,geometry),1)).squeeze(1))


def pair_eligible(anchor: np.ndarray|torch.Tensor, available: np.ndarray|torch.Tensor):
    if isinstance(anchor,torch.Tensor):
        return available.bool().all(1) & (anchor[:,0] < torch.minimum(anchor[:,1],anchor[:,2]))
    p=np.asarray(anchor); return np.asarray(available,bool).all(1) & (p[:,0] < np.minimum(p[:,1],p[:,2]))


def bounded_motion_candidate(anchor,delta,available):
    tensor=isinstance(anchor,torch.Tensor)
    p=anchor if tensor else np.asarray(anchor,np.float64)
    d=delta if tensor else np.asarray(delta,np.float64)
    eligible=pair_eligible(p,available)
    if tensor:
        q=p[:,1]+p[:,2]; logit=torch.log(p[:,1].clamp_min(1e-12)/p[:,2].clamp_min(1e-12))
        ratio=torch.sigmoid(logit+d); candidate=torch.stack((p[:,0],q*ratio,q*(1-ratio)),1)
        return torch.where(eligible[:,None],candidate,p),eligible
    q=p[:,1]+p[:,2];ratio=1/(1+np.exp(-(np.log(np.clip(p[:,1],1e-12,None)/np.clip(p[:,2],1e-12,None))+d)))
    candidate=np.column_stack((p[:,0],q*ratio,q*(1-ratio)));candidate[~eligible]=p[~eligible]
    return candidate,eligible


def motion_loss(delta: torch.Tensor,anchor: torch.Tensor,labels: torch.Tensor,available: torch.Tensor):
    candidate,eligible=bounded_motion_candidate(anchor,delta,available)
    nll=F.nll_loss(torch.log(candidate.clamp_min(1e-12)),labels)
    kl=(anchor*(torch.log(anchor.clamp_min(1e-12))-torch.log(candidate.clamp_min(1e-12)))).sum(1).mean()
    magnitude=delta[eligible].abs().mean() if eligible.any() else delta.sum()*0
    loss=nll+.1*kl+.01*magnitude
    return loss,{"task_nll":nll,"anchor_kl":kl,"eligible_abs_delta":magnitude}
