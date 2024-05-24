# encoding: utf-8

from torch.nn.modules.loss import _Loss

from utils.torch.comp import torch_any_wodim
from utils.torch.ext import cosim, pearson_corr

from cnfg.ihyp import ieps_ln_default


def token_loss_mask_reduction(loss, mask=None, reduction="mean"):
    _is_mean_reduction = reduction == "mean"
    if _is_mean_reduction:
        _num = loss.numel()
    if mask is not None:
        loss.masked_fill_(mask, 0.0)
        if _is_mean_reduction:
            _num -= mask.int().sum().item()
    if reduction != "none":
        loss = loss.sum()
    if _is_mean_reduction:
        loss = loss.div_(float(_num))

    return loss


def cosim_loss(a, b, mask=None, dim=-1, reduction="mean", eps=ieps_ln_default):
    return -token_loss_mask_reduction(cosim(a, b, dim=dim, keepdim=False, eps=eps), mask=mask, reduction=reduction)


class Cosim(_Loss):

    def __init__(self, dim=-1, reduction="mean", eps=ieps_ln_default, **kwargs):
        super(Cosim, self).__init__()
        self.dim, self.reduction, self.eps = dim, reduction, eps

    def forward(self, input, target, mask=None, **kwargs):
        return cosim_loss(input, target, mask=mask, dim=self.dim, reduction=self.reduction, eps=self.eps)


def pearson_loss(a, b, mask=None, dim=-1, reduction="mean", eps=ieps_ln_default):
    return -token_loss_mask_reduction(pearson_corr(a, b, dim=dim, keepdim=False, eps=eps), mask=mask,
                                      reduction=reduction)


class PearsonCorr(Cosim):

    def forward(self, input, target, mask=None, **kwargs):
        return pearson_loss(input, target, mask=mask, dim=self.dim, reduction=self.reduction, eps=self.eps)


def scaledis_loss(a, b, mask=None, dim=-1, reduction="mean", sort=True, stable=False):
    if sort:
        _sb, _ib = b.sort(dim=dim, stable=stable)
        _sa = a.gather(dim, _ib)
    else:
        _sa, _sb = a, b
    _n = _sa.size(-1) - 1
    _ha, _ta = _sa.narrow(dim, 0, _n), _sa.narrow(dim, 1, _n)
    _hb, _tb = _sb.narrow(dim, 0, _n), _sb.narrow(dim, 1, _n)
    _zero_mask = _ta.eq(0.0) | _tb.eq(0.0)
    if torch_any_wodim(_zero_mask).item():
        loss = (_ha / _ta - _hb / _tb).masked_fill(_zero_mask, 0.0).abs().sum(dim) / (
                    float(_n) - _zero_mask.to(a.dtype).sum(dim))
    else:
        loss = (_ha / _ta - _hb / _tb).abs().mean(dim)

    return token_loss_mask_reduction(loss, mask=mask, reduction=reduction)


class ScaleDis(_Loss):

    def __init__(self, dim=-1, reduction="mean", sort=True, stable=False, **kwargs):
        super(ScaleDis, self).__init__()
        self.dim, self.reduction, self.sort, self.stable = dim, reduction, sort, stable

    def forward(self, input, target, mask=None, **kwargs):
        return scaledis_loss(input, target, mask=mask, dim=self.dim, reduction=self.reduction, sort=self.sort,
                             stable=self.stable)


def orderdis_loss(a, b, mask=None, dim=-1, reduction="mean", stable=False):
    _sa = a.gather(dim, b.argsort(dim=dim, descending=False, stable=stable))
    _n = _sa.size(-1) - 1

    return token_loss_mask_reduction((_sa.narrow(dim, 1, _n) - _sa.narrow(dim, 0, _n)).clamp_(min=0.0).mean(dim),
                                     mask=mask, reduction=reduction)


class OrderDis(_Loss):

    def __init__(self, dim=-1, reduction="mean", stable=False, **kwargs):
        super(OrderDis, self).__init__()
        self.dim, self.reduction, self.stable = dim, reduction, stable

    def forward(self, input, target, mask=None, **kwargs):
        return orderdis_loss(input, target, mask=mask, dim=self.dim, reduction=self.reduction, stable=self.stable)


def simorder_loss(a, b, mask=None, dim=-1, reduction="mean", eps=ieps_ln_default, stable=False, sim_func=pearson_corr):
    _sim = sim_func(a, b, dim=dim, keepdim=False, eps=eps)
    _sa = a.gather(dim, b.argsort(dim=dim, descending=False, stable=stable))
    _n = _sa.size(-1) - 1
    _order_loss = (_sa.narrow(dim, 1, _n) - _sa.narrow(dim, 0, _n)).clamp_(min=0.0).mean(dim)

    _is_mean_reduction = reduction == "mean"
    if _is_mean_reduction:
        _num = _sim.numel()
    if mask is not None:
        _sim.masked_fill_(mask, 0.0)
        _order_loss.masked_fill_(mask, 0.0)
        if _is_mean_reduction:
            _num -= mask.int().sum().item()
    if reduction != "none":
        _sim = _sim.sum()
        _order_loss = _order_loss.sum()
    loss = _order_loss.sub_(_sim)
    if _is_mean_reduction:
        loss = loss.div_(float(_num))

    return loss


class SimOrder(_Loss):

    def __init__(self, dim=-1, reduction="mean", eps=ieps_ln_default, stable=False, sim_func=pearson_corr, **kwargs):
        super(SimOrder, self).__init__()
        self.dim, self.reduction, self.eps, self.stable, self.sim_func = dim, reduction, eps, stable, sim_func

    def forward(self, input, target, mask=None, **kwargs):
        return simorder_loss(input, target, mask=mask, dim=self.dim, reduction=self.reduction, eps=self.eps,
                             stable=self.stable, sim_func=self.sim_func)
