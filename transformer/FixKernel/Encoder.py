#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.act import Custom_Act, LGLU
from modules.base import Dropout, Linear, SparseNormer
from modules.group.base import GroupLinear
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class EncoderLayer(nn.Module):

	# share_head: use a shared FFN for all heads

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, num_anchor=None, share_head=False, norm_residual=norm_residual_default, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, sparsenorm=False, custom_act=use_adv_act_default, use_glu=use_glu_ffn, **kwargs):

		super(EncoderLayer, self).__init__()

		_ahsize = parse_none(ahsize, isize)
		_act_drop = parse_none(act_drop, dropout)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize
		self.num_anchor = parse_none(num_anchor, num_head)
		self.attn_dim = _ahsize // num_head
		self.ahsize = self.attn_dim * num_head
		self.num_head = num_head
		self.anchor_size = self.num_anchor * self.attn_dim
		_ = _fhsize // self.num_head // self.num_anchor
		if (use_glu is not None) and (_ % 2 == 1):
			_ += 1
		_fhsize = _ * self.num_head * self.num_anchor

		self.q_anchor = nn.Parameter(torch.Tensor(1, self.num_head, self.num_anchor, self.attn_dim).uniform_(- sqrt(1.0 / self.attn_dim), sqrt(1.0 / self.attn_dim)))
		self.adaptor = Linear(isize, self.ahsize * 3, bias=enable_proj_bias)
		if sparsenorm:
			self.p1normer, self.p2normer = SparseNormer(dim=-1), SparseNormer(dim=-1)
		else:
			self.p1normer = self.p2normer = nn.Softmax(dim=-1)
		self.attn_drop = Dropout(attn_drop, inplace=sparsenorm) if attn_drop > 0.0 else None

		_ = [Linear(self.anchor_size, _fhsize, bias=True) if share_head else GroupLinear(self.anchor_size * self.num_head, _fhsize * self.num_head, self.num_head, bias=True, trans_input=False, shuffle=False, flatten_output=False)]
		if use_glu is None:
			_.append(Custom_Act() if custom_act else nn.ReLU(inplace=True))
		else:
			use_glu = use_glu.lower()
			if use_glu == "glu":
				_.append(nn.GLU())
			else:
				_act = get_act(use_glu, None)
				if _act is not None:
					_.append(_act())
				_.append(LGLU())
		if _act_drop > 0.0:
			_.append(Dropout(_act_drop, inplace=inplace_after_Custom_Act))
		self.anchor_net = nn.Sequential(*_)

		_isize = _fhsize * self.num_head // self.num_anchor
		if use_glu is not None:
			_isize = _isize // 2
		if (_isize * _fhsize) > (self.ahsize * (_fhsize + _isize)):
			_ = [Linear(_isize, self.ahsize, bias=enable_proj_bias), Linear(self.ahsize, _fhsize)]
			_drop_ind = 3
		else:
			_ = [Linear(_isize, _fhsize)]
			_drop_ind = 2
		if use_glu is None:
			_.extend([Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_fhsize, self.ahsize, bias=enable_bias)])
		else:
			use_glu = use_glu.lower()
			if use_glu == "glu":
				_.append(nn.GLU())
			else:
				_act = get_act(use_glu, None)
				if _act is not None:
					_.append(_act())
					_drop_ind += 1
				_.append(LGLU())
			_.append(Linear(_fhsize // 2, self.ahsize, bias=enable_bias))
		if _act_drop > 0.0:
			_.insert(_drop_ind, Dropout(_act_drop, inplace=inplace_after_Custom_Act))
		self.net = nn.Sequential(*_)
		self.k_adaptor = Linear(self.ahsize, self.ahsize, bias=enable_proj_bias)# how about use GroupLinear? => GKAEncoder
		self.outer = Linear(self.ahsize, isize, bias=enable_bias)

		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None
		self.norm_residual = norm_residual

	def forward(self, inputs, mask=None, **kwargs):

		_inputs = self.normer(inputs)
		bsize, nquery = _inputs.size()[:2]
		nheads, adim, nanchor = self.num_head, self.attn_dim, self.num_anchor
		real_iQ, real_iK, real_iV = self.adaptor(_inputs).view(bsize, nquery, 3, nheads, adim).unbind(2)
		real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
		# (bsize, nheads, nanchor, nquery)
		scores = self.q_anchor.matmul(real_iK) / sqrt(adim)
		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)
		scores = self.p1normer(scores)
		if self.attn_drop is not None:
			scores = self.attn_drop(scores)
		# (bsize, nheads, nanchor * adim)
		out = scores.matmul(real_iV).view(bsize, nheads, -1)
		# (bsize, nanchor, nheads * adim)
		out = self.anchor_net(out).view(bsize, nheads, nanchor, -1).transpose(1, 2).contiguous().view(bsize, nanchor, -1)
		out = self.net(out)
		# how about re-use scores.transpose(-1, -2)? => RAEncoder
		_k = self.k_adaptor(out).view(bsize, nanchor, nheads, adim).permute(0, 2, 3, 1)
		# (bsize, nheads, nquery, nanchor)
		scores = self.p2normer(real_iQ.matmul(_k) / sqrt(adim))
		if self.attn_drop is not None:
			scores = self.attn_drop(scores)
		out = self.outer(scores.matmul(out.view(bsize, nanchor, nheads, adim).transpose(1, 2)).transpose(1, 2).contiguous().view(bsize, nquery, self.ahsize))
		if self.drop is not None:
			out = self.drop(out)

		return out + (_inputs if self.norm_residual else inputs)

	def fix_init(self):

		if hasattr(self, "fix_load"):
			self.fix_load()
		with torch_no_grad():
			self.q_anchor.uniform_(- sqrt(1.0 / self.attn_dim), sqrt(1.0 / self.attn_dim))

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, num_anchor=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		_num_anchor = parse_none(num_anchor, num_head)
		if share_layer:
			_shared_layer = EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, num_anchor=_num_anchor)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, num_anchor=_num_anchor) for i in range(num_layer)])
