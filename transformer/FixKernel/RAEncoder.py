#encoding: utf-8

from math import sqrt
from torch import nn

from modules.base import Linear
from transformer.Encoder import Encoder as EncoderBase
from transformer.FixKernel.Encoder import EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, num_anchor=None, norm_residual=norm_residual_default, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		super(EncoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=ahsize, num_anchor=num_anchor, norm_residual=norm_residual, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, **kwargs)

		self.adaptor = Linear(isize, self.ahsize + self.ahsize, bias=enable_proj_bias)
		self.k_adaptor = self.p2normer = None

	def forward(self, inputs, mask=None, **kwargs):

		_inputs = self.normer(inputs)
		bsize, nquery = _inputs.size()[:2]
		nheads, adim, nanchor = self.num_head, self.attn_dim, self.num_anchor
		real_iK, real_iV = self.adaptor(_inputs).view(bsize, nquery, 2, nheads, adim).unbind(2)
		real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
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
		# (bsize, nheads, nquery, nanchor)
		out = self.outer(scores.transpose(-1, -2).matmul(out.view(bsize, nanchor, nheads, adim).transpose(1, 2)).transpose(1, 2).contiguous().view(bsize, nquery, self.ahsize))
		if self.drop is not None:
			out = self.drop(out)

		return out + (_inputs if self.norm_residual else inputs)

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
