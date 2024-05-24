#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.mulang.eff.base import LayerNorm, MBLinear
from modules.mulang.eff.m2m import CrossAttn, PositionwiseFF, SelfAttn
from modules.mulang.eff.m2o import PositionwiseFF as m2oPositionwiseFF
from modules.mulang.eff.o2m import SelfAttn as o2mSelfAttn
from transformer.MuLang.M2M.Decoder import Decoder as DecoderBase, DecoderLayer as DecoderLayerBase
from utils.base import index_tensors, select_zero_
from utils.decode.beam import expand_bsize_for_beam
from utils.fmt.parser import parse_none
from utils.sampler import SampleMax
from utils.torch.comp import all_done, torch_no_grad

from cnfg.ihyp import *
from cnfg.vocab.base import eos_id, pad_id

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, ngroup=None, ntask=None, expand_layer=False, merge_layer=False, k_rel_pos=use_k_relative_position_decoder, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, ngroup=ngroup, ntask=ntask, expand_layer=expand_layer, merge_layer=merge_layer, k_rel_pos=k_rel_pos, **kwargs)

		self.layer_normer1 = LayerNorm(isize if expand_layer else (ngroup, isize,), ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.layer_normer2 = LayerNorm((ngroup, isize,), ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		SAttn = o2mSelfAttn if expand_layer else SelfAttn
		self.self_attn = SAttn(isize, _ahsize, isize, ngroup, num_head=num_head, dropout=attn_drop, k_rel_pos=k_rel_pos, uni_direction_reduction=True)
		self.cross_attn = CrossAttn(isize, _ahsize, isize, ngroup, num_head=num_head, dropout=attn_drop)
		FFN = m2oPositionwiseFF if merge_layer else PositionwiseFF
		self.ff = FFN(isize, ngroup, hsize=_fhsize, dropout=dropout, ntask=ntask)

	def forward(self, inpute, inputo, sattn_w=None, cattn_w=None, ffn_w=None, taskid=None, src_pad_mask=None, tgt_pad_mask=None, query_unit=None, **kwargs):

		if query_unit is None:
			_inputo = self.layer_normer1(inputo, taskid=taskid)

			context = self.self_attn(_inputo, mask=tgt_pad_mask)

			if self.drop is not None:
				context = self.drop(context)
			_res_add = _inputo if self.norm_residual else inputo

		else:
			_query_unit = self.layer_normer1(query_unit, taskid=taskid)

			context, states_return = self.self_attn(_query_unit, states=inputo)

			if self.drop is not None:
				context = self.drop(context)
			_res_add = _query_unit if self.norm_residual else query_unit

		if self.expand_layer:
			_res_add = _res_add.unsqueeze(-2)
		context = context + _res_add

		if sattn_w is not None:
			_osize = list(context.size())
			_osize[-2], _osize[-1] = _osize[-1], _osize[-2]
			context = context.transpose(-1, -2).contiguous().view(-1, self.ngroup).mm(sattn_w).view(_osize).transpose(-1, -2).contiguous()

		_context = self.layer_normer2(context, taskid=taskid)
		_context_new = self.cross_attn(_context, inpute, mask=src_pad_mask)

		if self.drop is not None:
			_context_new = self.drop(_context_new)

		context = _context_new + (_context if self.norm_residual else context)

		if cattn_w is not None:
			_osize = list(context.size())
			_osize[-2], _osize[-1] = _osize[-1], _osize[-2]
			context = context.transpose(-1, -2).contiguous().view(-1, self.ngroup).mm(cattn_w).view(_osize).transpose(-1, -2).contiguous()

		context = self.ff(context, weight=ffn_w, taskid=taskid)

		if query_unit is None:
			return context
		else:
			return context, states_return

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, ntask=None, ngroup=None, task_emb_w=None, share_layer=False, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, ntask=ntask, ngroup=ngroup, task_emb_w=task_emb_w, share_layer=share_layer, **kwargs)

		if share_layer:
			_shared_layer = DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ngroup=ngroup, ntask=ntask, expand_layer=False, merge_layer=False)
			self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ngroup=ngroup, ntask=ntask, expand_layer=True, merge_layer=False)] + [_shared_layer for i in range(num_layer - 2)] + [DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ngroup=ngroup, ntask=ntask, expand_layer=False, merge_layer=True)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ngroup=ngroup, ntask=ntask, expand_layer=(i == 0), merge_layer=(i == num_layer - 1)) for i in range(num_layer)])

		self.classifier = MBLinear(isize, nwd, ntask)
		if bindemb:
			self.classifier.weight = self.wemb.weight

		if norm_output:
			self.out_normer = LayerNorm(isize, ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	def forward(self, inpute, inputo, taskid=None, src_pad_mask=None, **kwargs):

		nquery = inputo.size(-1)

		out = self.wemb(inputo) + self.task_emb.weight[taskid]
		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))

		_gwf, _gw, _gwl = self.group_weight_fl_common[taskid].softmax(-2), self.group_weight[taskid].softmax(-2), self.group_weight_layer[taskid].softmax(-1)

		if self.drop is not None:
			out = self.drop(out)
			_gwf = self.gw_drop(_gwf)
			_gw = self.gw_drop(_gw)
			_gwl = self.gw_drop(_gwl)

		_mask = self._get_subsequent_mask(nquery)

		_flayer_w, _layer_w = _gwf.unbind(0)
		_w = [[None] + list(_flayer_w.unbind(0))] + [_wu.unbind(0) for _wu in _gw.unbind(0)] + [list(_layer_w.unbind(0)) + [_gwl]]
		for net, (_w_sattn, _w_cattn, _w_ffn,) in zip(self.nets, _w):
			out = net(inpute, out, sattn_w=_w_sattn, cattn_w=_w_cattn, ffn_w=_w_ffn, taskid=taskid, src_pad_mask=src_pad_mask, tgt_pad_mask=_mask)

		if self.out_normer is not None:
			out = self.out_normer(out, taskid=taskid)

		out = self.lsm(self.classifier(out, taskid))

		return out

	def greedy_decode(self, inpute, taskid=None, src_pad_mask=None, max_len=512, fill_pad=False, sample=False, **kwargs):

		bsize = inpute.size(0)

		out = self.get_sos_emb(inpute)
		_task_emb = self.task_emb.weight[taskid]

		out = out + _task_emb
		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
			out = self.pemb.get_pos(0).add(out, alpha=sqrt_isize)

		_gwf, _gw, _gwl = self.group_weight_fl_common[taskid].softmax(-2), self.group_weight[taskid].softmax(-2), self.group_weight_layer[taskid].softmax(-1)

		if self.drop is not None:
			out = self.drop(out)
			_gwf = self.gw_drop(_gwf)
			_gw = self.gw_drop(_gw)
			_gwl = self.gw_drop(_gwl)

		states = {}
		_flayer_w, _layer_w = _gwf.unbind(0)
		_w = [[None] + list(_flayer_w.unbind(0))] + [_wu.unbind(0) for _wu in _gw.unbind(0)] + [list(_layer_w.unbind(0)) + [_gwl]]
		for _tmp, (net, (_w_sattn, _w_cattn, _w_ffn,),) in enumerate(zip(self.nets, _w)):
			out, _state = net(inpute, (None, None,), sattn_w=_w_sattn, cattn_w=_w_cattn, ffn_w=_w_ffn, taskid=taskid, src_pad_mask=src_pad_mask, tgt_pad_mask=None, query_unit=out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out, taskid=taskid)

		out = self.classifier(out, taskid)
		wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

		trans = [wds]

		done_trans = wds.eq(eos_id)

		for i in range(1, max_len):

			out = self.wemb(wds) + _task_emb
			if self.pemb is not None:
				out = self.pemb.get_pos(i).add(out, alpha=sqrt_isize)
			if self.drop is not None:
				out = self.drop(out)

			for _tmp, (net, (_w_sattn, _w_cattn, _w_ffn,),) in enumerate(zip(self.nets, _w)):
				out, _state = net(inpute, states[_tmp], sattn_w=_w_sattn, cattn_w=_w_cattn, ffn_w=_w_ffn, taskid=taskid, src_pad_mask=src_pad_mask, tgt_pad_mask=None, query_unit=out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out, taskid=taskid)

			out = self.classifier(out, taskid)
			wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

			trans.append(wds.masked_fill(done_trans, pad_id) if fill_pad else wds)

			done_trans = done_trans | wds.eq(eos_id)
			if all_done(done_trans, bsize):
				break

		return torch.cat(trans, 1)

	def beam_decode(self, inpute, taskid=None, src_pad_mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp, fill_pad=False, **kwargs):

		bsize, seql = inpute.size()[:2]

		beam_size2 = beam_size * beam_size
		bsizeb2 = bsize * beam_size2
		real_bsize = bsize * beam_size

		out = self.get_sos_emb(inpute)
		_task_emb = self.task_emb.weight[taskid]

		if length_penalty > 0.0:
			lpv = out.new_ones(real_bsize, 1)
			lpv_base = 6.0 ** length_penalty

		out = out + _task_emb
		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
			out = self.pemb.get_pos(0).add(out, alpha=sqrt_isize)

		_gwf, _gw, _gwl = self.group_weight_fl_common[taskid].softmax(-2), self.group_weight[taskid].softmax(-2), self.group_weight_layer[taskid].softmax(-1)

		if self.drop is not None:
			out = self.drop(out)
			_gwf = self.gw_drop(_gwf)
			_gw = self.gw_drop(_gw)
			_gwl = self.gw_drop(_gwl)

		states = {}
		_flayer_w, _layer_w = _gwf.unbind(0)
		_w = [[None] + list(_flayer_w.unbind(0))] + [_wu.unbind(0) for _wu in _gw.unbind(0)] + [list(_layer_w.unbind(0)) + [_gwl]]
		for _tmp, (net, (_w_sattn, _w_cattn, _w_ffn,),) in enumerate(zip(self.nets, _w)):
			out, _state = net(inpute, (None, None,), sattn_w=_w_sattn, cattn_w=_w_cattn, ffn_w=_w_ffn, taskid=taskid, src_pad_mask=src_pad_mask, tgt_pad_mask=None, query_unit=out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out, taskid=taskid)

		out = self.lsm(self.classifier(out, taskid))

		scores, wds = out.topk(beam_size, dim=-1)
		scores = scores.squeeze(1)
		sum_scores = scores
		wds = wds.view(real_bsize, 1)
		trans = wds
		_inds_add_beam2 = torch.arange(0, bsizeb2, beam_size2, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)
		_inds_add_beam = torch.arange(0, real_bsize, beam_size, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)

		done_trans = wds.view(bsize, beam_size).eq(eos_id)

		self.repeat_cross_attn_buffer(beam_size)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

		states = expand_bsize_for_beam(states, beam_size=beam_size)

		for step in range(1, max_len):

			out = self.wemb(wds) + _task_emb
			if self.pemb is not None:
				out = self.pemb.get_pos(step).add(out, alpha=sqrt_isize)
			if self.drop is not None:
				out = self.drop(out)

			for _tmp, (net, (_w_sattn, _w_cattn, _w_ffn,),) in enumerate(zip(self.nets, _w)):
				out, _state = net(inpute, states[_tmp], sattn_w=_w_sattn, cattn_w=_w_cattn, ffn_w=_w_ffn, taskid=taskid, src_pad_mask=_src_pad_mask, tgt_pad_mask=None, query_unit=out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out, taskid=taskid)

			out = self.lsm(self.classifier(out, taskid)).view(bsize, beam_size, -1)

			_scores, _wds = out.topk(beam_size, dim=-1)
			_done_trans_unsqueeze = done_trans.unsqueeze(2)
			_scores = (_scores.masked_fill(_done_trans_unsqueeze.expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).repeat(1, 1, beam_size).masked_fill_(select_zero_(_done_trans_unsqueeze.repeat(1, 1, beam_size), -1, 0), -inf_default))

			if length_penalty > 0.0:
				lpv.masked_fill_(~done_trans.view(real_bsize, 1), ((step + 6.0) ** length_penalty) / lpv_base)

			if clip_beam and (length_penalty > 0.0):
				scores, _inds = (_scores.view(real_bsize, beam_size) / lpv.expand(real_bsize, beam_size)).view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + _inds_add_beam2).view(real_bsize)
				sum_scores = _scores.view(bsizeb2).index_select(0, _tinds).view(bsize, beam_size)
			else:
				scores, _inds = _scores.view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + _inds_add_beam2).view(real_bsize)
				sum_scores = scores

			wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)

			_inds = (_inds // beam_size + _inds_add_beam).view(real_bsize)

			trans = torch.cat((trans.index_select(0, _inds), wds.masked_fill(done_trans.view(real_bsize, 1), pad_id) if fill_pad else wds), 1)

			done_trans = (done_trans.view(real_bsize).index_select(0, _inds) | wds.eq(eos_id).squeeze(1)).view(bsize, beam_size)

			_done = False
			if length_penalty > 0.0:
				lpv = lpv.index_select(0, _inds)
			elif (not return_all) and all_done(done_trans.select(1, 0), bsize):
				_done = True

			if _done or all_done(done_trans, real_bsize):
				break

			states = index_tensors(states, indices=_inds, dim=0)

		if (not clip_beam) and (length_penalty > 0.0):
			scores = scores / lpv.view(bsize, beam_size)
			scores, _inds = scores.topk(beam_size, dim=-1)
			_inds = (_inds + _inds_add_beam).view(real_bsize)
			trans = trans.view(real_bsize, -1).index_select(0, _inds)

		if return_all:

			return trans.view(bsize, beam_size, -1), scores
		else:

			return trans.view(bsize, beam_size, -1).select(1, 0)

	def update_vocab(self, indices, wemb_weight=None):

		_nwd = indices.numel()
		_wemb = nn.Embedding(_nwd, self.wemb.weight.size(-1), padding_idx=self.wemb.padding_idx)
		_classifier = MBLinear(self.classifier.weight.size(-1), _nwd, self.classifier.bias.size(0))
		with torch_no_grad():
			if wemb_weight is None:
				_wemb.weight.copy_(self.wemb.weight.index_select(0, indices))
			else:
				_wemb.weight = wemb_weight
			if self.classifier.weight.is_set_to(self.wemb.weight):
				_classifier.weight = _wemb.weight
			else:
				_classifier.weight.copy_(self.classifier.weight.index_select(0, indices))
			_classifier.bias.copy_(self.classifier.bias.index_select(1, indices))
		self.wemb, self.classifier = _wemb, _classifier
