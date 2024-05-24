#encoding: utf-8

from transformer.NMT import NMT as NMTBase
from transformer.SelfKD.FeatDecoder import Decoder
from transformer.SelfKD.FeatEncoder import Encoder
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, kd_layers=None, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		_kd_layers = [] if kd_layers is None else kd_layers
		if _kd_layers and isinstance(_kd_layers[0], (list, tuple)):
			kd_enc_layers, kd_dec_layers = tuple(set(_) for _ in _kd_layers)
		else:
			kd_enc_layers = kd_dec_layers = set(_kd_layers)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, kd_layers=kd_enc_layers)

		emb_w = self.enc.wemb.weight if global_emb else None
		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index, kd_layers=kd_dec_layers)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, mask=None, gold=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

		if self.training and (gold is not None):
			ence, _enc_kd_loss = self.enc(inpute, mask=_mask, gold=gold)
			out, _dec_kd_loss = self.dec(ence, inputo, src_pad_mask=_mask, gold=gold)
			return out, _enc_kd_loss + _dec_kd_loss
		else:
			return self.dec(self.enc(inpute, mask=_mask), inputo, src_pad_mask=_mask)
