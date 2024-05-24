#encoding: utf-8

from utils.fmt.vocab.base import no_unk_mapper
from utils.fmt.vocab.token import ldvocab as ldvocab_base

from cnfg.vocab.doc.mono import *

def ldvocab(*args, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id, **kwargs):

	return ldvocab_base(*args, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id, **kwargs)

def map_batch_core(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, cls_id=cls_id, **kwargs):

	if isinstance(i_d[0], (tuple, list,)):
		return [map_batch_core(idu, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, cls_id=cls_id, **kwargs) for idu in i_d]
	else:
		rsi = [cls_id, sos_id]
		rsi.extend([vocabi.get(wd, unk_id) for wd in i_d] if use_unk else no_unk_mapper(vocabi, i_d))#[vocabi[wd] for wd in i_d if wd in vocabi]
		rsi.append(eos_id)
		return rsi

def map_batch(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, cls_id=cls_id, **kwargs):

	return map_batch_core(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, cls_id=cls_id, **kwargs), 3
