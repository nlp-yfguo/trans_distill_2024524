#encoding: utf-8

from utils.fmt.vocab.base import no_unk_mapper
from utils.fmt.vocab.token import ldvocab as ldvocab_base

from cnfg.vocab.pad import *

def ldvocab(*args, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id, **kwargs):

	return ldvocab_base(*args, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id, **kwargs)

def map_batch_core(i_d, vocabi, use_unk=use_unk, unk_id=unk_id, **kwargs):

	if isinstance(i_d[0], (tuple, list,)):
		return [map_batch_core(idu, vocabi, use_unk=use_unk, unk_id=unk_id, **kwargs)[0] for idu in i_d]
	else:
		rsi = [vocabi.get(wd, unk_id) for wd in i_d] if use_unk else no_unk_mapper(vocabi, i_d)
		return rsi

def map_batch(i_d, vocabi, use_unk=use_unk, unk_id=unk_id, **kwargs):

	return map_batch_core(i_d, vocabi, use_unk=use_unk, unk_id=unk_id, **kwargs), 0
