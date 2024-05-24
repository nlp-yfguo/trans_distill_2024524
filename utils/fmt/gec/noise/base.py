#encoding: utf-8

from math import ceil, floor
from random import choices, gauss, randint, sample, shuffle

from utils.fmt.base import sys_open
from utils.fmt.parser import parse_none
from utils.fmt.vocab.char import ldvocab_list
from utils.math import cumsum, pos_norm

from cnfg.vocab.plm.custbert import init_token_id, vocab_size

def load_replace_data(fname):

	rsd = {}
	with sys_open(fname, "rb") as f:
		for _ in f:
			tmp = _.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				if len(tmp) > 1:
					for _c in tmp:
						if _c in rsd:
							rsd[_c].append(tmp)
						else:
							rsd[_c] = [tmp]

	return rsd

def filter_bi_same(samples, uni):

	_a, _b = samples

	return _b if _a == uni else _a

class CharReplacer:

	def __init__(self, df, sample_func=sample):

		self.rpd = load_replace_data(df)
		self.sample_func = sample_func

	def __call__(self, x, sample_func=None, data=None, **kwargs):

		_sample_func, _rpd = parse_none(sample_func, self.sample_func), parse_none(data, self.rpd)
		rs = []
		for _ in x:
			rs.append(filter_bi_same(_sample_func(_sample_func(_rpd[_], 1)[0], 2), _) if _ in _rpd else _)

		return "".join(rs)

class VocabReplacer:

	def __init__(self, df, ls=1.0, sample_func=sample):

		self.rpd = ldvocab_list(df)[0]
		self.ls, self.sample_func = ls, sample_func

	def __call__(self, x, sample_func=None, data=None, **kwargs):

		_sample_func, _rpd = parse_none(sample_func, self.sample_func), parse_none(data, self.rpd)
		_src_s = set(x)
		_rs_len = len(x)
		if self.ls != 0.0:
			_ = gauss(mu=0.0, sigma=self.ls)
			_ = (floor if _ < 0.0 else ceil)(_)
			if _ != 0:
				_rs_len = max(1, _rs_len + _)
		rs = [_ for _ in _sample_func(_rpd, _rs_len + len(_src_s)) if _ not in _src_s]

		return "".join(rs[:_rs_len])

def shuffler(x, **kwargs):

	_ = list(x)
	shuffle(_)

	return "".join(_)

def repeat(x, **kwargs):

	return "%s%s" % (x, x,)

def drop(x, **kwargs):

	return ""

def select_noise_span(spl, l):

	_ = [(i, len(_tmp),) for i, _tmp in enumerate(spl)]
	shuffle(_)
	_ind, _span_len = _[0]
	yield _ind
	_l = l - _span_len
	if _l > 0:
		for _ind, _span_len in _[1:]:
			_l -= _span_len
			if (_l < 0):
				break
			yield _ind

class Noiser:

	def __init__(self, char=None, vcb=None, min_span_len=1, max_span_len=5, p=0.15, w_char=0.2, w_vcb=0.2, w_shuf=0.1, w_repeat=0.1, w_drop=0.1):

		self.edits = []
		w = []
		if char is not None:
			if isinstance(char, str):
				self.edits.append(CharReplacer(char))
				w.append(w_char)
			else:
				self.edits.extend([CharReplacer(_) for _ in char])
				if isinstance(w_char, list):
					w.extend(w_char)
				else:
					_l = len(char)
					_avg = w_char / float(_l)
					w.extend([_avg for _ in range(_l)])
		if vcb is not None:
			self.edits.append(VocabReplacer(vcb))
			w.append(w_vcb)
		if w_shuf > 0.0:
			self.edits.append(shuffler)
			w.append(w_shuf)
		if w_repeat > 0.0:
			self.edits.append(repeat)
			w.append(w_repeat)
		if w_drop > 0.0:
			self.edits.append(drop)
			w.append(w_drop)
		self.sample_cw = cumsum(pos_norm(w))
		self.sample_ind = list(range(len(self.edits)))
		self.min_span_len, self.max_span_len, self.p = min_span_len, max_span_len, p

	def __call__(self, x, **kwargs):

		_r_len = len(x)
		if _r_len == 1:
			return x
		_min_span_len, _max_span_len, _sample_ind, _sample_cw = self.min_span_len, self.max_span_len, self.sample_ind, self.sample_cw
		_corr_len = max(int(_r_len * self.p), 1)
		_min_span_len, _max_span_len = min(_min_span_len, _corr_len), min(_max_span_len, _corr_len)
		_sind = 0
		_spans = []
		while _sind < _r_len:
			_span_len = 1 if _max_span_len == 1 else min(randint(_min_span_len, _max_span_len), _r_len)
			_eind = _sind + _span_len
			_spans.append(x[_sind:_eind])
			_sind = _eind
		_sinds = set(select_noise_span(_spans, _corr_len))
		rs = []
		for _ind, _ in enumerate(_spans):
			if _ind in _sinds:
				rs.append(self.edits[choices(_sample_ind, cum_weights=_sample_cw, k=1)[0]](_, **kwargs))
			else:
				rs.append(_)

		return "".join(rs)
