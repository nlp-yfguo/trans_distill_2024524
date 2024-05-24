#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, list_reader as file_reader
from utils.fmt.dual import batch_padder as batch_padder_base

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rst = []
	nd = maxlen = mlen_i = mlen_t = 0
	for i_d, td in zip(file_reader(finput, keep_empty_line=True), file_reader(ftarget, keep_empty_line=True)):
		lid = len(i_d)
		ltd = len(td)
		lgth = lid + ltd
		if maxlen == 0:
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			mlen_i = lid
		if (lid == mlen_i) and ((nd < minbsize) or (lgth <= maxlen and nd < _bsize)):
			rsi.append(i_d)
			rst.append(td)
			if ltd > mlen_t:
				mlen_t = ltd
			nd += 1
		else:
			yield rsi, rst, mlen_i, mlen_t
			rsi = [i_d]
			rst = [td]
			mlen_i = lid
			mlen_t = ltd
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rst, mlen_i, mlen_t

def batch_padder(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, batch_loader=batch_loader, **kwargs):

	return batch_padder_base(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, batch_loader=batch_loader, **kwargs)
