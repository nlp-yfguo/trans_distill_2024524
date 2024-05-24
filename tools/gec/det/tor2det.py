#encoding: utf-8

import sys

from utils.fmt.base import sys_open

from cnfg.vocab.gec.det import correct_id, incorrect_id

def handle(srcf, tgtf, rssf, rstf, unbalance=True):

	ens, c_ens, i_ens = "\n".encode("utf-8"), ("%d\n" % correct_id).encode("utf-8"), ("%d\n" % incorrect_id).encode("utf-8")
	with sys_open(srcf, "rb") as frds, sys_open(tgtf, "rb") as frdt, sys_open(rssf, "wb") as fwrts, sys_open(rstf, "wb") as fwrtt:
		for s, t in zip(frds, frdt):
			_s, _t = s.strip(), t.strip()
			if _s and _t:
				if _s != _t:
					fwrts.write(_s)
					fwrts.write(ens)
					fwrtt.write(i_ens)
					fwrts.write(_t)
					fwrts.write(ens)
					fwrtt.write(c_ens)
				elif unbalance:
					fwrts.write(_t)
					fwrts.write(ens)
					fwrtt.write(c_ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
