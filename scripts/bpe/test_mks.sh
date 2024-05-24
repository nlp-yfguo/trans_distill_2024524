#!/bin/bash

set -e -o pipefail -x

export cachedir=/home/yfguo/Data_Cache/wmt18zhen/

export dataid=rs_6144_2

export srcd=/home/yfguo/Data_Cache/wmt18zhen/rs_6144_2/
export srcvf=newstest2017.tc.zh
export tgtvf=newstest2017.tc.en

export vratio=0.2
export rratio=0.6
export maxtokens=256

export bpeops=32000
export minfreq=8
export share_bpe=false

export tgtd=$cachedir/$dataid

mkdir -p $tgtd

export src_cdsf=$tgtd/src.cds
export tgt_cdsf=$tgtd/tgt.cds

subword-nmt apply-bpe -c $src_cdsf --vocabulary $tgtd/src.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$srcvf > $tgtd/src.test.bpe &
subword-nmt apply-bpe -c $tgt_cdsf --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$tgtvf > $tgtd/tgt.test.bpe &


