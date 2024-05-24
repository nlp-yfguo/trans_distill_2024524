# encoding: utf-8

import sys


import numpy as np

from utils.tqdm import tqdm

from utils.h5serial import h5File


from cnfg.ihyp import *
'''from utils.fmt.dual import batch_padder,freq_batch_padder
from transformer.ori_NMT import NMT
from transformer.EnsembleNMT import NMT as Ensemble
from parallel.parallelMT import DataParallelMT
import cnfg.base as cnfg
import torch
from utils.fmt.base4torch import parse_cuda_decode
from utils.fmt.base import tostr, pad_id,unk_id,sos_id,eos_id'''
from utils.base import *
from utils.fmt.vocab.token import ldvocab,ldvocab_list,ld_freq_vcb,save_vocab,ld_weight_vcb
from utils.fmt.vocab.base import reverse_dict
# from utils.fmt.base import ldvocab, reverse_dict, eos_id
from utils.fmt.base import clean_list_iter



def load_fixing(module):
    if hasattr(module, "fix_load"):
        module.fix_load()

def id2token(src,vcb):
    result = []
    for id in src:
        cd = vcb[id]
        result.append(cd)
    return result

def handles(file1,file2,file3,file4,file5,file6,src_freq_dict,sort_tgt_freq_dict):
    test_len = len(sort_tgt_freq_dict)
    high_id, low_id = test_len / 3, test_len * 2 / 3
    id = 0
    with open(file1,'w',encoding='utf8') as src_high_file,open(file2,'w',encoding='utf8') as tgt_high_file, open(file3,'w',encoding='utf8') as src_middle_file,open(file4,'w',encoding='utf8') as tgt_middle_file, open(file5,'w',encoding='utf8') as src_low_file,open(file6,'w',encoding='utf8') as tgt_low_file:
        for k in sort_tgt_freq_dict.keys():
            if id < high_id:
                src_sentence_id ,tgt_sentence_id = src_freq_dict[k] ,sort_tgt_freq_dict[k]
                src_sentence_token,tgt_sentence_token = id2token(src_sentence_id,rev_vcbt),id2token(tgt_sentence_id,rev_vcbt)
                for token in src_sentence_token:
                    src_high_file.write(token)
                    src_high_file.write(' ')
                src_high_file.write('\n')
                for token1 in tgt_sentence_token:
                    tgt_high_file.write(token1)
                    tgt_high_file.write(' ')
                tgt_high_file.write('\n')
            if id >= high_id and id < low_id:
                src_sentence_id ,tgt_sentence_id = src_freq_dict[k] ,sort_tgt_freq_dict[k]
                src_sentence_token,tgt_sentence_token = id2token(src_sentence_id,rev_vcbt),id2token(tgt_sentence_id,rev_vcbt)
                for token2 in src_sentence_token:
                    src_middle_file.write(token2)
                    src_middle_file.write(' ')
                src_middle_file.write('\n')
                for token3 in tgt_sentence_token:
                    tgt_middle_file.write(token3)
                    tgt_middle_file.write(' ')
                tgt_middle_file.write('\n')
            if id >= low_id:
                src_sentence_id, tgt_sentence_id = src_freq_dict[k], sort_tgt_freq_dict[k]
                src_sentence_token, tgt_sentence_token = id2token(src_sentence_id, rev_vcbt), id2token(tgt_sentence_id,rev_vcbt)
                for token4 in src_sentence_token:
                    src_low_file.write(token4)
                    src_low_file.write(' ')
                src_low_file.write('\n')
                for token5 in tgt_sentence_token:
                    tgt_low_file.write(token5)
                    tgt_low_file.write(' ')
                tgt_low_file.write('\n')
            id += 1


# src_test_data = "C:/Users/GuoYifan/Desktop/transformer_base/test/src.test.bpe"        #sys.argv[1]
# # true_test_data = "C:/Users/GuoYifan/Desktop/transformer_base/test/src.test.en"
#
# tgt_test_data = "C:/Users/GuoYifan/Desktop/transformer_base/test/tgt.test.bpe"            #sys.argv[2]
# true_tgt_test_data = "C:/Users/GuoYifan/Desktop/transformer_base/test/tgt.test.de.tok"            #sys.argv[3]



# vcbt, nwordt = ldvocab(sys.argv[2]) # <pad>:0
vcbt, nwordt = ldvocab("/home/yfguo/Data_Cache/wmt16/wmt16cache/rs_6144/common.vcb") # <pad>:0
rev_vcbt = reverse_dict(vcbt) # 0:<pad>

# freq_vcbt,_nwordt = ld_freq_vcb(sys.argv[2])# 'a:1
freq_vcbt,_nwordt = ld_freq_vcb("/home/yfguo/Data_Cache/wmt16/wmt16cache/rs_6144/common.vcb")# 'a:1
# print(freq_vcbt)

Count_sum = sum(freq_vcbt.values())- 6

src_sentence_dict = {}
tgt_sentence_dict = {}
true_tgt_sentence_dict = {}
with open(sys.argv[1],'r',encoding='utf-8') as src_test, open(sys.argv[2],'r',encoding='utf-8') as tgt_test,open(sys.argv[3],'r',encoding='utf-8') as token_tgt:
    for line in tgt_test:
        sum = 0
        tmp = line.strip()
        sentence_len = len(tmp.split())
        if tmp:
            for token in clean_list_iter(tmp.split()):
                token_freq = freq_vcbt[token]
                _d = np.log(token_freq / Count_sum)
                sum = sum + _d
            d_rarity = - (sum / sentence_len)
        tgt_sentence_dict[d_rarity] = line
    tgt_id_list = []
    for k in tgt_sentence_dict.keys():
        tgt_id_list.append(k)
    i = 0
    for line in src_test:
        src_sentence_dict[tgt_id_list[i]] = line
        i += 1
    j = 0
    for line in token_tgt:
        true_tgt_sentence_dict[tgt_id_list[j]] = line
        j += 1
    with open('/home/yfguo/Data_Cache/wmt16/wmt16testcache/wmt16cache/src.txt','w',encoding='utf-8') as src,open('/home/yfguo/Data_Cache/wmt16/wmt16testcache/wmt16cache/tgt.txt','w',encoding='utf-8') as tgt,open('/home/yfguo/Data_Cache/wmt16/wmt16testcache/wmt16cache/dtc_detoken_tgt.txt','w',encoding='utf-8') as true_tgt:
        for lines in src_sentence_dict.values():
            src.write(lines)
        for liness in tgt_sentence_dict.values():
            tgt.write((liness))
        for linesss in true_tgt_sentence_dict.values():
            true_tgt.write(linesss)
    # sort_tgt_sentence_dict = dict(sorted(tgt_sentence_dict.items()))
    sort_tgt_sentence_dict = dict(sorted(true_tgt_sentence_dict.items()))
    # print(sort_tgt_sentence_dict) 从小到大排列

    sentence_num = len(sort_tgt_sentence_dict)
    low_freq_id = sentence_num  / 3
    high_freq_id = sentence_num * 2 / 3
    id = sentence_num
    print(sentence_num,high_freq_id,low_freq_id)
    #with open('C:/Users/GuoYifan/Desktop/transformer_base/test/lsrc.txt','w',encoding='utf-8') as low_src,open('C:/Users/GuoYifan/Desktop/transformer_base/test/msrc.txt','w',encoding='utf-8') as midlle_src,open('C:/Users/GuoYifan/Desktop/transformer_base/test/hsrc.txt','w',encoding='utf-8') as high_src,open('C:/Users/GuoYifan/Desktop/transformer_base/test/ltgt.txt','w',encoding='utf-8') as low_tgt,open('C:/Users/GuoYifan/Desktop/transformer_base/test/mtgt.txt','w',encoding='utf-8') as midlle_tgt,open('C:/Users/GuoYifan/Desktop/transformer_base/test/htgt.txt','w',encoding='utf-8') as high_tgt:
    # with open('C:/Users/GuoYifan/Desktop/transformer_base/test/lsrc.txt', 'w', encoding='utf-8') as low_src, open(
    #         'C:/Users/GuoYifan/Desktop/transformer_base/test/msrc.txt', 'w', encoding='utf-8') as midlle_src, open(
    #         'C:/Users/GuoYifan/Desktop/transformer_base/test/hsrc.txt', 'w', encoding='utf-8') as high_src, open(
    #         'C:/Users/GuoYifan/Desktop/transformer_base/test/ltgt.txt', 'w', encoding='utf-8') as low_tgt, open(
    #         'C:/Users/GuoYifan/Desktop/transformer_base/test/mtgt.txt', 'w', encoding='utf-8') as midlle_tgt, open(
    #         'C:/Users/GuoYifan/Desktop/transformer_base/test/htgt.txt', 'w', encoding='utf-8') as high_tgt:
    with open(sys.argv[4], 'w', encoding='utf-8') as low_src, open(sys.argv[5], 'w', encoding='utf-8') as midlle_src, open(sys.argv[6], 'w', encoding='utf-8') as high_src, open(sys.argv[7], 'w', encoding='utf-8') as low_tgt, open(sys.argv[8], 'w', encoding='utf-8') as midlle_tgt, open(sys.argv[9], 'w', encoding='utf-8') as high_tgt:
        for k,v in sort_tgt_sentence_dict.items():
            if id >=0 and id < low_freq_id:
                low_tgt.write(v)
                low_src.write(src_sentence_dict[k])
            if id >= low_freq_id and id < high_freq_id:
                midlle_tgt.write(v)
                midlle_src.write(src_sentence_dict[k])
            if id >= high_freq_id:
                high_tgt.write(v)
                high_src.write(src_sentence_dict[k])
            id -= 1




