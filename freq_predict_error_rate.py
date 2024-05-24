# encoding: utf-8

import sys

import torch

from utils.tqdm import tqdm

from utils.h5serial import h5File

import cnfg.freq_base as cnfg
from cnfg.ihyp import *

from transformer.ori_NMT import NMT
from transformer.EnsembleNMT import NMT as Ensemble
from parallel.parallelMT import DataParallelMT
from utils.io import load_model_cpu
from utils.base import *
from utils.fmt.vocab.token import ldvocab,ldvocab_list,ld_freq_vcb,save_vocab,ld_weight_vcb
from utils.fmt.vocab.base import reverse_dict
from utils.fmt.base4torch import parse_cuda_decode
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode
from cnfg.vocab.base import unk_id,sos_id,eos_id,pad_id



def load_fixing(module):
    if hasattr(module, "fix_load"):
        module.fix_load()


td = h5File(cnfg.test_data, "r")


ntest = td["ndata"][()].item()
nwordi = td["nword"][()].tolist()[0]
vcbt, nwordt = ldvocab(sys.argv[2])
vcbt = reverse_dict(vcbt)

if len(sys.argv) == 4:
    mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop,
                  cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb,
                  cnfg.forbidden_indexes)

    mymodel = load_model_cpu(sys.argv[3], mymodel)
    mymodel.apply(load_fixing)

else:
    models = []
    for modelf in sys.argv[3:]:
        tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop,
                  cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb,
                  cnfg.forbidden_indexes)

        tmp = load_model_cpu(modelf, tmp)
        tmp.apply(load_fixing)

        models.append(tmp)
    mymodel = Ensemble(models)

mymodel.eval()

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)
use_amp = cnfg.use_amp and use_cuda

# Important to make cudnn methods deterministic
set_random_seed(cnfg.seed, use_cuda)

if cuda_device:
    mymodel.to(cuda_device)
    if multi_gpu:
        mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True,
                                 gather_output=False)

beam_size = cnfg.beam_size
length_penalty = cnfg.length_penalty

ens = "\n".encode("utf-8")

src_grp = td["src"]
tgt_grp = td["tgt"]
init_id =2
with open(sys.argv[1], "wb") as f, torch.no_grad():
    w = r = s = h_s = m_s = m_r = h_r = 0
    low_freq = 24000
    high_freq = 12000
    for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
        seq_batch = torch.from_numpy(src_grp[str(i)][()])
        seq_o = torch.from_numpy(tgt_grp[str(i)][()])
        lo = seq_o.size(1) - 1
        if cuda_device:
            seq_batch = seq_batch.to(cuda_device)
            seq_o = seq_o.to(cuda_device)
        seq_batch,seq_o = seq_batch.long(),seq_o.long()
        ot = seq_o.narrow(1, 1, lo).contiguous()
        with torch_autocast(enabled=use_amp):
            output = mymodel(seq_batch, seq_o.narrow(1, 0, lo))
        # output = mymodel.train_decode(seq_batch, beam_size, None, length_penalty)
        if multi_gpu:
            trans = torch.cat([outu.argmax(-1).to(cuda_device) for outu in output], 0)
        else:
            trans = output.argmax(-1)

        data_mask = ot.gt(init_id)#      [true,true,true,true,true,true,false,false,fasle,false]
        # data_eos_mask = ot.ne(eos_id)#  [true,true,true,true,true,false,true,true,true,true] 除掉eos
        # data_sos_mask = ot.ne(sos_id)#  [false,true,true,true,true,true,true,true,true,true] 除掉sos
        # data_unk_mask = ot.ne(unk_id)

        _ot_high_freq = ot.le(high_freq) & data_mask #高频词全置为True，目标里面含有多少高频词
        # _ot_high_freq = _ot_high_freq & data_eos_mask & data_sos_mask
        _ot_low_freq = ot.gt(low_freq) & data_mask  # 大于low_freq的返回为True &pad_mask
        _ot_middle_freq = (~(_ot_high_freq | _ot_low_freq)) & data_mask

        #ot中高频词，中频词分别为：
        h_s += _ot_high_freq.int().sum().item()
        m_s += _ot_middle_freq.int().sum().item()
        s += _ot_low_freq.int().sum().item()

        correct_trans = trans.eq(ot)

        h_s_correct = (correct_trans & _ot_high_freq).int()
        m_s_correct = (correct_trans & _ot_middle_freq).int()
        correct = (correct_trans & _ot_low_freq).int()

        w += data_mask.int().sum().item()

        h_r += h_s_correct.sum().item()
        m_r += m_s_correct.sum().item()
        r += correct.sum().item()
    print("测试集低频词的占比为:",s / float(w) * 100.0)
    print("测试集中频词的占比为:", m_s / float(w) * 100.0)
    print("测试集高频词的占比为:", h_s / float(w) * 100.0)

    print("测试集低频词的正确率为:", r / float(s) * 100.0)
    print("测试集中频词的正确率为:", m_r / float(m_s) * 100.0)
    print("测试集高频词的正确率为:", h_r / float(h_s) * 100.0)

td.close()
