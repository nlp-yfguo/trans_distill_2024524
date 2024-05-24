#encoding: utf-8

root_dir = "/home/yfguo/4090data/Data_Cache/"

group_id = "zhen_sigle"

run_id = "zhen_sigletea_300k_252550_A1B05"

data_id = "rs_3072"

exp_dir = "expm/"
cache_dir = "wmt18zhendivide/"

low_train_data = root_dir + cache_dir + data_id + "/train_low.h5"
middle_train_data = root_dir + cache_dir + data_id + "/train_middle.h5"
high_train_data = root_dir + cache_dir + data_id + "/train_high.h5"
dev_data = root_dir + cache_dir + data_id + "/dev.h5"
test_data =root_dir +  cache_dir + data_id + "/test.h5"

fine_tune_m = None

fine_tune_teach = "/home/yfguo/4090data/transformer_924/expm/rs_3072/zhen_low_FT/std/checkpoint_49.h5"
fine_tune_student = "/home/yfguo/4090data/transformer_924/expm/rs_6144_2/zhen_base_100k/std/train_6_2.536_3.016_39.51.h5"

# non-exist indexes in the classifier.
# "<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3
# add 3 to forbidden_indexes if there are <unk> tokens in data
# must be None if use_fast_loss is set in cnfg/hyp.py

#from fbind import fbl

forbidden_indexes = None#[0, 1] + fbl

save_auto_clean = True
overwrite_eva = False
save_every = 2000
num_checkpoint = 50
epoch_start_checkpoint_save = 0

tokens_optm = 25000

earlystop = 128
maxrun = 128
training_steps = 300000

batch_report = 10000
report_eva = True

use_cuda = True
# Data Parallel multi-GPU support can be enabled with values like: "cuda:0, 1, 3". Set to None to use all GPUs.
gpuid = "cuda:0"
use_amp = False
multi_gpu_optimizer = True

bindDecoderEmb = True
share_emb = False

isize = 512
ff_hsize = isize * 4
nhead = max(1, isize // 64)
attn_hsize = isize

nlayer = 6

drop = 0.1
attn_drop = drop
act_drop = drop

# False for Hier/Incept Models
norm_output = True

warm_step = 8000
lr_scale = 1.0

label_smoothing = 0.1

weight_decay = 0

beam_size = 4
length_penalty = 0.0
# use multi-gpu for translating or not. `predict.py` will take the last gpu rather than the first in case multi_gpu_decoding is set to False to avoid potential break due to out of memory, since the first gpu is the main device by default which takes more jobs.
multi_gpu_decoding = False

seed = 666666

epoch_save = False

# to accelerate training through sampling, 0.8 and 0.1 in: Dynamic Sentence Sampling for Efficient Training of Neural Machine Translation
dss_ws = None
dss_rm = None

use_ams = False

src_emb = None
freeze_srcemb = False
tgt_emb = None
freeze_tgtemb = False
scale_down_emb = True

train_statesf = None
save_train_state = False
