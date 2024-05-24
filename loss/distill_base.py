# encoding: utf-8

import torch
from torch.nn.functional import cross_entropy, kl_div, nll_loss
from torch.nn.modules.loss import CrossEntropyLoss as CrossEntropyLossBase, NLLLoss as NLLLossBase, _Loss

from utils.base import clear_pad_mask, eq_indexes
from loss.kd import PearsonCorr as new_loss
from cnfg.ihyp import *
from cnfg.vocab.base import pad_id



class distillloss(_Loss):
    '''
    输入的是没有经过softmax的结果和经过softmax之前除以T的结果
	教师模型的输出soft：output_teach = [bsize, seql-1, nword] -->soft labels --> [0.1, 0.2, 0.1, 0.2, 0.4]
	学生模型的输出soft：output_stu_soft = [bsize, seql-1, nword] -->soft labels --> [0.1, 0.2, 0.3, 0.1, 0.3]
	target = [bsize,seql-1]
	'''

    def __init__(self, nclass, T, label_smoothing=0.1, ignore_index=-1, reduction="mean", **kwargs):
        super(distillloss, self).__init__()
        self.ignore_index, self.smoothing_value, self.reduction = ignore_index, label_smoothing / (
                nclass - 1), reduction
        self.conf = 1.0 - label_smoothing - self.smoothing_value
        self.A = 1
        self.B = 0.5
        self.nword = nclass
        self.softloss = torch.nn.KLDivLoss(reduction="sum")
        self.crossloss = torch.nn.CrossEntropyLoss(reduction="sum") # fastlabelsmothloss
        self.fastloss = FastLabelSmoothingLoss(nclass,reduction="sum")
        self.newloss = new_loss(reduction="sum")


    def forward(self, cross_loss, teach_out_softmax_withoutT, stu_out_softmax_withoutT,
                teach_out_sotfmax_byT, stu_out_sotfmax_byT, teach_out, stu_out, target, use_fast=True,use_new=True, mask=None, **kwargs):
        # 两个软标签
        input_tsize = list(teach_out.size())
        input_tsize[-1] = 1

        class_num = teach_out.size(-1)

        target_tsize = list(target.size())
        target_tsize.append(class_num)
        _target = target.view(input_tsize)  # teach_target = [bsize,seql-1,1] 对应最后一位是预测值的正确结果

        # 找到对应target位置的概率差，判断是否大于某一个界限,[bsize,seql-1,1]
        target_sub_prob = teach_out_softmax_withoutT.gather(dim=-1, index=_target) - stu_out_softmax_withoutT.gather(
            dim=-1, index=_target)  # softmax
        # print("target_sub_prob",target_sub_prob)
        # TF_target_sub = torch.gt(target_sub_prob, 0.2)  # TF_target_sub = [bsize,seql-1,1]

        TF_target_sub = torch.le(target_sub_prob, 0.0)  # TF_target_sub = [bsize,seql-1,1],<0.2的置为True
        # repeat_TF_int = TF_target_sub.repeat(1, 1, self.nword).int()  # 得到的这个矩阵用于最后是否保留教师模型的loss使用
        # repeat_TF_int = TF_target_sub.repeat(1, 1, self.nword).float() #?需要float吗
        #mask掉预测错误的token,得到教师模型预测结果的最大位置的索引index
        max_values, max_index = torch.max(teach_out_softmax_withoutT, dim=-1, keepdim=True)
        mask_wrong_filled = torch.ne(max_index, _target) #不相等置为True,需要给他mask掉，相等为False
        _mask =TF_target_sub | mask_wrong_filled

        trans = stu_out.argmax(-1).view(input_tsize)
        padding_id = 0
        data_mask = _target.ne(padding_id)
        total_batch_word_num = data_mask.int().sum().item()
        _teach_predict_useful = ((~_mask) & data_mask).int().sum().item()

        # student_loss = self.crossloss(stu_out.view(-1, stu_out.size(-1)), target.view(-1))
        # student_loss = self.fastloss(stu_out, target)
        if use_fast:
            student_loss = self.fastloss(stu_out, target)
        else:
            student_loss = self.crossloss(stu_out.view(-1, stu_out.size(-1)), target.view(-1))

        if _teach_predict_useful == 0:
            _loss = student_loss
        else:
            if use_new:
                distill_loss = self.newloss(stu_out_softmax_withoutT, teach_out_softmax_withoutT)
                _loss = student_loss * self.B + ((distill_loss / _teach_predict_useful) * total_batch_word_num) * self.A
            else:
                distill_loss = self.softloss(stu_out_sotfmax_byT.log(), teach_out_sotfmax_byT.masked_fill(_mask, 0.0))
                _loss = student_loss * self.B  + ((distill_loss / _teach_predict_useful) * total_batch_word_num) * self.A
        return _loss


# ignores forbidden_index
class FastLabelSmoothingLoss(_Loss):

    def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction="mean", **kwargs):

        super(FastLabelSmoothingLoss, self).__init__()
        self.ignore_index, self.smoothing_value, self.reduction = ignore_index, label_smoothing / (
                nclass - 1), reduction
        self.conf = 1.0 - label_smoothing - self.smoothing_value

    # Faster implementation from fairseq: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py#L33-L50, but do not support fbil.
    def forward(self, input, target, mask=None, **kwargs):
        # target = [bsize,seql-1]
        # _target = [bsize,seql-1,1]
        # input = [bsize, seql-1, t_vocab] --->输入的是每个位置上每个单词对应的概率
        # 将target扩充至[bsize,seql-1,1]
        _tsize = list(input.size())
        _tsize[-1] = 1
        _target = target.view(_tsize)
        nll_loss = -input.gather(dim=-1, index=_target)
        smooth_loss = -input.sum(dim=-1, keepdim=True)
        _pad_mask = mask
        if _pad_mask is None:
            if isinstance(self.ignore_index, (list, tuple,)):
                _pad_mask = eq_indexes(_target, self.ignore_index)
            elif self.ignore_index >= 0:
                _pad_mask = (_target == self.ignore_index)
        else:
            _pad_mask = _pad_mask.view(_tsize)
        if _pad_mask is not None:
            nll_loss.masked_fill_(_pad_mask, 0.0)
            smooth_loss.masked_fill_(_pad_mask, 0.0)
        if self.reduction != "none":
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        loss = self.conf * nll_loss + self.smoothing_value * smooth_loss
        if self.reduction == "mean":
            loss = loss / float(target.numel())

        return loss


class new_FastLabelSmoothingLoss(_Loss):

    def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction="mean", **kwargs):

        super(new_FastLabelSmoothingLoss, self).__init__()
        '''
		self.ignore_index = ignore_index = pad_id = 0
		self.smoothing_value = label_smoothing / (nclass - 1) = 0.1/(nword-1)
		self.reduction = reduction
		'''
        self.ignore_index, self.smoothing_value, self.reduction = ignore_index, label_smoothing / (
                nclass - 1), reduction
        # self.cpmf = 1 - 0.1 - 0.1/(nword-1)
        self.conf = 1.0 - label_smoothing - self.smoothing_value

    # Faster implementation from fairseq: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py#L33-L50, but do not support fbil.
    # forward 前向传播时如下：loss = lossf(output, ot)
    # output = []
    # output=  (bsize, nquery ,t_word)
    # input (bsize, nquery-1) target (bsize, nquery-1)
    def forward(self, input, target, weight_batch, mask=None):
        '''
		print(weight_batch.size(),input.size())
		'''
        _tsize = list(input.size())
        _bsize, _seql = _tsize[0], _tsize[1]
        _weight = weight_batch.unsqueeze(dim=0).unsqueeze(dim=0)
        _weight = _weight.repeat(_bsize, _seql, 1)  # 此时weight的形状为[bsize,seql,nword] 跟Input完全对应上
        _tsize[-1] = 1
        # target = [bsize,seql-1]
        # _target = [bsize,seql-1,1]
        # input = [bsize, seql-1, t_vocab] --->输入的是每个位置上每个单词对应的概率
        # 将target扩充至[bsize,seql-1,1]
        _target = target.view(_tsize)  # 每个位置上都是同样的目标词语对应的id
        nll_loss = -input.gather(dim=-1, index=_target)
        nll_weight = _weight.gather(dim=-1, index=_target)  # 这里得到的weight
        nword_weight_sum = nll_weight.float().sum().item()  # 返回得到对应权重和，其实就是	“多少个词”
        nll_loss = nll_loss.mul(nll_weight)  # 对应位置权重相乘
        # input 的结构为[bsize,seql-1,nword],index = _target = [bsize,seql-1,1],目的为在nword中找到目标单词的索引对应的概率，即正确单词的概率
        # 得到的nll_loss为 [bsize ,seql-1, 1]
        # smooth_loss = -input.sum(dim=-1, keepdim=True)#?????smooth_loss = [bsize,seql-1,1]
        '''
		print(_weight.size(), input.size())
		'''
        weight_input = input.mul(_weight)
        smooth_loss = -weight_input.sum(dim=-1, keepdim=True)  # ?????smooth_loss = [bsize,seql-1,1]
        # smooth_loss = -weight_input.sum(dim=-1, keepdim=True).mul(_weight)  # ?????smooth_loss = [bsize,seql-1,1]
        _pad_mask = mask
        if _pad_mask is None:
            if isinstance(self.ignore_index, (list, tuple,)):
                _pad_mask = eq_indexes(_target, self.ignore_index)  # _pad_mask = [false,false......true,true]
            elif self.ignore_index >= 0:
                _pad_mask = (_target == self.ignore_index)
        else:
            _pad_mask = _pad_mask.view(_tsize)
        if _pad_mask is not None:
            nll_loss.masked_fill_(_pad_mask, 0.0)  # 将nll_loss中对应pad_mask位置上true的位置，即padding位置置为0，就是预测这个地方为0
            # nll_loss = [ [ [0.1],[0.3],[0.2],[0.0],[0.0] ] , [ [0.2], [0.3], [0.0], [0.0], [0.0] ] ]
            smooth_loss.masked_fill_(_pad_mask, 0.0)  # 将smooth_loss中对应pad_mask位置上true的位置，即padding位置置为0，就是预测这个地方为0
        if self.reduction != "none":
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        loss = self.conf * nll_loss + self.smoothing_value * smooth_loss
        if self.reduction == "mean":
            # loss = loss/float(nll_weight.numel())
            loss = loss / float(target.numel())

        return loss,nword_weight_sum


"""	from: Rethinking the Inception Architecture for Computer Vision (https://arxiv.org/abs/1512.00567)
	With label smoothing, KL-divergence between q_{smoothed ground truth prob.}(w) and p_{prob. computed by model}(w) is minimized.
"""


class StdLabelSmoothingLoss(_Loss):

    def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction="mean", forbidden_index=-1, **kwargs):

        super(StdLabelSmoothingLoss, self).__init__()

        self.reduction = reduction

        fbil = set()
        if isinstance(ignore_index, (list, tuple,)):
            tmp = []
            for _tmp in ignore_index:
                if (_tmp >= 0) and (_tmp not in fbil):
                    tmp.append(_tmp)
                    fbil.add(_tmp)
            _nid = len(tmp)
            if _nid > 0:
                self.ignore_index = tuple(tmp) if _nid > 1 else tmp[0]
            else:
                self.ignore_index = ignore_index[0] if len(ignore_index) > 0 else -1
        else:
            self.ignore_index = ignore_index
            if (ignore_index >= 0) and (ignore_index not in fbil):
                fbil.add(ignore_index)

        if isinstance(forbidden_index, (list, tuple,)):
            for fi in forbidden_index:
                if (fi >= 0) and (fi not in fbil):
                    fbil.add(fi)
        else:
            if forbidden_index is not None and forbidden_index >= 0:
                fbil.add(forbidden_index)

        smoothing_value = label_smoothing / (nclass - 1 - len(fbil))

        weight = torch.full((nclass,), smoothing_value)
        if fbil:
            weight.index_fill_(0, torch.as_tensor(tuple(fbil), dtype=torch.long, device=weight.device), 0.0)
        self.register_buffer("weight", weight.unsqueeze(0), persistent=False)
        self.conf = 1.0 - label_smoothing

    # input: (batch size, num_classes)
    # target: (batch size)
    # they will be flattened automatically if the dimension of input is larger than 2.

    def forward(self, input, target, mask=None, **kwargs):

        _input = input.view(-1, input.size(-1)) if input.dim() > 2 else input
        _target = target.view(-1, 1)
        model_prob = self.weight.repeat(_target.size(0), 1)
        model_prob.scatter_(1, _target, self.conf)

        _pad_mask = mask
        if _pad_mask is None:
            if isinstance(self.ignore_index, (list, tuple,)):
                _pad_mask = eq_indexes(_target, self.ignore_index)
            elif self.ignore_index >= 0:
                _pad_mask = _target.eq(self.ignore_index)
        else:
            _pad_mask = _pad_mask.view(-1, 1)
        if _pad_mask is not None:
            model_prob.masked_fill_(_pad_mask, 0.0)

        rs = kl_div(_input, model_prob, reduction=self.reduction)

        return rs.view(input.size()) if self.reduction == "none" and target.dim() > 1 else rs


LabelSmoothingLoss = new_FastLabelSmoothingLoss if use_fast_loss else StdLabelSmoothingLoss


class NLLLoss(NLLLossBase):

    def forward(self, input, target, **kwargs):
        rs = nll_loss(input.view(-1, input.size(-1)), target.view(-1), weight=self.weight,
                      ignore_index=self.ignore_index, reduction=self.reduction)

        return rs.view(input.size()) if self.reduction == "none" and target.dim() > 1 else rs


class CrossEntropyLoss(CrossEntropyLossBase):

    def forward(self, input, target, **kwargs):
        rs = cross_entropy(input.view(-1, input.size(-1)), target.view(-1), weight=self.weight,
                           ignore_index=self.ignore_index,
                           reduction=self.reduction)  # , label_smoothing=self.label_smoothing

        return rs.view(input.size()) if self.reduction == "none" and target.dim() > 1 else rs


class RankingLoss(_Loss):

    # input: (batch size)
    # target: (batch size)
    def forward(self, input, target, **kwargs):
        loss = input * target
        if self.reduction == "mean":
            loss = loss / loss.numel()

        return loss


class MultiLabelSmoothingLoss(_Loss):

    def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction="mean", forbidden_index=-1, **kwargs):

        super(MultiLabelSmoothingLoss, self).__init__()

        self.reduction = reduction

        fbil_common = set()
        if isinstance(ignore_index, (list, tuple,)):
            tmp = []
            for _tmp in ignore_index:
                if (_tmp >= 0) and (_tmp not in tmp):
                    tmp.append(_tmp)
                    if _tmp not in fbil_common:
                        fbil_common.add(_tmp)
            _nid = len(tmp)
            if _nid > 0:
                self.ignore_index = tuple(tmp) if _nid > 1 else tmp[0]
            else:
                self.ignore_index = ignore_index[0] if len(ignore_index) > 0 else -1
        else:
            self.ignore_index = ignore_index
            if (ignore_index >= 0) and (ignore_index not in fbil_common):
                fbil_common.add(ignore_index)

        fbil = []
        for fbilu in forbidden_index:
            tmp = set()
            if isinstance(fbilu, (list, tuple,)):
                for fi in fbilu:
                    if (fi >= 0) and (fi not in tmp):
                        tmp.add(fi)
            else:
                if fbilu is not None and fbilu >= 0:
                    tmp.add(forbidden_index)
            tmp |= fbil_common
            fbil.append(tmp)

        _weight = []
        for fbilu in fbil:
            smoothing_value = label_smoothing / (nclass - 1 - len(fbilu))
            _tmp_w = torch.full((nclass,), smoothing_value)
            if fbilu:
                _tmp_w.index_fill_(0, torch.as_tensor(tuple(fbilu), dtype=torch.long, device=_tmp_w.device), 0.0)
            _weight.append(_tmp_w)
        self.register_buffer("weight", torch.stack(_weight, 0).unsqueeze(1), persistent=False)
        self.conf = 1.0 - label_smoothing

    def forward(self, input, target, lang_id=0, mask=None, **kwargs):

        _input = input.view(-1, input.size(-1)) if input.dim() > 2 else input
        _target = target.view(-1, 1)

        model_prob = self.weight[lang_id].repeat(_target.size(0), 1)
        model_prob.scatter_(1, _target, self.conf)

        _pad_mask = mask
        if _pad_mask is None:
            if isinstance(self.ignore_index, (list, tuple,)):
                _pad_mask = eq_indexes(_target, self.ignore_index)
            elif self.ignore_index >= 0:
                _pad_mask = _target.eq(self.ignore_index)
        else:
            _pad_mask = _pad_mask.view(-1, 1)
        if _pad_mask is not None:
            model_prob.masked_fill_(_pad_mask, 0.0)

        rs = kl_div(_input, model_prob, reduction=self.reduction)

        return rs.view(input.size()) if self.reduction == "none" and target.dim() > 1 else rs


class ReducedLabelSmoothingLoss(StdLabelSmoothingLoss):

    def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction="mean", forbidden_index=-1,
                 reduce_dim=None, pad_id=pad_id, **kwargs):

        super(ReducedLabelSmoothingLoss, self).__init__(nclass, label_smoothing=label_smoothing,
                                                        ignore_index=ignore_index, reduction=reduction,
                                                        forbidden_index=forbidden_index)

        self.reduce_dim, self.pad_id = reduce_dim, pad_id

    def forward(self, input, target, mask=None, pad_id=None, **kwargs):

        if self.reduce_dim is not None:
            input, target = clear_pad_mask([input, target], target.eq(self.pad_id if pad_id is None else pad_id),
                                           [self.reduce_dim - 1, self.reduce_dim], mask_dim=self.reduce_dim,
                                           return_contiguous=True)[0]

        _input = input.view(-1, input.size(-1)) if input.dim() > 2 else input
        _target = target.view(-1, 1)

        model_prob = self.weight.repeat(_target.size(0), 1)
        model_prob.scatter_(1, _target, self.conf)

        _pad_mask = mask
        if _pad_mask is None:
            if isinstance(self.ignore_index, (list, tuple,)):
                _pad_mask = eq_indexes(_target, self.ignore_index)
            elif self.ignore_index >= 0:
                _pad_mask = _target.eq(self.ignore_index)
        else:
            _pad_mask = _pad_mask.view(-1, 1)
        if _pad_mask is not None:
            model_prob.masked_fill_(_pad_mask, 0.0)

        rs = kl_div(_input, model_prob, reduction=self.reduction)

        return rs.view(input.size()) if self.reduction == "none" and target.dim() > 1 else rs
