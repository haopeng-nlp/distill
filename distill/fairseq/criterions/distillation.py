# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterionConfig
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss

n = 0

import random

def distillation_loss(
    student_lprobs,
    teacher_lprobs,
    target,
    ignore_index=None,
    reduce=True
):
    loss = F.kl_div(student_lprobs, teacher_lprobs, reduction="none", log_target=True)
    # TODO
    loss = torch.sum(loss, dim=-1)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        loss.masked_fill_(pad_mask, 0.0)
        # print(torch.count_nonzero(pad_mask), pad_mask.size())
    else:
        loss = loss.squeeze(-1)
    if reduce:
        loss = loss.sum()
    return loss


@dataclass
class DistillationConfig(LabelSmoothedCrossEntropyCriterionConfig):
    distillation: float = field(
        default=0.0,
         metadata={"help": "epsilon for label smoothing, 0 means no distillation"},
    )


@register_criterion("distillation", dataclass=DistillationConfig)
class Distillation(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        distillation,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.coef = distillation

    def get_lprobs_and_target(
        self, 
        *,
        model, 
        teacher_output,
        student_output,
        sample,
        debug,
        ignore_index,
    ):
        student_lprobs = model.get_normalized_probs(student_output, log_probs=True)
        target = model.get_targets(sample, student_output)
        if self.ignore_prefix_size > 0:
            if getattr(student_lprobs, "batch_first", False):
                student_lprobs = student_lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                student_lprobs = student_lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        student_lprobs = student_lprobs.view(-1, student_lprobs.size(-1))
        target = target.view(-1)

        teacher_lprobs = None
        if teacher_output is not None:

            teacher_lprobs = self.add_noise(model, teacher_output, "gaussian", [0.0, 0.01/30_000], debug, pad_mask=target.eq(ignore_index))

            if self.ignore_prefix_size > 0:
                if getattr(teacher_lprobs, "batch_first", False):
                    teacher_lprobs = teacher_lprobs[:, self.ignore_prefix_size :, :].contiguous()
                else:
                    teacher_lprobs = teacher_lprobs[self.ignore_prefix_size :, :, :].contiguous()
            teacher_lprobs = teacher_lprobs.view(-1, teacher_lprobs.size(-1))
        return student_lprobs, teacher_lprobs, target
    
    def add_noise(
        self,
        model,
        teacher_output,
        noise_type,
        coefs,
        debug,
        pad_mask,
    ):
        """returns teacher_lprobs with noise"""
        global n
        (teacher_logits, *unused_params) = teacher_output
        teacher_logits = teacher_logits.to(torch.float32)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        if noise_type == "gaussian":
            mean, std = coefs
            noise = mean + std * torch.randn(teacher_probs.size(), device=teacher_probs.device)
            noise = torch.clamp(noise, min=1e-15, max=std)
            teacher_probs_noise = teacher_probs + noise
            teacher_probs_noise /= teacher_probs_noise.sum(dim=-1, keepdim=True)
            teacher_lprobs = torch.log(teacher_probs_noise)
            if debug:
                teacher_lprobs_no_noise = torch.log(teacher_probs)
                print(torch.mean((teacher_lprobs - teacher_lprobs_no_noise) ** 2))
        elif noise_type == "topk_uniform":
            k, = coefs
            topk_values, topk_indices = torch.topk(teacher_probs, k=k, dim=-1)
            # zero all elements of teacher_probs except topk
            teacher_probs_topk = torch.zeros_like(teacher_probs).scatter_(-1, topk_indices, topk_values)  # TODO: mask
            if debug:
                print(teacher_probs_topk)
            topk_sum = topk_values.sum(dim=-1, keepdim=True)
            unif = (1 - topk_sum) / (teacher_probs.size(dim=-1) - k)
            unif = unif.repeat(1, 1, teacher_probs.size(dim=-1)).scatter_(-1, topk_indices, 0)
            if debug:
                print(unif)
            teacher_probs_topk += unif
            if debug:
                print(teacher_probs_topk.sum(dim=-1))
            teacher_lprobs = torch.log(teacher_probs_topk)
        elif noise_type == "topk":
            k, = coefs
            topk_values, topk_indices = torch.topk(teacher_probs, k=k, dim=-1)
            # zero all elements of teacher_probs except topk
            teacher_probs_topk = torch.zeros_like(teacher_probs).scatter_(-1, topk_indices, topk_values)
            teacher_probs_topk /= teacher_probs_topk.sum(dim=-1, keepdim=True)
            teacher_lprobs = torch.log(teacher_probs_topk)
        elif noise_type == "temp":
            high_probs = (teacher_probs > 0.5).float()
            low_probs = F.dropout(torch.ones_like(teacher_probs, device=teacher_probs.device), p=0.9).bool().float() * (1 - high_probs)

            noise_high_probs = torch.zeros_like(teacher_probs, device=teacher_probs.device).uniform_(-0.5, 0.0)
            noise_high_probs *= high_probs
            noise_low_probs = torch.abs(torch.sum(noise_high_probs, dim=-1, keepdim=True)) / torch.sum(low_probs, dim=-1, keepdim=True).repeat(1, 1, teacher_probs.size(-1))
            noise_low_probs *= low_probs
            
            teacher_probs_noise = torch.clone(teacher_probs)
            teacher_probs_noise += noise_high_probs + noise_low_probs
            return teacher_probs_noise
        else:  # teacher dropout
            teacher_lprobs = model.get_normalized_probs(teacher_output, log_probs=True)
            if debug and n < 100:
                print("here")
                teacher_probs = teacher_probs.view(-1, teacher_probs.size(-1))
                print(teacher_probs)
                teacher_probs.masked_fill_(torch.unsqueeze(pad_mask, 1).repeat(1, teacher_probs.size(-1)), 0.0)
                print(teacher_probs)
                torch.save(teacher_probs, f"probs/tdrop{model.teacher_dropout}-{n}.pt")
                n += 1
        return teacher_lprobs
    
    def compute_loss(
        self,
        model,
        teacher_output,
        student_output,
        sample,
        reduce=True,
        debug=False,
    ):
        student_lprobs, teacher_lprobs, target = self.get_lprobs_and_target(
            model=model,
            teacher_output=teacher_output,
            student_output=student_output,
            sample=sample,
            debug=debug,
            ignore_index=self.padding_idx,
        )
        loss, nll_loss = label_smoothed_nll_loss(
            student_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        kl_loss = distillation_loss(
            student_lprobs,
            teacher_lprobs,
            target,
            ignore_index=self.padding_idx,
            reduce=reduce
        ) if model.training else 0.0
        return (1.0 - self.coef) * loss + self.coef * kl_loss, nll_loss

    def forward(self, model, sample, reduce=True, debug=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        global n
        if debug and n < 100:
            torch.save(sample["target"], f"target/tdrop{model.teacher_dropout}-{n}.pt")

        teacher_output, student_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(
            model,
            teacher_output,
            student_output,
            sample,
            reduce=reduce,
            debug=debug,
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, student_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output
