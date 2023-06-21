#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import argparse
import json
import sys
import traceback
from subprocess import SubprocessError
import torch

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

import ast
import fileinput
import logging
import math
import os
import sys
import time
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


def stdio_predictor_wrapper(predictor):
    """
    Wrap a predictor in a loop that reads from stdin and writes to stdout.
    The predictor implements `predict` function that takes a single string and
    returns the label. Assumes each input instance ends with "\n".
    """
    try:
        for line in sys.stdin:
            line = line.rstrip()
            inputs = json.loads(line)
            assert isinstance(inputs, list)
            # Participants need to connect their inference code
            # to our wrapper through the following line.
            outputs = predictor.predict(inputs)
            outputs = list(outputs)
            # Writes are \n deliminated, so adding \n is essential
            # to separate this write from the next loop iteration.
            sys.stdout.write(f"{json.dumps(outputs)}\n")
            # Writes to stdout are buffered.
            # The flush ensures the output is immediately sent through
            # the pipe instead of buffered.
            sys.stdout.flush()
    except:
        sys.stdout.write("Efficiency benchmark exception: SubprocessError\n")
        sys.stdout.write(traceback.format_exc())
        sys.stdout.flush()
        raise SubprocessError


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if cfg.generation.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    if cfg.generation.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )



def main(cfg: FairseqConfig):
    predictor = Predictor(cfg)
    stdio_predictor_wrapper(predictor)


class Predictor():
    def __init__(self, cfg):
        self.cfg = cfg
        if isinstance(cfg, Namespace):
            cfg = convert_namespace_to_omegaconf(cfg)

        utils.import_user_module(cfg.common)

        if cfg.interactive.buffer_size < 1:
            cfg.interactive.buffer_size = 1
        if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
            cfg.dataset.batch_size = 1

        assert (
            not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
        ), "--sampling requires --nbest to be equal to --beam"
        assert (
            not cfg.dataset.batch_size
            or cfg.dataset.batch_size <= cfg.interactive.buffer_size
        ), "--batch-size cannot be larger than --buffer-size"

        self.use_cuda = torch.cuda.is_available() and not cfg.common.cpu
        # Setup task, e.g., translation
        self.task = tasks.setup_task(cfg.task)

        # Load ensemble
        overrides = ast.literal_eval(cfg.common_eval.model_overrides)
        logger.info("loading model(s) from {}".format(cfg.common_eval.path))
        self.models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=self.task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Optimize ensemble for generation
        for model in self.models:
            if model is None:
                continue
            if cfg.common.fp16:
                model.half()
            if self.use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)

        # Initialize generator
        self.generator = self.task.build_generator(self.models, cfg.generation)

        # Handle tokenization and BPE
        self.tokenizer = self.task.build_tokenizer(cfg.tokenizer)
        self.bpe = self.task.build_bpe(cfg.bpe)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(cfg.generation.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )

    def encode_fn(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
        if self.bpe is not None:
            x = self.bpe.encode(x)
        return x

    def decode_fn(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x
        
    def predict(self, inputs):

        start_id = 0
        results = []
        for batch in make_batches(inputs, self.cfg, self.task, self.max_positions, self.encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            constraints = batch.constraints
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }
            translations = self.task.inference_step(
                self.generator, self.models, sample, constraints=constraints
            )
            list_constraints = [[] for _ in range(bsz)]
            if self.cfg.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                constraints = list_constraints[i]
                results.append(
                (
                    start_id + id,
                    src_tokens_i,
                    hypos,
                    {
                        "constraints": constraints,
                    },
                )
            )
        
        start_id += len(inputs)

        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            src_str = ''
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.cfg.common_eval.post_process)

                # print("S-{}\t{}".format(id_, src_str))

            # Process top predictions
            for hypo in hypos[: min(len(hypos), self.cfg.generation.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
                )
                detok_hypo_str = self.decode_fn(hypo_str)

                # original hypothesis (after tokenization and BPE)
                # print("H-{}\t{}".format(id_, hypo_str))
                # # detokenized hypothesis
                # print("D-{}\t{}".format(id_, detok_hypo_str))
                # print(
                #     "P-{}\t{}".format(
                #         id_,
                #         " ".join(
                #             map(
                #                 lambda x: "{:.4f}".format(x),
                #                 # convert from base e to base 2
                #                 hypo["positional_scores"].div_(math.log(2)).tolist(),
                #             )
                #         ),
                #     )
                # )

                # detokenized hypothesis
                yield detok_hypo_str


def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()
