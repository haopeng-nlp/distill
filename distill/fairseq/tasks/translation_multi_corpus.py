# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq import utils
from fairseq.data import LanguagePairDataset
from collections import OrderedDict
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset

from . import register_task
from .translation import TranslationTask, load_langpair_dataset


@register_task("translation_multi_corpus")
class TranslationMultiCorpusTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationTask.add_args(parser)
        parser.add_argument('--noise-levels', type=str)
        parser.add_argument('--path-to-data', type=str)

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.args = args
        self.noise_levels = args.noise_levels.split(":")
        self.path_to_data = args.path_to_data

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        dataset_map = OrderedDict()
        for noise in self.noise_levels:
            data_path = self.path_to_data.replace('noise', noise)
            dataset_map[noise] = load_langpair_dataset(
                data_path,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                left_pad_source=self.cfg.left_pad_source,
                left_pad_target=self.cfg.left_pad_target,
                max_source_positions=self.cfg.max_source_positions,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=self.cfg.load_alignments,
                truncate_source=self.cfg.truncate_source,
                num_buckets=self.cfg.num_batch_buckets,
                shuffle=(split != "test"),
                pad_to_multiple=self.cfg.required_seq_len_multiple,
            )

        self.datasets[split] = MultiCorpusSampledDataset(dataset_map)
