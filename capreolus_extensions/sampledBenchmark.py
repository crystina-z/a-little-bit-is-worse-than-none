import os
import sys
import json
import math
import copy
import random

from pathlib import Path
from collections import defaultdict
import numpy as np

from nirtools.ir import load_topic_trec
from capreolus import ConfigOption, Dependency, constants, parse_config_string
from capreolus.benchmark import Benchmark
from capreolus.benchmark.robust04 import Robust04Yang19
from capreolus.utils.loginit import get_logger
from capreolus.utils.common import download_file
from capreolus.utils.trec import load_qrels, topic_to_trectxt

PACKAGE_PATH = constants["PACKAGE_PATH"]
logger = get_logger(__name__)


@Benchmark.register
class GOV2Benchmark(Benchmark):
    module_name = "gov2benchmark"
    query_type = "title"

    file_fn = (Path(os.path.dirname(__file__)) / "data" / module_name).absolute()
    qrel_file = file_fn / "qrels.gov2.txt"
    topic_file = file_fn / "topics.gov2.701-750.751-800.801-850.txt"
    fold_file = file_fn / "gov2.json"
    dependencies = [
        Dependency(key="collection", module="collection", name="gov2collection")]

    @staticmethod
    def combine_files(fns_to_combine, out_fn):
        out_fn.parent.mkdir(exist_ok=True, parents=True)
        with open(out_fn, "w", encoding="utf-8") as fout:
            for fn in fns_to_combine:
                print(f"adding {fn} to {out_fn}")
                with open(fn) as fin:
                    for line in fin:
                        fout.write(line)

    def build(self):
        self.download_if_missing()

    def split_5_folds(self, qids):
        all_qids, fold_list = [], []
        for i in range(1, 6):
            fold_fn = Path(os.path.dirname(__file__)).absolute() / "data" / "gov2_folds" / f"fold{i}"
            qids = [line.strip() for line in open(fold_fn)]
            fold_list.append(qids)
            all_qids.extend(qids)
        assert set(all_qids) == set(qids)

        fold_dict = {}
        for i in range(5):
            train = fold_list[i] + fold_list[(i+1) % 5] + fold_list[(i+2) % 5]
            dev, test = fold_list[(i+3) % 5], fold_list[(i+4) % 5]
            fold_dict[f"s{i+1}"] = {"train_qids": train, "predict": {"dev": dev, "test": test}}
        json.dump(fold_dict, open(self.fold_file, "w"))

    def split_3_folds(self, qids):
        overlap_qids = self.get_overlap_qids()
        assert len(overlap_qids) == 56
        f1 = random.choices(overlap_qids, k=50)
        non_f1 = [q for q in qids if q not in f1]
        f2 = random.choices(non_f1, k=50)
        f3 = [q for q in non_f1 if q not in f2]

        fold_list = [f1, f2, f3]
        fold_dict = {}
        for i in range(3):
            train, dev, test = fold_list[i], fold_list[(i+1) % 3], fold_list[(i+2) % 3]
            fold_dict[f"s{i+1}"] = {"train_qids": train, "predict": {"dev": dev, "test": test}}
        json.dump(fold_dict, open(self.fold_file, "w"))

    def download_if_missing(self):
        tmp_dir = self.get_cache_path() / "tmp"
        tmp_dir.mkdir(exist_ok=True, parents=True)
        if self.topic_file.exists() and self.qrel_file.exists() and self.fold_file.exists():
            return

        topic_urls = [
            "http://trec.nist.gov/data/terabyte/04/04topics.701-750.txt",
            "http://trec.nist.gov/data/terabyte/05/05.topics.751-800.txt",
            "http://trec.nist.gov/data/terabyte/06/06.topics.801-850.txt",
        ]
        qrel_urls = [
            "http://trec.nist.gov/data/terabyte/04/04.qrels.12-Nov-04",
            "http://trec.nist.gov/data/terabyte/05/05.adhoc_qrels",
            "http://trec.nist.gov/data/terabyte/06/qrels.tb06.top50",
        ]

        topic_tmp_fns, qrels_tmp_fns = [], []
        for url in topic_urls:
            fn = tmp_dir / url.split("/")[-1]
            if not fn.exists():
                download_file(url, fn)
            topic_tmp_fns.append(fn)

        for url in qrel_urls:
            fn = tmp_dir / url.split("/")[-1]
            if not fn.exists():
                download_file(url, fn)
            qrels_tmp_fns.append(fn)

        self.combine_files(fns_to_combine=topic_tmp_fns, out_fn=self.topic_file)
        self.combine_files(fns_to_combine=qrels_tmp_fns, out_fn=self.qrel_file)
        qids = [qid for qid, topic in load_topic_trec(self.topic_file)]
        self.split_3_folds(qids)


class SampleMixin:
    @staticmethod
    def get_judgement_stat(qrels, type="sum"):
        if type == "sum":
            return sum([len(docs) for q, docs in qrels.items()])
        elif type == "avg":
            return sum([len(docs) for q, docs in qrels.items()]) / len(qrels)

    @staticmethod
    def get_pos_neg_ratio(qrels):
        n_pos = sum([len([d for d, label in docs.items() if label > 0]) for q, docs in qrels.items()])
        n_neg = sum([len([d for d, label in docs.items() if label == 0]) for q, docs in qrels.items()])
        return n_pos / n_neg

    @staticmethod
    def contain_empty_label(docs, docs2label):
        pos_doc = [docid for docid in docs if docs2label[docid] > 0]
        neg_doc = [docid for docid in docs if docs2label[docid] == 0]
        if pos_doc and neg_doc:
            return False
        return True

    def build(self):
        mode, rate = self.config["mode"], self.config["rate"]
        self.unsampled_qrel_file = self.qrel_file
        self.unsampled_fold_file = self.fold_file
        self.qrel_file = self.file_fn / (f"{mode}.%.2f.qrels.txt"%rate)
        self.fold_file = self.file_fn / (f"{mode}.%.2f.fold.json"%rate)
        self.download_if_missing()

    @property
    def unsampled_qrels(self):
        if not hasattr(self, "_unsampled_qrels"):
            self._unsampled_qrels = load_qrels(self.unsampled_qrel_file)
        return self._unsampled_qrels

    @property
    def unsampled_folds(self):
        if not hasattr(self, "_unsampled_folds"):
            self._unsampled_folds = json.load(open(self.unsampled_fold_file, "rt"))
        return self._unsampled_folds

    def prune_redundant_judgement(self, sampled_qrels, n_expected_judgement, mode):
        """ keep same number of judgements in qrels while try to keep the pos & neg label ratio same as before """
        # expected_pos_neg_rate = self.get_pos_neg_ratio(self.qrels)  # TODO: add pos/neg ratio

        if mode not in ["deep", "shallow"]:
            raise ValueError(f"Unexpected mode: {mode}, should be one of 'deep', 'shallow']")

        n_judgement = self.get_judgement_stat(sampled_qrels, "sum")
        n_remove = n_judgement - n_expected_judgement
        assert n_remove >= 0, f"expect {n_expected_judgement} judgements but the input contains only {n_judgement} judegment. current mode: {mode}, rate: {self.config['rate']}"
        logger.info(f"{n_remove} more judgements to remove")
        # if n_judgement < n_expected_judgement:
        #     logger.warning(f"The qrels are already oversampled")

        pruned_qrels = copy.deepcopy(sampled_qrels)
        qid_n_judgement = [(qid, len(docs)) for qid, docs in pruned_qrels.items()]
        qid_n_judgement = sorted(qid_n_judgement, key=lambda q_n: q_n[1])  # sort qid according to num_judgement

        def gen_qid(revert=False):
            while True:
                for qid, n in qid_n_judgement[::-1] if revert else qid_n_judgement:
                    yield qid, n

        if mode == "deep":  # higher priority to trim query
            smallest_qid, n1 = next(gen_qid(revert=False))
            random_qid, n2 = random.sample(qid_n_judgement, 1)[0]
            if n2 < n_remove:
                n_remove -= n2
                del pruned_qrels[random_qid]
            else:
                assert n1 <= n_remove
                n_remove -= n1
                del pruned_qrels[smallest_qid]

        for qid, n in gen_qid(revert=True):  # document from queries with more judgement will be dropped
            if n_remove == 0:
                break

            if not pruned_qrels[qid]:  # has already been emptied
                if mode == "shallow":
                    raise ValueError(f"Lost {qid} during pruning")
                else:
                    logger.warning(f"{qid} is removed during qrel pruning at {mode} mode. "
                                   f"This is expected if {qid} is the only query removed at this stage")
                    continue

            doc_to_drop = random.sample(pruned_qrels[qid].keys(), 1)[0]
            if self.contain_empty_label(  # don't remove if we run out of either pos or neg judged doc
                    docs=[d for d in pruned_qrels[qid] if d != doc_to_drop],
                    docs2label=pruned_qrels[qid]):
                logger.info(f"Potential empty pos/neg is detected, skip dropping qid {qid} "
                            f"(#judgement left: {len(pruned_qrels[qid])})")
                continue

            del pruned_qrels[qid][doc_to_drop]
            n_remove -= 1

        n_after_judgement = self.get_judgement_stat(pruned_qrels, "sum")
        assert n_after_judgement == n_expected_judgement, f"expect {n_expected_judgement} yet got {n_after_judgement}"
        return pruned_qrels

    def download_if_missing(self):
        """ prepare sampled files from Robust04 qrels"""
        if self.qrel_file.exists() and self.fold_file.exists():
            return

        self.qrel_file.parent.mkdir(parents=True, exist_ok=True)
        self.fold_file.parent.mkdir(parents=True, exist_ok=True)

        mode, rate, qrels = self.config["mode"], self.config["rate"], self.unsampled_qrels

        if rate <= 0 or rate > 1:
            raise ValueError(f"Sampling rate out of range: expect it from (0, 1], yet got {rate}")

        if rate == 1:
            self.qrel_file = self.unsampled_qrel_file
            self.fold_file = self.unsampled_fold_file
            logger.warning(f"No sampling is performed since sampling rate is set to {rate}")
            return

        sampled_qrels = defaultdict(dict)
        if mode == "deep":  # sample `rate` percent queries from qrels.
            for fold, split in self.unsampled_folds.items():
                # sampling within each fold's test qids s.t. the train qids in each would be kept uniformly
                test_qids = split["predict"]["test"]
                for qid in copy.deepcopy(test_qids):
                    if qid not in qrels:
                        logger.warning(f"{qid} unfound in qrels, removed")
                        test_qids.remove(qid)

                n_expected_qids = math.ceil(len(test_qids) * rate) + 1  # add one to avoid over-trim
                sampled_qids = random.sample(test_qids, n_expected_qids)
                sampled_qrels.update({qid: qrels[qid] for qid in sampled_qids})

        elif mode == "shallow":  # sample `rate` percent of documents from qrels
            for qid, docs in self.unsampled_qrels.items():
                n_expected_docids = math.ceil(len(docs) * rate)
                assert n_expected_docids > 0, f"{qid} has zero document after sampling with rate {rate}"
                sampled_docids = random.sample(docs.keys(), n_expected_docids)

                if self.contain_empty_label(sampled_docids, docs):  # skip if run out of either pos or neg judged doc
                    logger.info(f"Potential empty pos/neg is detected, skip dropping qid {qid}"
                                f"(#judgement left: {len(docs)}")
                    sampled_qrels[qid] = docs
                else:
                    sampled_qrels[qid] = {docid: docs[docid] for docid in sampled_docids}

        else:
            raise ValueError(f"Unexpected sampling mode: {mode}")

        # prune the sampled qrels s.t. shallow and deep sampling would keep same number of judgements
        n_judgement = self.get_judgement_stat(qrels, "sum")
        n_expected_judgement = math.floor(rate * n_judgement)
        sampled_qrels = self.prune_redundant_judgement(sampled_qrels, n_expected_judgement, mode)
        n_qids, n_avg_doc, n_sampled_judgement = \
            len(sampled_qrels), self.get_judgement_stat(sampled_qrels, "avg"), self.get_judgement_stat(sampled_qrels, "sum")
        logger.info(f"After {mode} sampling with rate={rate}: \n"
                    f"{n_qids} queries with average {n_avg_doc} judged documents are kept;\n"
                    f"Total judgement number: {n_sampled_judgement}/{n_judgement}; \n"
                    f"pos/neg ratio: {self.get_pos_neg_ratio(sampled_qrels)}\n")

        # update train and dev qids in each fold accordingly
        new_folds = {}
        for fold_name, split in self.unsampled_folds.items():
            train_qids, dev_qids = split["train_qids"], split["predict"]["dev"]
            new_folds[fold_name] = {
                "train_qids": [qid for qid in train_qids if qid in sampled_qrels],
                "predict": {
                    "dev": [qid for qid in dev_qids if qid in sampled_qrels],
                    "test": split["predict"]["test"]
                }
            }

        # save to disk
        with open(self.qrel_file, "w") as f:
            for qid in sorted(sampled_qrels.keys(), key=lambda x: int(x)):
                for docid, label in sampled_qrels[qid].items():
                    f.write(f"{qid} Q0 {docid} {label}\n")
        json.dump(new_folds, open(self.fold_file, "w"))


@Benchmark.register
class SampledRobust04(Robust04Yang19, SampleMixin):
    module_name = "sampled_rob04"
    file_fn = PACKAGE_PATH / "data" / module_name

    config_spec = [
        ConfigOption("mode", "shallow", "which sampling mode to use: shallow or deep"),
        ConfigOption("rate", 1.0, "sampling rate: fraction number between 0 to 1"),
    ]


@Benchmark.register
class SampledGOV2(GOV2Benchmark, SampleMixin):
    module_name = "sampled_gov2"
    file_fn = (Path(os.path.dirname(__file__)) / "data" / module_name).absolute()

    config_spec = [
        ConfigOption("mode", "shallow", "which sampling mode to use: shallow or deep"),
        ConfigOption("rate", 1.0, "sampling rate: fraction number between 0 to 1"),
    ]


if __name__ == "__main__":
    if sys.argv[1] == "gov":
        benchmark = GOV2Benchmark()
        benchmark.download_if_missing()
    elif sys.argv[1] == "govsample":
        for rate in np.arange(0.1, 1.1, 0.2):
            rate = f"%.2f" % rate
            print("rate", rate)
            benchmark = SampledGOV2(parse_config_string(f"rate={rate}"))
            benchmark.download_if_missing()
