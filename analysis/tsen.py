import random
import pickle
from time import time
from argparse import ArgumentParser

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from capreolus.searcher import Searcher
from capreolus.sampler import PredSampler
from capreolus import parse_config_string

from utils import load_optimal_config, get_shared_config, Timer
from capreolus_extensions.sampledBenchmark import *
from capreolus_extensions.wandbRerankerTask import WandbRerankerTask
from capreolus_extensions.collections import * 
from capreolus_extensions.gov2_utils import * 
from capreolus_extensions.gov_index import * 
from capreolus_extensions.sampledBenchmark import * 
from capreolus_extensions.tensorflowlog import *

import pdb

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--init_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="rob04")
    parser.add_argument("--model", type=str, default="maxp")
    parser.add_argument("--sampling_size", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--tpu", type=str, default="use_default")
    parser.add_argument("--gs_storage", type=str, default="gs://kelvin_project_crystina_dsg_us_f/reproduce")
    parser.add_argument("--tpuzone", type=str, default="us-central1-f")
    return parser.parse_args()


def get_capreolus_task(args):
    if args.dataset == "msmarco":
        config = {}
        pass  # TODO
    else:  # sampled_rob04 and sampled_gov2
        config = {
            **load_optimal_config(args),
            **get_shared_config(args)
        }
    config_string = " ".join([f"{k}={v}" for k, v in config.items()])
    print(config_string)
    return WandbRerankerTask(parse_config_string(config_string))


def filter_runs(runs, fold_qids, threshold=1000):
    dev_run = defaultdict(dict)
    for qid, docs in runs.items():
        if qid in fold_qids:  # folds[fold]["predict"]["dev"]:
            for idx, (docid, score) in enumerate(docs.items()):
                if idx >= threshold:
                    assert len(dev_run[qid]) == threshold, \
                        f"Expect {threshold} on each qid, got {len(dev_run[qid])} for query {qid}"
                    break
                dev_run[qid][docid] = score
    return dev_run


def get_data_generator(args, task, batch_size=1):
    fold = "s1"
    ds, size = args.dataset, args.sampling_size
    if size % batch_size:
        logger.warning(f"Batch size {batch_size} cannot be devided by total size {size}")

    rankTask = task.rank
    benchmark = task.benchmark
    extractor = task.reranker.extractor
    reranker = task.reranker

    best_search_run = Searcher.load_trec_run(
        rankTask.evaluate()["path"][fold])
    docids = set(docid for querydocs in best_search_run.values() for docid in querydocs)
    reranker.extractor.preprocess(
	qids=best_search_run.keys(), docids=docids, topics=benchmark.topics[benchmark.query_type])

    dev_run = filter_runs(
        runs=best_search_run,
        threshold=100,
        fold_qids=benchmark.folds[fold]["predict"]["dev"],
    )
    dev_dataset = PredSampler()
    dev_dataset.prepare(
        qid_to_docids=dev_run,
        qrels=benchmark.qrels,
        extractor=extractor,
        relevance_level=benchmark.relevance_level,
    )
    n_generated_data = 0 
    batch = defaultdict(list)
    bar = tqdm(total=size)
    for data in dev_dataset: 
        if batch and len(list(batch.values())[0]) == batch_size:
            yield {k: np.array(v) for k, v in batch.items()}
            batch = defaultdict(list)

        if random.random() < 0.5:
            continue
        if n_generated_data >= size:
            break

        n_generated_data += 1
        bar.update()
        for k, v in data.items():
            if k in ["pos_bert_input", "pos_mask", "pos_seg"]:
                batch[k.split("_")[-1]].append(v)
    if batch:
        logger.warning(f"{len(batch)} are discarded since it does not fit into batch size {batch_size}")


def get_bert_activation(inputs, reranker):
    inp, mask, seg = [x.reshape(-1, 256) for x in [inputs["input"], inputs["mask"], inputs["seg"]]]
    bert_main_layer = reranker.model.bert.bert
    pooled_output = bert_main_layer(
        inputs=inp,
        attention_mask=mask,
        token_type_ids=seg,
    )  # all_outputs, cls token
    return pooled_output[1] # shape: (batchsize * n_psg, nhidden) 


def main():
    args = get_args()
    task = get_capreolus_task(args=args)
    trainer, reranker = task.reranker.trainer, task.reranker
    trainer.load_best_model(reranker, args.init_path, do_not_hash=True)

    with Timer(desc="Preparing Bert output"):
        X = np.array([
            get_bert_activation(inputs, task.reranker) for inputs in get_data_generator(args=args, task=task, batch_size=args.batch_size)])
        print("before reshape: ", X.shape) # expect: n_batch, batchsize * n_psg, hidden 
        n_batch, batch_size, n_hidden = X.shape
        X = X.reshape([n_batch * batch_size, n_hidden])
        print("after reshape: ", X.shape) # expect: n_batch, batchsize * n_psg, hidden 

    with Timer(desc="Training t-SNE"):
        tsne = TSNE(random_state=0)
        Y = tsne.fit_transform(X)

    pickle.dump({"original": X, "transformed": Y}, open("tmp.pkl", "wb"))
    plt.scatter(Y[:0], Y[:1])
    plt.title(f"{args.dataset} - {args.sampling_size}")
    # plt.show()
    plt.savefig("tsne.png")


if __name__ == "__main__":
    main()

