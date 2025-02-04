import random
import pickle
from argparse import ArgumentParser

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from capreolus.searcher import Searcher
from capreolus.sampler import PredSampler
from capreolus import parse_config_string
import tensorflow.compat.v1 as tf_v1

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


def get_capreolus_task(dataset, args):
    config = {
        **load_optimal_config(args),
        **get_shared_config(args, dataset=dataset)
    }
    config_string = " ".join([f"{k}={v}" for k, v in config.items()])
    if dataset == "gov2":
        config_string += " rank.searcher.index.name=gov2index reranker.extractor.index.name=gov2index "
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


def get_msmarco_generator(args):
    def extract_fn(data_record):
        features = {
              "query_ids": tf_v1.io.FixedLenSequenceFeature([], tf_v1.int64, allow_missing=True),
              "doc_ids": tf_v1.io.FixedLenSequenceFeature([], tf_v1.int64, allow_missing=True),
              "label": tf_v1.io.FixedLenFeature([], tf_v1.int64),
        }
        sample = tf_v1.io.parse_single_example(data_record, features)
        query_ids = tf_v1.cast(sample["query_ids"], tf_v1.int32)
        doc_ids = tf_v1.cast(sample["doc_ids"], tf_v1.int32)
        label_ids = tf.cast(sample["label"], tf_v1.int32)
        input_ids = tf_v1.concat((query_ids, doc_ids), 0)
        query_segment_id = tf_v1.zeros_like(query_ids)
        doc_segment_id = tf_v1.ones_like(doc_ids)
        segment_ids = tf_v1.concat((query_segment_id, doc_segment_id), 0)
        input_mask = tf_v1.ones_like(input_ids)
        features = {
          "input_ids": input_ids,
          "segment_ids": segment_ids,
          "input_mask": input_mask,
          "label_ids": label_ids,
        }
        return features

    # pdb.set_trace()
    path = "data/dataset_dev.tf"
    dataset = tf_v1.data.TFRecordDataset([path])
    dataset = dataset.map(extract_fn, num_parallel_calls=4).prefetch(1000)
    iter = tf_v1.data.make_one_shot_iterator(dataset)

    def _expand(x):
        return tf_v1.expand_dims(tf_v1.expand_dims(x, axis=0), axis=0)

    bar = tqdm(total=args.sampling_size * 30)  # hardcoded passage size
    for i, data in enumerate(iter):
        if i == (args.sampling_size * 30):
            break

        yield {
            "input": _expand(data["input_ids"]),
            "mask": _expand(data["input_mask"]),
            "seg": _expand(data["segment_ids"]),
        }
        bar.update()


def get_data_generator(args, task, batch_size=1):
    fold = "s1"
    size = args.sampling_size
    if size % batch_size:
        logger.warning(f"Batch size {batch_size} cannot be devided by total size {size}")

    rankTask = task.rank
    benchmark = task.benchmark
    extractor = task.reranker.extractor
    extractor.pad = 0
    extractor.pad_tok = extractor.tokenizer.bert_tokenizer.pad_token
    extractor.index.create_index()    

    qid2topic = benchmark.topics[benchmark.query_type]
    best_search_run = Searcher.load_trec_run(rankTask.evaluate()["path"][fold])

    n_generated_data = 0
    batch = defaultdict(list)
    bar = tqdm(total=size)

    for qid, docid2scores in best_search_run.items():
        topic = extractor.tokenizer.tokenize(qid2topic[qid])
        for docid in docid2scores:
            if batch and len(list(batch.values())[0]) == batch_size:
                yield {k: np.array(v) for k, v in batch.items()}
                batch = defaultdict(list)

            if random.random() < 0.5:
                continue
            if n_generated_data >= size:
                break

            doc = extractor.index.get_doc(docid).split()
            passages = extractor.get_passages_for_doc(doc)
            assert len(passages) == extractor.config["numpassages"]
            psg_inps, psg_masks, psg_segs = [], [], []
            for i, psg in enumerate(passages):
                inputs, segs, masks = extractor.tok2bertinput(topic, psg)
                psg_inps.append(inputs)
                psg_masks.append(masks)
                psg_segs.append(segs)

            batch["input"].append(psg_inps)
            batch["mask"].append(psg_masks)
            batch["seg"].append(psg_segs)

            n_generated_data += 1
            bar.update()


def get_bert_activation(inputs, reranker):
    n_step = inputs["input"].shape[-1]
    print(">>>>>> nstep", n_step)
    inp, mask, seg = [
            tf_v1.reshape(x, [-1, n_step]) 
            for x in [inputs["input"], inputs["mask"], inputs["seg"]]]
    bert_main_layer = reranker.model.bert.bert
    pooled_output = bert_main_layer(
        inputs=inp,
        attention_mask=mask,
        token_type_ids=seg,
    )  # all_outputs, cls token
    return pooled_output[1]  # shape: (batchsize * n_psg, nhidden)


def get_tNE_feature(dataset, args):
    path = f"{dataset}_slurm.pkl"
    try:
        return pickle.load(open(path, "rb"))["transformed"]
    except:
        logger.warning(f"Fail to load cached features, preparing...")

    if dataset == "msmarco":
        task = get_capreolus_task(dataset="rob04", args=args)
        data_generator = get_msmarco_generator(args)
    else:
        task = get_capreolus_task(dataset=dataset, args=args)
        kwargs = {"args": args, "task": task, "batch_size": args.batch_size}
        data_generator = get_data_generator(**kwargs)
    task.reranker.trainer.load_best_model(task.reranker, args.init_path, do_not_hash=True)
    with Timer(desc="Preparing Bert output"):
        X = np.array([
            get_bert_activation(inputs, task.reranker) for inputs in data_generator])
        n_batch, batch_size, n_hidden = X.shape
        X = X.reshape([n_batch * batch_size, n_hidden])

    with Timer(desc="Training t-SNE"):
        tsne = TSNE(random_state=0)
        Y = tsne.fit_transform(X)

    pickle.dump({"original": X, "transformed": Y}, open(path, "wb"))
    return Y


def main():
    args = get_args()
    datasets = args.dataset.split("+")
    for dataset in datasets:
        Y = get_tNE_feature(dataset, args)
        plt.scatter(Y[:, 0], Y[:, 1], label=dataset)
    # plt.title(f"{args- {args.sampling_size}")
    # plt.show()
    plt.legend()
    plt.savefig("tsne.png")


if __name__ == "__main__":
    main()

