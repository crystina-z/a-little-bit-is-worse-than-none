from argparse import ArgumentParser

from tqdm import tqdm
import tensorflow as tf
from nirtools.tensorflow import ckpt
from tensorflow.train import list_variables, load_variable, load_checkpoint


def check_dl4_weights():
    ori_model_path = "weights/model.ckpt-100000"
    ckpt.inspect(ori_model_path)


def check_hf_weights():
    from transformers import TFAutoModel
    weights_names = ckpt.inspect_huggingface_model(TFAutoModel, "bert-base-uncased")
    for weight, shape in weights_names:
        print(weight, shape)


def rename_dl42cap(name):
    name = f"model/bert/{name}".replace("layer_", "layer/")
    if "/word_embeddings" in name:
        return name.replace("word_embeddings", "weight")
    elif "/position_embeddings" in name:
        return name.replace("position_embeddings", "position_embeddings/embeddings")
    elif "/token_type_embeddings" in name:
        return name.replace("token_type_embeddings", "token_type_embeddings/embeddings")
    # the sequence of the following two conditions is important!
    elif "attention/output/" in name:
        return name.replace("attention/output/", "attention/dense_output/")
    elif "/self/" in name:
        return name.replace("/self/", "/self_attention/")
    elif "/output/" in name:
        return name.replace("/output/", "/bert_output/")
    elif "output_weights" in name:
        return name.replace("output_weights", "classifier/kernel")
    elif "output_bias" in name:
        return name.replace("output_bias", "classifier/bias")
    return name


def reconvert_trial(inp_fn, outp_fn):
    # reference: https://www.tensorflow.org/guide/checkpoint#saving_object-based_checkpoints_with_estimator
    var_names, ckpt_reader = list_variables(inp_fn), load_checkpoint(inp_fn)
    renamed_tvars = {}

    for name, shape in tqdm(var_names, "converting"):
        if "adam_m" in name or "adam_v" in name or "global_step" in name:
            continue

        var = ckpt_reader.get_tensor(name)
        if name == "output_weights":
            var = var.transpose()

        name = rename_dl42cap(name)
        renamed_tvars[name] = tf.Variable(var, name=name)

    def _merge_a_into_b(a, b):
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                if not isinstance(b[k], dict):
                    raise TypeError
                b[k] = _merge_a_into_b(v, b[k])
            else:
                b[k] = v
        return b

    global_dict = {}
    ckpt = tf.train.Checkpoint()
    for name in renamed_tvars:
        cur_dict = {}
        for subname in name.split("/")[::-1]:
            cur_dict = {subname: cur_dict if cur_dict else renamed_tvars[name]}
        global_dict = _merge_a_into_b(cur_dict, global_dict)

    assert list(global_dict) == ["model"]
    ckpt.model = global_dict["model"]
    ckpt.save(outp_fn)


if __name__ == "__main__":
    ''' To convert ckpt downloaded from https://github.com/nyu-dl/dl4marco-bert to Capreolus: '''
    parser = ArgumentParser()
    parser.add_argument("--inp_fn", "-i", type=str, description="tf ckpt file to convert: /path/to/weight.ckpt")
    parser.add_argument("--outp_fn", "-o", type=str, description="expected path to store converted weight")
    args = parser.parse_args()

    reconvert_trial(inp_fn=args.inp_fn, outp_fn=args.outp_f)
