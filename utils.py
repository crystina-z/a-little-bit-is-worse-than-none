from time import time
from contextlib import nullcontext


class Timer(object):
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        self.start = time()

    def __exit__(self, type, value, traceback):
        n_secs = time() - self.start
        time_desc = "%.3f sec" % n_secs if n_secs < 300 else "%.3f min" % (n_secs / 60)
        print(f"{self.desc}: {time_desc}")


def get_shared_config(args):
    return {
        "reranker.trainer.name": "tensorflowlog",
        "benchmark.name": "sampled_rob04" if args.dataset == "rob04" else "sampled_gov2",
        "reranker.trainer.tpuzone": args.tpuzone,
        "reranker.trainer.storage": args.gs_storage,
        "reranker.trainer.usecache": args.use_cache,
    }


def load_optimal_config(args):
    fn = f"{args.model}.txt"
    with open(f"optimal_configs/{fn}") as f:
        configs = {
            line.split("=")[0].strip(): line.split("=")[1].strip() for line in f
            if line.strip() and (not line.lstrip().startswith("#"))}  # use # as comment
    return configs


def get_wandb():
    try:
        import wandb
    except ModuleNotFoundError:
        print("Fail to import wandb, config and scores will only be printed locally.")
        class wandb:
            @staticmethod
            def init(*args, **kwargs):
                return nullcontext  # as an replacement for "run" object

            @staticmethod
            def log(*args, **kwargs):
                print(args, kwargs)

            @staticmethod
            def join():
                return

            @staticmethod
            def save(file_path):
                return

    return wandb
