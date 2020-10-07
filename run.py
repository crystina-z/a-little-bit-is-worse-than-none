from multiprocessing import Process

from capreolus import parse_config_string
from capreolus.utils.loginit import get_logger

from wandbRerankerTask import WandbRerankerTask
from cedr_task import CEDRTASK
from tensorflowlog import TensorflowLogTrainer  # it's not direclty used, yet it's necessary for the module to be registered

from args import get_args, _parse_args
from utils import get_wandb

logger = get_logger(__name__)
wandb = get_wandb()

FOLD2TPU = {f"s{i}": f"node-crys{i}" for i in range(1, 6)}


def init_wandb(args, config, cv=False, project_name="default_project"):
    exclusion_list = [
        "benchmark.collection.name", "reranker.trainer.usecache", "reranker.trainer.tpuname",
        "reranker.trainer.tpuzone", "reranker.trainer.storage", "reranker.trainer.loss"]
    config = {
        "trainer.loss": config.get("reranker.trainer.loss", "not found"),
        **{"customize_init": True if args.init_path != "none" else False, "init_path": args.init_path},
        **{k if k.endswith("name") else k.split(".")[-1]: v for k, v in config.items() if k not in exclusion_list}
    }
    task, model = args.task, args.model

    fold = "cross-validate" if cv else args.fold
    run = wandb.init(
        project=project_name,
        name=f"{model}-{task}-{fold}",
        group=f"{model}-{task}",
        config=config,
        sync_tensorboard=True,
        reinit=True,
    )
    return run


def _get_shared_config(args):
    return {
        "benchmark.collection.name": "robust04",
        "reranker.trainer.name": "tensorflowlog",
        "reranker.trainer.tpuzone": args.tpuzone,
        "reranker.trainer.storage": args.gs_storage,
        "reranker.trainer.usecache": args.use_cache,
    }


def _get_optimal_config():
    with open(f"optimal_configs/maxp.txt") as f:
        configs = {
            line.split("=")[0].strip(): line.split("=")[1].strip() for line in f
            if line.strip() and (not line.lstrip().startswith("#"))}  # use # as comment
    return configs


def get_configs(args):
    task_configs = _parse_args(args)
    shared_config = _get_shared_config(args)
    optimal_config = _get_optimal_config()  # load maxp default parameters

    if args.task in ["optimal", "inference"]:
        yield {
            **shared_config,
            **optimal_config,
        }
    else:
        for task_config in task_configs:
            yield {
                **shared_config,
                **optimal_config,
                **task_config,
            }


def run_single_fold(config_string, fold, args, config):
    parsed_string = parse_config_string(config_string)
    task = WandbRerankerTask(parsed_string)
    run = init_wandb(args, {"fold": fold, **config}, project_name=args.project_name)

    if args.task == "inference":
        init_path = args.init_path if args.init_path != "none" else None
        scores = task.predict_and_eval(init_path=init_path)
        print(f"test metrics on fold {fold}: ", scores["fold_test_metrics"])
        wandb.log(scores["fold_test_metrics"])
        wandb.join()

        if scores["cv_metrics"]:
            wandb, run = init_wandb(args, config, cv=True, project_name=args.project_name)
            print(f"cross validated score:", scores["cv_metrics"])
            print(f"interpolate score:", scores["interpolated_results"])
            with run:
                wandb.config.update({"fold": "cross-validate"})
                wandb.log(scores["cv_metrics"])
        return

    if args.train:
        init_path = args.init_path if args.init_path != "none" else ""
        print(f"TASK: {args.task}\tTRAINING ON FOLD {fold}")
        task.train(init_path=init_path)

    if args.eval:
        print(f"TASK: {args.task}\tEVALUATING ON FOLD {fold}")
        scores = task.evaluate()
        print(f"test metrics on fold {fold}: ", scores["fold_test_metrics"])
        wandb.log(scores["fold_test_metrics"])
        wandb.join()

        if scores["cv_metrics"]:
            run = init_wandb(args, config, cv=True, project_name=args.project_name)
            print(f"cross validated score:", scores["cv_metrics"])
            with run:
                wandb.config.update({"fold": "cross-validate"})
                wandb.log(scores["cv_metrics"])


def main():
    args = get_args()
    configs = get_configs(args)
    for config in configs:
        common_config_string = " ".join([f"{k}={v}" for k, v in config.items()])
        processes = []
        for i in range(1, 6):
            fold = f"s{i}"
            if args.fold != "all" and fold not in args.fold.split("-"):
                print(f"Skip fold {fold}, as it's not in the {args.fold.split()}")
                continue

            tpu = args.tpu if args.tpu != "none" else FOLD2TPU[fold]
            config_string = common_config_string + f" reranker.trainer.tpuname={tpu} fold={fold}"
            p = Process(target=run_single_fold, args=(config_string, fold, args, config))  # so that 5 folds can run tgt
            processes.append(p)
            p.start()

        print(f"Totally {len(processes)} processes are created")
        for p in processes:
            p.join()
            p.close()
        print("FINISHED")


if __name__ == "__main__":
    main()
