from multiprocessing import Process

from capreolus import parse_config_string
from capreolus.utils.loginit import get_logger

from wandbRerankerTask import WandbRerankerTask
from tensorflowlog import TensorflowLogTrainer  # it's not direclty used, yet it's necessary for the module to be registered
from capreolus_extensions.collections import GOV2Collection
from capreolus_extensions.sampledBenchmark import *

from utils import *  # get_wandb, load_optimal_config, _get_shared_config
from args import get_args, get_task_config


wandb = get_wandb()
logger = get_logger(__name__)
FOLD2TPU = {f"s{i}": f"node-crys{i}" for i in range(1, 6)}


def init_wandb(args, config, cv=False, project_name="default_project"):
    exclusion_list = [
        "reranker.trainer.usecache", "reranker.trainer.tpuname", "reranker.trainer.tpuzone", "reranker.trainer.storage"
    ]
    config = {
        "trainer.loss": config.get("reranker.trainer.loss", "not found"),
        "init_path": args.init_path,
        "customize_init": True if args.init_path != "none" else False,
        **{
            k if k.endswith("name") else k.split(".")[-1]: v
            for k, v in config.items() if k not in exclusion_list
        }
    }
    task, model, rate, benchmark = args.task, args.model, args.rate, config["benchmark.name"]

    fold = "cross-validate" if cv else args.fold
    run = wandb.init(
        project=project_name,
        name=f"{model}-{task}-{benchmark}-{fold}",
        group=f"{model}-{task}-{benchmark}-{rate}",
        config=config,
        sync_tensorboard=True,
        reinit=True,
    )
    return run


def get_configs(args):
    task_configs = get_task_config(args)
    shared_config = get_shared_config(args)
    optimal_config = load_optimal_config()  # load maxp default parameters

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
        scores = task.predict_and_eval(init_path=args.init_path)
        logger.info(f"test metrics on fold {fold}")
        logger.info(scores["fold_test_metrics"])
    else:
        if args.train:
            logger.info(f"TASK: {args.task}\tTRAINING ON FOLD {fold}")
            task.train(init_path=args.init_path)

        if args.eval:
            logger.info(f"TASK: {args.task}\tEVALUATING ON FOLD {fold}")
            scores = task.evaluate()

    logger.info(f"test metrics on fold {fold}: ")
    logger.info(scores["fold_test_metrics"])
    wandb.log(scores["fold_test_metrics"])
    wandb.join()

    if scores["cv_metrics"]:
        run = init_wandb(args, {**config, "fold": "cross-validate"}, cv=True, project_name=args.project_name)
        print(f"cross validated score:", scores["cv_metrics"])
        with run:
            wandb.log(scores["cv_metrics"])


def main():
    args = get_args()
    configs = get_configs(args)
    nfolds = 5 if args.dataset == "rob04" else 3  # if args.dataset == "gov2"
    for config in configs:
        common_config_string = " ".join([f"{k}={v}" for k, v in config.items()])
        processes = []
        for i in range(1, 1+nfolds):
            fold = f"s{i}"
            if args.fold != "all" and fold not in args.fold.split("-"):
                logger.info(f"Skip fold {fold}, as it's not in the {args.fold.split()}")
                continue

            tpu = args.tpu if args.tpu != "use_default" else FOLD2TPU[fold]
            config_string = common_config_string + f" reranker.trainer.tpuname={tpu} fold={fold}"
            p = Process(target=run_single_fold, args=(config_string, fold, args, config))  # so that 5 folds can run tgt
            processes.append(p)
            p.start()

        logger.info(f"Totally {len(processes)} processes are created")
        for p in processes:
            p.join()
            p.close()
        logger.info("FINISHED")


if __name__ == "__main__":
    main()
