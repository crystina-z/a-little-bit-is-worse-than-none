## A Little Bit Is Worse Than None: Ranking with Limited Training Data

This paper contains the code to reproduce the results in the SustaiNLP2020 paper: *A Little Bit Is Worse Than None: Ranking with Limited Training Data*.
Note that since we highly rely on the *ad-hoc retrieval framework [capreolus](https://capreolus.ai/)*, 
the modules in this repo are mostly the extension of the framework (under `./capreolus_extensions`) and do not contain the data processing and training logic.
Please find the details in the [framework github](https://github.com/capreolus-ir/capreolus) if you are interested in these.

The hyperparameters are listed under `./optimal_configs/maxp.txt`, with the format `config_key=config_value` each line.
Feel free to try other settings using this format. Note that lines starting with `#` is considered as comments and will be ignored by the program.
For the config key format and acceptable values, please find more details [here](https://capreolus.ai/en/latest/quick.html#command-line-interface). 

### Reproduce
The code is written in tensorflow-2.3 and supports TPU v2 and v3 (by Capreolus). 
This section provides the code to replicate all the experiments listed in the paper, 
which can be also found under `./scripts`

#### Train and Evaluate 
The following script is used to **train** and **evaluate** the experiments with sampled dataset. 
```
fold=s1       # Robust04 has fold from s1 to s5, GOV2 has fold from s1 to s3 
rate=1.0      # supports (0.0, 1.0], where 1.0 (default) means no sampling will be done 
dataset=rob04 # supports rob04 or gov2, 
              # since gov2 is not public dataset, the collection and built index need to be locally available, more in "Misc" section below  
do_train=True # if False, training will be skipped. Acceptable if training results are already available 
do_eval=True  # if False, evaluation will be skipped. You can review it later using "--do_train=False --do_eval=True" 

python run.py \
    --task sampling \
    --dataset rob04 \
    --sampling_rate $rate \
    --fold $fold \
    --train $do_train \
    --eval $do_eval 
```
When all folds results are available, the prorgam will also show cross-validated results on the evaluation stage. 

#### Zero-shot Inference 
We uploaded the converted ckpt to [here](), which will be needed to receive the zero-shot results reported in the paper. 
```  
ckpt_path=/path/to/ckpt  # Note that for TPU users, the ckpt needs to be uploaded to gcs 
                         # Then this path would look like gs://path/to/ckpt  
fold=s1 
dataset=rob04 

python run.py --task inference --dataset rob04 --fold $fold 
```

#### TPU users 
If TPU is available, append the following arguments to the above scripts to run the experiments on TPU: 
```
--tpu your_tpu_name --tpuzone your_tpu_zone (e.g. us-central1-f) --gs_bucket gs://your_gs_bucket_path
``` 

#### WandB users
If you use wandb, the results can be easily synced to your project by simply adding `--project_name your_wandb_project_name` after `pip install wandb && wandb login`. 
You are expected to see all configs and the value of metric `mAP`, `P@20`, `nDCG@20` plotted. 


### Misc
#### Weight conversion

#### GOV2 Data 


