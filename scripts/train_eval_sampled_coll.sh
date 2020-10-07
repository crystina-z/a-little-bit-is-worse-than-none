fold=s1       # Robust04 has fold from s1 to s5, GOV2 has fold from s1 to s3
rate=1.0      # supports (0.0, 1.0], where 1.0 (default) means no sampling will be done
dataset=rob04 # supports rob04 or gov2,
              # since gov2 is not public dataset, the collection and built index need to be locally available, more in "Misc" section below
do_train=True # if False, training will be skipped. Acceptable if training results are already available
do_eval=True  # if False, evaluation will be skipped. You can review it later using "--do_train=False --do_eval=True"

other_command=$@

python run.py \
    --task sampling \
    --dataset rob04 \
    --sampling_rate $rate \
    --fold $fold \
    --train $do_train \
    --eval $do_eval \
    $other_command

    # to use TPU, add:
    # --tpu your_tpu_name --tpuzone your_tpu_zone (e.g. us-central1-f) --gs_bucket gs://your_gs_bucket_path

    # to use WandB, add
    # --project_name your_wandb_project_name`
