#!/bin/bash
export OMP_NUM_THREADS=3

CURR_QUERY=50

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="trees_v0_train_val2" \
general.project_name="trees" \
data/datasets=trees \
general.num_targets=2 \
data.num_labels=2 \
data.voxel_size=0.10 \
data.num_workers=10 \
data.batch_size=2 \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
model.config.backbone._target_=models.Res16UNet18B \
trainer.check_val_every_n_epoch=10

# TODO: num_labels and num_targets the same? is different by 1 for s3dis, may be difference in semantic vs instance


# TEST
# python main_instance_segmentation.py \
# general.experiment_name="trees_v0_test_query_${CURR_QUERY}" \
# general.project_name="trees_eval" \
# data/datasets=trees \
# general.num_targets=2 \
# data.num_labels=2 \
# data.voxel_size=0.05 \
# data.num_workers=10 \
# general.reps_per_epoch=100 \
# model.num_queries=${CURR_QUERY} \
# general.on_crops=true \
# model.config.backbone._target_=models.Res16UNet18B \
# general.train_mode=false \
# general.checkpoint="saved/trees_v0_train/last-epoch.ckpt" \\
