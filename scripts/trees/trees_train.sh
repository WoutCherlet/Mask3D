#!/bin/bash
export OMP_NUM_THREADS=3

CURR_QUERY=50

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="trees_v1_large_vx005" \
general.project_name="trees" \
data/datasets=trees \
general.num_targets=2 \
data.num_labels=2 \
data.voxel_size=0.05 \
data.num_workers=10 \
data.batch_size=2 \
model.num_queries=${CURR_QUERY} \
model.config.backbone._target_=models.Res16UNet18B \
trainer.check_val_every_n_epoch=20


# TODO: try dbscan postprocessing step
