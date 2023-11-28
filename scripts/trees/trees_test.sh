#!/bin/bash
export OMP_NUM_THREADS=3

CURR_QUERY=50
CURR_DBSCAN=2.0

# TODO: don't forget to change experiment name to the one used in train script
# CHANGE VOXEL SIZE
EXP_NAME="trees_v1_large_vx005"

# TEST
python main_instance_segmentation.py \
general.experiment_name="${EXP_NAME}_test_TOM2" \
general.project_name="trees_test" \
data/datasets=trees \
general.num_targets=2 \
data.num_labels=2 \
data.voxel_size=0.05 \
data.num_workers=10 \
data.batch_size=1 \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
model.config.backbone._target_=models.Res16UNet18B \
general.train_mode=false \
general.save_visualizations=true \
general.train_mode=false \
data.test_mode=test \
general.checkpoint="saved/${EXP_NAME}/last-epoch.ckpt" \
general.export=true \
# general.use_dbscan=true \
# general.dbscan_eps=${CURR_DBSCAN} \
