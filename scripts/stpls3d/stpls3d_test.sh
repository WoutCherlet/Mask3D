#!/bin/bash
export OMP_NUM_THREADS=3

CURR_DBSCAN=12.5
CURR_TOPK=-1
CURR_QUERY=20
CURR_SIZE=54


python main_instance_segmentation.py \
general.experiment_name="test_stpls_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}_size_${CURR_SIZE}" \
general.project_name="stpls3d_test" \
data/datasets=stpls3d \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.333 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
general.train_mode=false \
general.checkpoint="checkpoints/stpls3d/stpls3d_benchmark_03.ckpt" \
data.crop_length=${CURR_SIZE} \
general.eval_inner_core=50.0 \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN} \
data.test_mode=test \
general.export=true