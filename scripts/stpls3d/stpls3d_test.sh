#!/bin/bash
export OMP_NUM_THREADS=3

CURR_DBSCAN=12.5
CURR_TOPK=200
CURR_QUERY=160
CURR_SIZE=54
CURR_THRESHOLD=0.01


python main_instance_segmentation.py \
general.experiment_name="test_stpls_TODI_ALS_GENT_033" \
general.project_name="stpls3d_test" \
data/datasets=stpls3d \
general.num_targets=15 \
data.num_labels=15 \
data.voxel_size=0.33 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
general.train_mode=false \
general.checkpoint="data/Synthetic_v3_InstanceSegmentation/stpls3d_benchmark/stpls3d_benchmark_03.ckpt" \
data.crop_length=${CURR_SIZE} \
general.topk_per_image=${CURR_TOPK} \
data.test_mode=test \
general.export=true \
general.save_visualizations=true \
# general.use_dbscan=true \
# general.dbscan_eps=${CURR_DBSCAN} \