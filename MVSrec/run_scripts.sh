#!/bin/bash

SCENE_DIR="../generate_mvimages/results--20241011-205152"      
PLY_PATH="mvsgs_pointcloud"  
ITER="3000"                  

for SCENE_NAME in $(ls ${SCENE_DIR}); do
    echo "Processing scene: ${SCENE_NAME}"
    
    python lib/colmap/imgs2poses.py -s ${SCENE_DIR}/${SCENE_NAME}

    python run.py --type evaluate --cfg_file configs/mvsgs/colmap_eval.yaml test_dataset.data_root ${SCENE_DIR}/${SCENE_NAME} save_ply True dir_ply ${PLY_PATH}

    
    python lib/train.py --eval --iterations ${ITER} -s ${SCENE_DIR}/${SCENE_NAME} -p ${PLY_PATH}

    
    python lib/render.py -c -m output/${SCENE_NAME} --iteration ${ITER} -p ${PLY_PATH}

    python lib/metrics.py -m output/${SCENE_NAME}

    echo "Completed processing scene: ${SCENE_NAME}"
done
