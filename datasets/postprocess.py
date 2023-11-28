import os
import yaml

import numpy as np
from fire import Fire
from loguru import logger
import open3d as o3d

class BasePostProcessing:
    def __init__(
        self,
        database = "./data/processed/trees/validation_database.yaml",
        results_dir = "./eval_output/"
    ):
        self.database = database
        if not os.path.exists(database):
            logger.error("file database doesn't exist")
            raise FileNotFoundError
    
        if not os.path.exists(results_dir):
            logger.error("results folder doesn't exist")
            raise FileNotFoundError
        
        self.ply_dir = os.path.join(results_dir, "visualizations", "ply")
        if not os.path.exists(self.ply_dir):
            logger.error(f"ply folder doesn't exist, make sure save_visualizations is True and ply's are saved at {ply_dir}")
            raise FileNotFoundError
        
        self.odir = os.path.join(results_dir, "retransformed_out")
    
    def postprocess(self):
        transform_dict = self.read_transforms()
        for scene in transform_dict:
            scene_dir = os.path.join(self.ply_dir, scene)

            scene_o_dir = os.path.join(self.odir, scene)

            for file in os.listdir(scene_dir):
                pc = o3d.io.read_point_cloud(file)
                points = np.asarray(pc.points)
                scaling, translation = transform_dict[scene]
                points = self.retransform(points, scaling, translation)
                pc.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(os.path.join(scene_o_dir, os.path.basename(file)), pc)
        return

    def read_transforms(self):
        files = self._load_yaml(self.database)
        transform_dict = {filebase["scene"]: (filebase["scaling"], filebase["translation"]) for filebase in files}
        return transform_dict

    def retransform(self, points, scale_factor, translation):
        points = np.multiply(points, 1/scale_factor)
        points = np.substract(points, -translation)
        return points
    
    @classmethod
    def _load_yaml(cls, filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file



if __name__ == "__main__":
    Fire(BasePostProcessing)