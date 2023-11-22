import os
import yaml

import numpy as np
from fire import Fire
from loguru import logger



# TODO: plan: read in val_database for file names
# then read in points in visualizations/ply and retransform points

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
    
        # if not os.path.exists(results_dir):
        #     logger.error("results folder doesn't exist")
        #     raise FileNotFoundError
        
        # self.ply_dir = os.path.join(results_dir, "visualizations", "ply")
        # if not os.path.exists(self.ply_dir):
        #     logger.error(f"ply folder doesn't exist, make sure save_visualizations is True and ply's are saved at {ply_dir}")
        #     raise FileNotFoundError
        
    def postprocess(self):
        # TODO: do all postprocessing here
        transforms = self.read_transforms()
        pass
    
    def read_transforms(self):
        # TODO: figure out best way to read points and redo transform
        files = self._load_yaml(self.database)
        transform_dict = {filebase["scene"]: (filebase["scaling"], filebase["translation"]) for filebase in files}
        print(transform_dict)
        return transform_dict

    def retransform(points, scale_factor, translation):
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