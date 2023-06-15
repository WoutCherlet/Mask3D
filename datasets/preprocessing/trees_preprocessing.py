import re
import os
import numpy as np
from fire import Fire
from natsort import natsorted
from loguru import logger
import pandas as pd

from datasets.preprocessing.base_preprocessing import BasePreprocessing


class STPLS3DPreprocessing(BasePreprocessing):
    def __init__(
            self,
            data_dir: str = "../../data/raw/trees/", # TODO: name?
            save_dir: str = "../../data/processed/trees",
            modes: tuple = ("train", "validation", "test"),
            n_jobs: int = -1
    ):
        
        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.class_map = {
            'Terrain': 0,
            'Tree': 1,
            # 'Lying woody debris': 2,
            # 'Standing woody debris': 3,
            # 'Understory': 4,
            # 'Tripod': 5
        }

        self.color_map = [
            [0, 0, 255],     # Terrain
            [255, 0 , 0],    # Tree
            [0, 255, 255],   # Lying Woody debris
            [255, 255, 0],   # Standing woody debris
            [0, 255, 0],     # Understory
            [255, 0, 255]]   # Tripod
        
        self.create_label_database()

        for mode in self.modes:
            filepaths = []
            for scene_path in [f.path for f in os.scandir(self.data_dir / mode)]:
                filepaths.append(scene_path)
            self.files[mode] = natsorted(filepaths)
    
    def create_label_database(self):
        label_database = dict()
        for class_name, class_id in self.class_map.items():
            label_database[class_id] = {
                'color': self.color_map[class_id],
                'name': class_name,
                'validation': True
            }

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database
        
    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        
        # TODO!

        pass










if __name__ == "__main__":
    Fire(STPLS3DPreprocessing)