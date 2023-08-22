import os
import numpy as np
from fire import Fire
from natsort import natsorted

from datasets.preprocessing.base_preprocessing import BasePreprocessing
from utils.point_cloud_utils import load_ply_trees



class TreesPreprocessing(BasePreprocessing):
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
        
        filebase = {
            "filepath": filepath,
            "scene": filepath.split("/")[-1],
            "raw_filepath": str(filepath),
            "file_len": -1,
        }

        # read in ply's
        points, semantic_labels, instance_labels = load_ply_trees(filepath)
        file_len = len(points)
        filebase["file_len"] = file_len
        filebase["raw_segmentation_filepath"] = "" # TODO: ?

        # rgb, segment, normal dummy
        dummy_rgb = np.ones((points.shape[0], 3))*255
        segments_dummy = np.ones((points.shape[0], 1))
        dummy_normals = np.ones((points.shape[0], 3))

        points = np.hstack((points,
                            dummy_rgb,
                            segments_dummy,
                            dummy_normals,
                            semantic_labels[:,None],
                            instance_labels[:,None]))

        # encode gt data: semantic label * 1000 + instance_label per line
        gt_data = (points[:, -2] + 1) * 1000 + points[:, -1]

        # save scenes and instance_gt

        processed_filepath = self.save_dir / mode / f"{filebase['scene'].replace('.ply', '')}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        processed_gt_filepath = self.save_dir / "instance_gt" / mode / f"{filebase['scene'].replace('.ply', '')}.txt"
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)
        
        # TODO: we don't use color, skip this?
        filebase["color_mean"] = [
            float((points[:, 3] / 255).mean()),
            float((points[:, 4] / 255).mean()),
            float((points[:, 5] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((points[:, 3] / 255) ** 2).mean()),
            float(((points[:, 4] / 255) ** 2).mean()),
            float(((points[:, 5] / 255) ** 2).mean()),
        ]
        return filebase

    def compute_color_mean_std( # TODO: fix train database path
            self, train_database_path: str = "./data/processed/trees/train_database.yaml"
        ):
            train_database = self._load_yaml(train_database_path)
            color_mean, color_std = [], []
            for sample in train_database:
                color_std.append(sample["color_std"])
                color_mean.append(sample["color_mean"])

            color_mean = np.array(color_mean).mean(axis=0)
            color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean ** 2)
            feats_mean_std = {
                "mean": [float(each) for each in color_mean],
                "std": [float(each) for each in color_std],
            }
            self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)



if __name__ == "__main__":
    Fire(TreesPreprocessing)