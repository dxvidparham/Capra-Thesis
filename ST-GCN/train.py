# Copyright 2020-2022 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import torch
import wandb
from opendr.perception.skeleton_based_action_recognition import (
    ProgressiveSpatioTemporalGCNLearner,
    SpatioTemporalGCNLearner,
)

from opendr.engine.datasets import ExternalDataset


def main(args):
    # Action classifier
    if args.method == "pstgcn":
        model = ProgressiveSpatioTemporalGCNLearner(
            batch_size=32,
            checkpoint_after_iter=5,
            temp_path="temp/pstgcn",
            num_workers=16,
            epochs=5,
            experiment_name="pstgcn_capra",
            val_batch_size=32,
            dataset_name="capra",
            num_class=6,
            num_point=18,
            graph_type="openpose",
        )
    else:
        model = SpatioTemporalGCNLearner(
            batch_size=3,
            checkpoint_after_iter=5,
            temp_path="temp/stgcn",
            num_workers=16,
            epochs=50,
            experiment_name="stgcn_capra",
            val_batch_size=3,
            dataset_name="capra",
            num_class=6,
            num_point=18,
            graph_type="openpose",
            in_channels=2,
        )

    # Control wandb initialization
    if args.disable_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project="Thesis",
            entity="davidlbit",
            tags=["OpenPose"],
        )
        wandb.watch(model, log_freq=100)

    PATH = "./data/2D_skeletons/xview"
    train_dataset = ExternalDataset(path=PATH, dataset_type="KINETICS")
    val_dataset = ExternalDataset(path=PATH, dataset_type="KINETICS")

    model.fit(
        dataset=train_dataset,
        val_dataset=val_dataset,
        logging_path="log/",
        silent=True,
        train_data_filename="train_joints.npy",
        train_labels_filename="train_labels.pkl",
        val_data_filename="val_joints.npy",
        val_labels_filename="val_labels.pkl",
        skeleton_data_type="joint",
    )

    model.save(path=f"./saved_models/{args.method}", model_name="test_stgcn")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, default="stgcn", help="action detection method"
    )
    parser.add_argument(
        "--disable_wandb",
        help="Control wandb tracking",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
