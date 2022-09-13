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


import contextlib
import numpy as np
import os
import torch
import cv2
import time
import pandas as pd
from typing import Dict
import wandb
from ptflops import get_model_complexity_info
import sys
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from utils import gaze_direction

# opendr imports
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.pose_estimation.lightweight_open_pose.utilities import draw
import argparse
from opendr.perception.skeleton_based_action_recognition import (
    ProgressiveSpatioTemporalGCNLearner,
)
from opendr.perception.skeleton_based_action_recognition import SpatioTemporalGCNLearner


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        with contextlib.suppress(ValueError):
            self.file_name = int(file_name)

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError(f"Video {self.file_name} cannot be opened")
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def get_text_size(id):
    return (
        cv2.getTextSize(f"Person ID: {id}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] // 2
    )


def tile(a, dim, n_tile):
    a = torch.from_numpy(a)
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    tiled_a = torch.index_select(a, dim, order_index)
    return tiled_a.numpy()


def pose2numpy(args, num_current_frames, poses_list):
    C = 2
    T = args.num_frames
    V = 18
    M = 2  # num_person_in
    data_numpy = np.zeros((1, C, num_current_frames, V, M))
    skeleton_seq = np.zeros((1, C, T, V, M))
    for t in range(num_current_frames):
        for m in range(len(poses_list[t])):
            data_numpy[0, 0:2, t, :, m] = np.transpose(poses_list[t][m].data)

    # if we have less than num_frames, repeat frames to reach num_frames
    diff = T - num_current_frames
    if diff == 0:
        skeleton_seq = data_numpy
    while diff > 0:
        num_tiles = int(diff / num_current_frames)
        if num_tiles > 0:
            data_numpy = tile(data_numpy, 2, num_tiles + 1)
            num_current_frames = data_numpy.shape[2]
            diff = T - num_current_frames
        elif num_tiles == 0:
            skeleton_seq[:, :, :num_current_frames, :, :] = data_numpy
            for j in range(diff):
                skeleton_seq[:, :, num_current_frames + j, :, :] = data_numpy[
                    :, :, -1, :, :
                ]
            break
    return skeleton_seq


def select_2_poses(poses):
    energy = []
    for i in range(len(poses)):
        s = poses[i].data[:, 0].std() + poses[i].data[:, 1].std()
        energy.append(s)
    energy = np.array(energy)
    index = energy.argsort()[::-1][:2]
    return [poses[index]]


CAPRA_CLASSES = pd.read_csv(
    "data/labels/capra_labels.csv", verbose=True, index_col=0
).to_dict()["name"]


def preds2label(confidence):
    k = 3
    class_scores, class_inds = torch.topk(confidence, k=k)
    return {
        CAPRA_CLASSES[int(class_inds[j])]: float(class_scores[j].item())
        for j in range(k)
    }


def draw_preds(frame, preds: Dict):
    for i, (cls, prob) in enumerate(preds.items()):
        cv2.putText(
            frame,
            f"{prob:04.3f} {cls}",
            (10, 40 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )


# def draw_direction(frame, eyes):
#     eyes = set(eyes)
#     if len(eyes) > 1:
#         text = "FRONT"
#     else:
#         text = "BACK"

#     cv2.putText(
#         frame,
#         text,
#         (10, 200),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 255, 255),
#         2,
#     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument(
        "--device", help="Device to use (cpu, cuda)", type=str, default="cuda"
    )
    parser.add_argument(
        "--accelerate",
        help="Enables acceleration flags (e.g., stride)",
        default=False,
        action="store_true",
    )
    parser.add_argument("--video", default=0, help="path to video file or camera id")
    parser.add_argument(
        "--method", type=str, default="stgcn", help="action detection method"
    )
    parser.add_argument(
        "--action_checkpoint_name",
        type=str,
        default="test_stgcn",
        help="action detector model name",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=300,
        help="number of frames to be processed for each action",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="number of frames per second to be processed by pose estimator and action detector",
    )
    parser.add_argument(
        "--viewing_mode",
        type=str,
        choices=["skeleton", "rgb"],
        default="rgb",
        help="choose if the keypoints will be plotted on the rgb image or a black scale image",
    )
    parser.add_argument(
        "--save",
        help="Save output in avi video file",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--disable_wandb",
        help="Control wandb tracking",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--hide_FPS",
        help="Puts the FPS on the images",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    onnx, device = args.onnx, args.device
    accelerate = args.accelerate
    hide_FPS = args.hide_FPS
    onnx, device, accelerate = args.onnx, args.device, args.accelerate
    if accelerate:
        stride = True
        stages = 0
        half_precision = True
    else:
        stride = False
        stages = 2
        half_precision = False

    # pose estimator
    pose_estimator = LightweightOpenPoseLearner(
        device=device,
        num_refinement_stages=stages,
        mobilenet_use_stride=stride,
        half_precision=half_precision,
    )
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")

    # Action classifier
    if args.method == "pstgcn":
        action_classifier = ProgressiveSpatioTemporalGCNLearner(
            device=device,
            dataset_name="capra",
            topology=[5, 4, 5, 2, 3, 4, 3, 4],
            in_channels=2,
            num_point=18,
            graph_type="openpose",
        )
    else:
        action_classifier = SpatioTemporalGCNLearner(
            device=device,
            batch_size=32,
            checkpoint_after_iter=5,
            temp_path="temp/stgcn",
            num_workers=16,
            epochs=50,
            experiment_name="stgcn_capra",
            method_name=args.method,
            val_batch_size=32,
            dataset_name="capra",
            num_class=6,
            num_point=18,
            graph_type="openpose",
            in_channels=2,
        )

    print("print_numpoints", action_classifier.num_point)
    action_classifier.classes_dict = CAPRA_CLASSES

    action_classifier.load(
        path="./saved_models/stgcn",
        model_name="test_stgcn",
    )

    # Optimization
    if onnx:
        pose_estimator.optimize()
        action_classifier.optimize()

    # torch.backends.quantized.engine = "qnnpack"
    # pose_estimator.model = torch.quantization.quantize_dynamic(
    #     pose_estimator.model, {torch.nn.Conv2d}, dtype=torch.qint8
    # )
    # macs, params = get_model_complexity_info(
    #     pose_estimator.model,
    #     (3, 224, 224),
    #     as_strings=True,
    #     print_per_layer_stat=True,
    #     verbose=True,
    # )
    # print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    # print("{:<30}  {:<8}".format("Number of parameters: ", params))
    # sys.exit(-1)
    image_provider = VideoReader(args.video)  # loading a video or get the camera id 0
    if args.save:
        # Obtain frame size information using get() method
        frame_width = 640
        frame_height = 480
        frame_size = (frame_width, frame_height)
        fps = 20

        # Initialize video writer object
        output = cv2.VideoWriter(
            "output_video_from_file.avi",
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            fps,
            frame_size,
        )

    # Control wandb initialization
    disable_wandb = args.disable_wandb
    if disable_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project="Thesis",
            entity="davidlbit",
            tags=["OpenPose", device, "single", "Action recognition"],
        )
        wandb.watch(pose_estimator.model, log_freq=100)
        wandb.watch(action_classifier.model, log_freq=100)

    try:
        counter, avg_fps = 0, 0
        poses_list = []
        window = int(30 / args.fps)
        f_ind = 0
        for img in image_provider:
            if f_ind % window == 0:
                # eyes = []
                start_time = time.perf_counter()
                poses = pose_estimator.infer(img)
                centroid_dict = {}

                if args.viewing_mode == "skeleton":
                    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
                    mask[:] = (0, 0, 0)
                    img = mask

                # eyes.extend(poses[0]["l_eye"])
                # eyes.extend(poses[0]["r_eye"])
                # draw_direction(img, eyes)

                for pose in poses:
                    head_position_x = pose["nose"][0]
                    head_position_y = int(pose["nose"][1] * 0.7)

                    img = cv2.putText(
                        img,
                        f"Person ID: {pose._id}",
                        (head_position_x - get_text_size(pose.id), head_position_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    x1, x2, x3 = pose["neck"][0], pose["l_hip"][0], pose["r_hip"][0]
                    y1, y2, y3 = pose["neck"][1], pose["l_hip"][1], pose["r_hip"][1]

                    x = (x1 + x2 + x3) // 3
                    y = (y1 + y2 + y3) // 3

                    centroid_dict[str(pose.id)] = (x, y)
                    try:
                        gaze_direction(img, pose)
                    except Exception as e:
                        # print(e)
                        pass
                    draw(img, pose)

                # Measure the distance between subjects
                if len(centroid_dict) > 1:
                    centroid_mat = np.array(list(centroid_dict.values()))
                    proximity_measurement = squareform(pdist(centroid_mat))
                    # print(proximity_measurement)
                    G = nx.from_numpy_matrix(proximity_measurement)

                    for edge in list(G.edges()):
                        distance = G.get_edge_data(*edge).get("weight")
                        distance = distance * 0.0264583333
                        print(
                            f"Distance between Person {edge[0]} and Person {edge[1]} is {distance:.2f} cm"
                        )
                    else:
                        print("")

                if len(poses) > 0:
                    if len(poses) > 2:
                        # select two poses with highest energy
                        poses = select_2_poses(poses)
                    counter += 1
                    print(counter)
                    poses_list.append(poses)

                if counter > args.num_frames:
                    poses_list.pop(0)
                    counter = args.num_frames
                if counter > 0:
                    skeleton_seq = pose2numpy(args, counter, poses_list)

                    prediction = action_classifier.infer(skeleton_seq)
                    category_labels = preds2label(prediction.confidence)
                    print(category_labels)
                    draw_preds(img, category_labels)

                # Calculate a running average on FPS
                end_time = time.perf_counter()
                fps = 1.0 / (end_time - start_time)
                avg_fps = 0.8 * fps + 0.2 * fps
                wandb.log({"fps": avg_fps})
                # Wait a few frames for FPS to stabilize
                if counter > 5 and not hide_FPS:
                    img = cv2.putText(
                        img,
                        "FPS: %.2f" % (avg_fps,),
                        (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

                if args.save:
                    output.write(img)
                cv2.imshow("Result", img)
            f_ind += 1
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
    except Exception as e:
        print(e)

    wandb.finish()
    print("Average inference fps: ", avg_fps)
    cv2.destroyAllWindows()
