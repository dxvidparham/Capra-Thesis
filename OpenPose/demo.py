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

import cv2
import time
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.pose_estimation import draw
import argparse
import contextlib
import numpy as np
from utils import gaze_direction, get_text_size, print_distances


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
        "--hide_FPS",
        help="Puts the FPS on the images",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    # assign cmd line arguments to corresponding variables
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

    pose_estimator = LightweightOpenPoseLearner(
        device=device,
        num_refinement_stages=stages,
        mobilenet_use_stride=stride,
        half_precision=half_precision,
    )
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")

    if onnx:
        pose_estimator.optimize()

    # Use the first camera available on the system
    image_provider = VideoReader(0)
    if args.save:
        # The correct width and height of the frames has to be used to save the file
        frame_width = 640
        frame_height = 360
        frame_size = (frame_width, frame_height)
        fps = 20

        # Initialize video writer object
        output = cv2.VideoWriter(
            "1Person.avi",
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            fps,
            frame_size,
        )

    try:
        counter, avg_fps = 0, 0
        for i, img in enumerate(image_provider):

            start_time = time.perf_counter()

            # Perform inference
            poses = pose_estimator.infer(img)
            end_time = time.perf_counter()
            fps = 1.0 / (end_time - start_time)
            centroid_dict = {}

            if args.viewing_mode == "skeleton":
                mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
                mask[:] = (0, 0, 0)
                img = mask

            # Draw subject ID
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
                with contextlib.suppress(Exception):
                    gaze_direction(img, pose)
                draw(img, pose)

            # Measure the distance between subjects
            if len(centroid_dict) > 1:
                print_distances(centroid_dict)

            # Calculate a running average on FPS
            avg_fps = 0.8 * fps + 0.2 * fps

            # Wait a few frames for FPS to stabilize
            if counter > 10 and not hide_FPS:
                image = cv2.putText(
                    img,
                    "FPS: %.2f" % (avg_fps,),
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            if args.save:
                output.write(img)
            cv2.imshow("Result", img)
            if cv2.waitKey(1) & 0xFF == ord("y"):  # save on pressing 'y'
                cv2.imwrite(f"images/img{i}_skeleton.png", img)
            counter += 1
    except Exception as e:
        print(e)
        print("Average inference fps: ", avg_fps)
