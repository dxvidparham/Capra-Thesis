import contextlib
import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx

old_pose = {}


def get_keypoint_values(pose):
    # Available keypoints on the face
    nose = pose["nose"]
    l_eye = pose["l_eye"]
    r_eye = pose["r_eye"]
    l_ear = pose["l_ear"]
    r_ear = pose["r_ear"]

    # Prevents gaze direction from using (-1 values if keypoint not detected)
    # Instead last observed keypoint value will be used
    if -1 in r_ear:
        r_ear = [120, 132]
    else:
        old_pose["r_ear"] = r_ear

    if -1 in l_ear:
        l_ear = old_pose["l_ear"]
    else:
        old_pose["l_ear"] = l_ear

    return nose, l_eye, r_eye, l_ear, r_ear


def gaze_direction(img, pose):
    size = img.shape
    nose, l_eye, r_eye, l_ear, r_ear = get_keypoint_values(pose)
    img_points = np.array([nose, l_eye, r_eye, l_ear, r_ear], dtype="double")
    model_points = np.array(
        [
            (0, 0, 0),
            (-225, 120, -135),
            (225, 120, -135),
            (-350, 85, -350),
            (350, 85, -350),
        ],
        dtype="double",
    )

    focal_length = size[1]
    center = size[1] / 2, size[0] / 2
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeffs = np.zeros((4, 1))
    flags = [
        cv2.SOLVEPNP_ITERATIVE,
        cv2.SOLVEPNP_P3P,
        cv2.SOLVEPNP_EPNP,
        cv2.SOLVEPNP_AP3P,
        cv2.SOLVEPNP_IPPE,
        cv2.SOLVEPNP_IPPE_SQUARE,
        cv2.SOLVEPNP_SQPNP,
    ]

    with contextlib.suppress(Exception):
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, img_points, camera_matrix, dist_coeffs, flags=flags[6]
        )

    # print(img_points)
    nose_end_point2D, jacobian = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]),
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs,
    )

    # Responsible for the red dots
    # for p in img_points:
    #     # cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 0), -1)

    p1 = int(img_points[0][0]), int(img_points[0][1])
    p2 = int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])
    cv2.line(img, p1, p2, (255, 0, 0), 2)


def get_text_size(id):
    return (
        cv2.getTextSize(f"Person ID: {id}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] // 2
    )


def print_distances(centroid_dict):
    centroid_mat = np.array(list(centroid_dict.values()))
    proximity_measurement = squareform(pdist(centroid_mat))
    G = nx.from_numpy_matrix(proximity_measurement)

    for edge in list(G.edges()):
        distance = G.get_edge_data(*edge).get("weight")
        distance = distance
        print(
            f"Distance between Person {edge[0]} and Person {edge[1]} is {distance:.2f} cm"
        )
    print("")
