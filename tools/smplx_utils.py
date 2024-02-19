import os
import random
import json
import pickle as pkl
import cv2
import numpy as np
import imageio
import torch


class SMPLXLiteSeg:
    def __init__(self, json_path="data/smplx_lite/spec_verts_idx.json"):
        self.json_path = json_path

    def get_head_vertex_ids(self):
        smplx_lite_segs = json.laod(self.json_path)
        head_ids = smplx_lite_segs["head"]

        return head_ids
    

class SMPLXSeg:
    smplx_dir = "./data/smplx"
    smplx_segs = json.load(open(f"{smplx_dir}/smplx_vert_segementation.json"))
    flame_segs = pkl.load(open(f"{smplx_dir}/FLAME_masks.pkl", "rb"), encoding='latin1')
    smplx_face = np.load(f"{smplx_dir}/smplx_faces.npy")

    # 'eye_region',  , 'right_eye_region', 'forehead', 'lips', 'nose', 'scalp', 'boundary', 'face', 'left_ear', 'left_eye_region']
    # 'rightHand', 'rightUpLeg', 'leftArm', 'head',
    # 'leftEye', 'rightEye', 'leftLeg', 'leftToeBase', 'leftFoot',
    # 'spine1', 'spine2', 'leftShoulder', 'rightShoulder', 'rightFoot', 'rightArm',
    # 'leftHandIndex1', 'rightLeg', 'rightHandIndex1', 'leftForeArm', 'rightForeArm', 'neck',
    # 'rightToeBase', 'spine', 'leftUpLeg', 'eyeballs', 'leftHand', 'hips']
    # print(smplx_segs.keys())
    # exit()

    smplx_flame_vid = np.load(f"{smplx_dir}/FLAME_SMPLX_vertex_ids.npy", allow_pickle=True)

    eyeball_ids = smplx_segs["leftEye"] + smplx_segs["rightEye"]
    hands_ids = smplx_segs["leftHand"] + smplx_segs["rightHand"] + \
                smplx_segs["leftHandIndex1"] + smplx_segs["rightHandIndex1"]
    neck_ids = smplx_segs["neck"]
    head_ids = smplx_segs["head"]

    front_face_ids = list(smplx_flame_vid[flame_segs["face"]])
    ears_ids = list(smplx_flame_vid[flame_segs["left_ear"]]) + list(smplx_flame_vid[flame_segs["right_ear"]])
    forehead_ids = list(smplx_flame_vid[flame_segs["forehead"]])
    lips_ids = list(smplx_flame_vid[flame_segs["lips"]])
    nose_ids = list(smplx_flame_vid[flame_segs["nose"]])
    eyes_ids = list(smplx_flame_vid[flame_segs["right_eye_region"]]) + list(
        smplx_flame_vid[flame_segs["left_eye_region"]])
    check_ids = list(
        set(front_face_ids) - set(forehead_ids) - set(lips_ids) - set(nose_ids) - set(eyes_ids)
    )
    

def get_mesh_center_scale(phrase, vertices):
    # TODO need vertices from SMPLX
    if phrase == "head":
        vertices = vertices[SMPLXSeg.head_ids + SMPLXSeg.neck_ids]
    max_v = vertices.max(0)[0]
    min_v = vertices.min(0)[0]
    scale = (max_v[1] - min_v[1])
    center = (max_v + min_v) * 0.5
    # center = torch.mean(points, dim=0, keepdim=True)
    return center, scale