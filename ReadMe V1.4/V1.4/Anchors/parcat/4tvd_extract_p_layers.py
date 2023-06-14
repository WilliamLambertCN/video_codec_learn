# remap.py
# Copyright (c) 2020-2021 Nokia Corporation and/or its subsidiary(-ies). All rights reserved.
# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import sys
sys.path.append("/detectron2")
import struct
import numpy as np
import torch, torchvision
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
import detectron2

from detectron2.utils.logger import setup_logger

setup_logger()

import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog

# import sys
import oid_mask_encoding

# import matplotlib.pyplot as plt

import argparse
from glob import glob

class PLayerExtractor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-praocessing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB,
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}

            images = self.model.preprocess_image([inputs])
            p_layers = self.model.backbone(images.tensor)
            # print(f"images in {images.device}")
            return p_layers


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='./', \
                    help='image directory')
parser.add_argument('--task', type=str, default='detection', \
                    help='task: detection or segmentation')
parser.add_argument('--input_file', type=str, default='input.lst', \
                    help='input file that contains a list of image file names')
parser.add_argument('--yuv_dir', type=str, default='./output_yuv', \
                    help='output directory')
parser.add_argument('--exp_folder', type=str, default='.', \
                    help='exp_folder directory')
args = parser.parse_args()

if args.task == 'detection':
    # model_cfg_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    # model_cfg_name = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    model_cfg_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
elif args.task == 'segmentation':
    model_cfg_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
else:
    assert False, print("Unrecognized task:", args.task)

# construct detectron2 model
print('constructing detectron model ...')

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(model_cfg_name))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg_name)
extractor = PLayerExtractor(cfg)

# 1. obtain global max and min (uncomment the following lines for evaluation)
if args.task=="detection":
    global_max = 28.397470474243164
    global_min = -26.426828384399414
    # global_min = 1e15
    # global_max = 1e-15
    # with open(args.input_file, 'r') as f:
    #   for idx, img_fname in enumerate(f.readlines()):
    #     img_fname = os.path.join(args.image_dir, img_fname.strip())
    #
    #     img = cv2.imread(img_fname)
    #     if img is None:
    #       img = cv2.imread(os.path.splitext(img_fname)[0] + '.jpg')
    #     assert img is not None, print(f'Image file not found: {img_fname}')
    #     print(f'processing {idx}:{img_fname}...')
    #
    #     # generate ouptut
    #     # stem extraction
    #     p_layers = extractor(img)
    #     # p_layers = p_layers.clone().detach().cpu().numpy()
    #     del p_layers["p6"]
    #
    #     max_values = [np.max(p_layers["p2"].cpu().numpy()), np.max(p_layers["p3"].cpu().numpy()),
    #             np.max(p_layers["p4"].cpu().numpy()), np.max(p_layers["p5"].cpu().numpy())]
    #     min_values = [np.min(p_layers["p2"].cpu().numpy()), np.min(p_layers["p3"].cpu().numpy()),
    #                   np.min(p_layers["p4"].cpu().numpy()), np.min(p_layers["p5"].cpu().numpy())]
    #     max = np.max(max_values)
    #     min = np.min(min_values)
    #     if max > global_max:
    #       global_max = max
    #     if min < global_min:
    #       global_min = min
    # print("global max: {}, global min: {}".format(global_max, global_min))

#segmentation
if args.task=="segmentation":
    global_max = 28.397470474243164
    global_min = -26.426828384399414
    # global_min = 1e15
    # global_max = 1e-15
    # with open(args.input_file, 'r') as f:
    #   for idx, img_fname in enumerate(f.readlines()):
    #     img_fname = os.path.join(args.image_dir, img_fname.strip())
    #
    #     img = cv2.imread(img_fname)
    #     if img is None:
    #       img = cv2.imread(os.path.splitext(img_fname)[0] + '.jpg')
    #     assert img is not None, print(f'Image file not found: {img_fname}')
    #     print(f'processing {idx}:{img_fname}...')
    #
    #     # generate ouptut
    #     # stem extraction
    #     p_layers = extractor(img)
    #     # p_layers = p_layers.clone().detach().cpu().numpy()
    #     del p_layers["p6"]
    #
    #     max_values = [np.max(p_layers["p2"].cpu().numpy()), np.max(p_layers["p3"].cpu().numpy()),
    #             np.max(p_layers["p4"].cpu().numpy()), np.max(p_layers["p5"].cpu().numpy())]
    #     min_values = [np.min(p_layers["p2"].cpu().numpy()), np.min(p_layers["p3"].cpu().numpy()),
    #                   np.min(p_layers["p4"].cpu().numpy()), np.min(p_layers["p5"].cpu().numpy())]
    #     max = np.max(max_values)
    #     min = np.min(min_values)
    #     if max > global_max:
    #       global_max = max
    #     if min < global_min:
    #       global_min = min
    # print("global max: {}, global min: {}".format(global_max, global_min))


# 2. Tile / Quantisation / Save YUV 4:0:0
# yuv_dir = "./extracted_stem_yuv"


############!!!!!!!!!!!!!!!!!ATENTION!!!!!!!!!!!!!!!!!!!!!!!!!!!##########
assert args.exp_folder != "."
methods=["priorart","remapping","padding"]
feats=["p2","p3","p4","p5","p2p5"]

feat_choosed,method=args.exp_folder.split("_")[-2:]

assert method in methods
assert feat_choosed in feats
# method=methods[1] #choose here

###########################################################################
yuv_dir = args.yuv_dir
if not os.path.isdir(yuv_dir):
    os.mkdir(yuv_dir)

lst=glob(f"{args.image_dir}/*.jpg")
total_frames=len(lst)

with open(f"{args.exp_folder}/{args.input_file}_yuvinfo_{method}_{args.task}.txt", 'w') as f_yuvinfo:
    for idx in range(1,total_frames+1):
        if idx>65:
            break

        orig_fname=f"{idx:04d}.jpg" # used twice

        img_fname = os.path.join(args.image_dir,orig_fname)

        img = cv2.imread(img_fname)
        if img is None:
            img = cv2.imread(img_fname)
        assert img is not None, print(f'Image file not found: {img_fname}')
        print(f'processing {idx}:{img_fname}...')

        # stem extraction
        p_layers = extractor(img)
        # p_layers = p_layers.clone().detach().cpu().numpy()
        del p_layers["p6"]
        # image_feat = quant_fix(features.copy())


        # 2-1 Tile
        # feat = [p_layers["p2"].squeeze().cpu(), p_layers["p3"].squeeze().cpu(), p_layers["p4"].squeeze().cpu(),
        #         p_layers["p5"].squeeze().cpu()]
        # width_list = [16, 32, 64, 128]
        # height_list = [16, 8, 4, 2]
        if feat_choosed in ["p2","p3","p4","p5"]:
            feat = [p_layers[feat_choosed].squeeze().cpu()]
            width_list = [16]
            height_list = [16]
        elif feat_choosed == "p2p5":
            feat = [p_layers["p2"].squeeze().cpu(), p_layers["p3"].squeeze().cpu(), p_layers["p4"].squeeze().cpu(),
                    p_layers["p5"].squeeze().cpu()]
            width_list = [16, 32, 64, 128]
            height_list = [16, 8, 4, 2]
        else:
            raise

        print(f"{method} is used")
###############################     1. prior_art  #######################
        if method=="priorart":
            plane = torch.empty((0, feat[0].shape[2] * width_list[0]))
            for blk, width, height in zip(feat, width_list, height_list):
                big_blk = torch.empty((0, blk.shape[2] * width))
                for row in range(height):
                    big_blk_col = torch.empty((blk.shape[1], 0))
                    for col in range(width):
                        tile = blk[col + row * width]
                        # if debug:
                        #     cv2.putText(
                        #         tile,
                        #         f"{col + row * width}",
                        #         (32, 32),
                        #         cv2.FONT_HERSHEY_SIMPLEX,
                        #         0.5,
                        #         (255, 255, 255),
                        #         1,
                        #     )
                        big_blk_col = torch.hstack((big_blk_col, tile))
                    big_blk = torch.vstack((big_blk, big_blk_col))
                plane = torch.vstack((plane, big_blk))
            
            plane=plane.numpy()
            
#################################   2.remapping   #####################
        elif method=="remapping":
            blks = feat[0].numpy()
            blks = np.transpose(blks)
            print(blks.shape)
            blks = np.reshape(blks, (blks.shape[0], blks.shape[1], 16, 16))
            print(blks.shape)
            plane = blks[0, 0, :, :]
            for n in range (1, blks.shape[0]): 
                plane = np.hstack((plane, blks[n, 0, :,:]))
                
            for i in range (1, blks.shape[1]): 
                temp = blks[0, i, :, :]
                for j in range (1, blks.shape[0]):
                    temp = np.hstack((temp, blks[j, i, :, :]))
                plane = np.vstack((plane,temp))

################################   3.padding   ########################
        elif method=="padding":
            nearest_2size=int(2**np.ceil(np.log2(max(2*feat[0].shape[1],feat[0].shape[2])))) 
            plane = np.empty((0, nearest_2size * width_list[0]//2))

            for blk, width, height in zip(feat, width_list, height_list):
                # 2*h, w
                nearest_2size=int(2**np.ceil(np.log2(max(2*blk.shape[1],blk.shape[2])))) 
                big_blk = np.empty((0, nearest_2size * width//2))
                # rows to fill in left,right,up,down
                pad_right = nearest_2size-blk.shape[2]
                pad_down = nearest_2size-2*blk.shape[1]
                # np.pad(ar,((up,down),(left,right)),'edge')
                for row in range(height):
                    big_blk_col = np.empty((nearest_2size, 0))
                    for col in range(0,width,2):
                        tile1=blk[col + row * width].numpy()
                        tile2=blk[col + row * width +1].numpy()
                        tile = np.vstack((tile1,tile2))
                        tile = np.pad(tile,((0,pad_down),(0,pad_right)),'edge')

                        big_blk_col = np.hstack((big_blk_col, tile))
                    big_blk = np.vstack((big_blk, big_blk_col))
                plane = np.vstack((plane, big_blk))

#########################################################################
        # 2-2 Quantisation
        bits = 10
        steps = np.power(2, bits) - 1
        scale = steps / (global_max - global_min)
        scaled_plane = (plane - global_min) * scale
        quantized_plane = np.round(scaled_plane).astype(int)
        # for debug
        # cv2.imwrite('temp.png', q_plane/4)

        # 2-3 Save YUV
        yuv_fname = os.path.splitext(orig_fname)[0] + '.yuv'
        yuv_fname = f"{yuv_dir}/{yuv_fname.strip()}"
        height = quantized_plane.shape[0]
        width = quantized_plane.shape[1]
        f_yuvinfo.write("{},{},{}\n".format(yuv_fname, width, height))

        flattened_quantized_plane = quantized_plane.reshape(-1)

        with open(yuv_fname, "wb") as f_yuv:
            for int_val in flattened_quantized_plane:
                bytes = (int_val.item()).to_bytes(2, byteorder='little')
                f_yuv.write(bytes)