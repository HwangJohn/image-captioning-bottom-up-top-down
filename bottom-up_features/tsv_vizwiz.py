"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.
Hierarchy of HDF5 file:
{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

import os
import io
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.join(os.getcwd(), 'vqa-maskrcnn-benchmark'))

import base64
import csv
import h5py
import pickle as cPickle
import numpy as np
import utils
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

OUTPUT_BASEPATH = 'mid_pre_input'

os.makedirs(OUTPUT_BASEPATH, exist_ok=True)

train_data_file = os.path.join(OUTPUT_BASEPATH,'train36_vizwiz.hdf5')
val_data_file = os.path.join(OUTPUT_BASEPATH, 'val36_vizwiz.hdf5')
train_indices_file = os.path.join(OUTPUT_BASEPATH, 'train36_imgid2idx_vizwiz.pkl')
val_indices_file = os.path.join(OUTPUT_BASEPATH, 'val36_imgid2idx_vizwiz.pkl')
train_ids_file = os.path.join(OUTPUT_BASEPATH, 'train_ids_vizwiz.pkl')
val_ids_file = os.path.join(OUTPUT_BASEPATH, 'val_ids_vizwiz.pkl')

feature_length = 2048
num_fixed_boxes = 36

# npy image feature 가져오기
import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
from tqdm import tqdm
import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd
import csv
import base64
import pickle

from glob import glob
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
# from IPython.display import display, HTML, clear_output
# from ipywidgets import widgets, Layout
from io import BytesIO

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

input_img_paths = []
train_paths = glob("../mypythia/data/vizwiz/train/*.jpg")
val_paths = glob("../mypythia/data/vizwiz/val/*.jpg")
train_paths = sorted(train_paths, key=lambda x: int(os.path.split(x)[-1].split("_")[-1].split(".")[0]))
val_paths = sorted(val_paths, key=lambda x: int(os.path.split(x)[-1].split("_")[-1].split(".")[0]))

input_img_paths.extend(train_paths)
input_img_paths.extend(val_paths)

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
tsv_info_list = []
BASE_PATH = '../mypythia'

def _image_transform(image_path):

    img = Image.open(image_path)
    im = np.array(img).astype(np.float32)
    im = im[:, :, ::-1]
    im -= np.array([102.9801, 115.9465, 122.7717])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(800) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > 1333:
       im_scale = float(1333) / float(im_size_max)
    im = cv2.resize(
       im,
       None,
       None,
       fx=im_scale,
       fy=im_scale,
       interpolation=cv2.INTER_LINEAR
    )
    img = torch.from_numpy(im).permute(2, 0, 1)
    return img, im_scale

def _build_detection_model():
    
    cfg.merge_from_file(os.path.join(BASE_PATH,'content/model_data/detectron_model.yaml'))
    cfg.freeze()

    model = build_detection_model(cfg)
    checkpoint = torch.load(os.path.join(BASE_PATH,'content/model_data/detectron_model.pth'), 
                          map_location=torch.device("cpu"))

    load_state_dict(model, checkpoint.pop("model"))

    model.to("cuda")
    model.eval()
    return model

# ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
def _process_feature_extraction(output,
                             im_scales,
                             feat_name='fc6',
                             conf_thresh=0.2):
    batch_size = len(output[0]["proposals"])
    n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
    score_list = output[0]["scores"].split(n_boxes_per_image)
    score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
    feats = output[0][feat_name].split(n_boxes_per_image)
    cur_device = score_list[0].device

    feat_list = []
    keep_boxes_list = []

    for i in range(batch_size):
        dets = output[0]["proposals"][i].bbox / im_scales[i]
        #         print(f"im_scales[i]: {im_scales[i]!r}")
        scores = score_list[i]

        max_conf = torch.zeros((scores.shape[0])).to(cur_device)

        for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.5)
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                       cls_scores[keep],
                                       max_conf[keep])

        keep_boxes = torch.argsort(max_conf, descending=True)[:num_fixed_boxes]
        feat_list.append(feats[i][keep_boxes])
        
        
    return feat_list, output[0]["proposals"][0].bbox[keep_boxes]

#batchsize 는 1로 고정
def get_tsv_info(path):
    
    image_id = int(os.path.split(path)[-1].split('_')[-1].split('.')[0])
    
    im, im_scale = _image_transform(path)
    img_tensor, im_scales = [im], [im_scale]
    current_img_list = to_image_list(img_tensor, size_divisible=32)
    current_img_list = current_img_list.to('cuda')
    output = model(current_img_list)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    feat_list, bbox = _process_feature_extraction(output, im_scales, 'fc6', 0.2)
    
    image_width = output[1][0].size[0]
    image_height= output[1][0].size[1]
    print(path, ' saved!!')
    return {'image_id':image_id, 'image_w':image_width, 'image_h':image_height, 'num_boxes':num_fixed_boxes, 'boxes':np.array(bbox.tolist()), 'features':np.array(feat_list[0].tolist())}    
    #     return {'image_id':image_id, 'image_w':image_width, 'image_h':image_height, 'num_boxes':100, 'boxes':base64.encodebytes(np.array(bbox.tolist()).tobytes()), 'features':base64.encodebytes(np.array(feat_list[0].tolist()).tobytes())}
    #     return [image_id, image_width, image_height, 100, base64.encodestring(np.array(bbox.tolist()).tobytes()), base64.encodestring(np.array(feat_list[0].tolist()).tobytes())]


model = _build_detection_model()

h_train = h5py.File(train_data_file, "w")
h_val = h5py.File(val_data_file, "w")

if os.path.exists(train_ids_file) and os.path.exists(val_ids_file):
    print(f"----------p train_ids_file: {train_ids_file!r}")
    train_imgids = cPickle.load(open(train_ids_file))
    val_imgids = cPickle.load(open(val_ids_file))
else:
    train_imgids = utils.load_imageid('../mypythia/data/vizwiz/train')
    val_imgids = utils.load_imageid('../mypythia/data/vizwiz/val')
    cPickle.dump(train_imgids, open(train_ids_file, 'wb'),protocol=2)
    cPickle.dump(val_imgids, open(val_ids_file, 'wb'),protocol=2)

train_indices = {}
val_indices = {}

train_img_features = h_train.create_dataset(
    'image_features', (len(train_imgids), num_fixed_boxes, feature_length), 'f')
train_img_bb = h_train.create_dataset(
    'image_bb', (len(train_imgids), num_fixed_boxes, 4), 'f')
train_spatial_img_features = h_train.create_dataset(
    'spatial_features', (len(train_imgids), num_fixed_boxes, 6), 'f')

val_img_bb = h_val.create_dataset(
    'image_bb', (len(val_imgids), num_fixed_boxes, 4), 'f')
val_img_features = h_val.create_dataset(
    'image_features', (len(val_imgids), num_fixed_boxes, feature_length), 'f')
val_spatial_img_features = h_val.create_dataset(
    'spatial_features', (len(val_imgids), num_fixed_boxes, 6), 'f')

train_counter = 0
val_counter = 0

for path in tqdm(input_img_paths):
    item = get_tsv_info(path)
    item['num_boxes'] = int(item['num_boxes'])
    image_id = int(item['image_id'])
    image_w = float(item['image_w'])
    image_h = float(item['image_h'])
    bboxes = item['boxes'].reshape((item['num_boxes'], -1))

    box_width = bboxes[:, 2] - bboxes[:, 0]
    box_height = bboxes[:, 3] - bboxes[:, 1]
    scaled_width = box_width / image_w
    scaled_height = box_height / image_h
    scaled_x = bboxes[:, 0] / image_w
    scaled_y = bboxes[:, 1] / image_h

    box_width = box_width[..., np.newaxis]
    box_height = box_height[..., np.newaxis]
    scaled_width = scaled_width[..., np.newaxis]
    scaled_height = scaled_height[..., np.newaxis]
    scaled_x = scaled_x[..., np.newaxis]
    scaled_y = scaled_y[..., np.newaxis]

    spatial_features = np.concatenate(
        (scaled_x,
            scaled_y,
            scaled_x + scaled_width,
            scaled_y + scaled_height,
            scaled_width,
            scaled_height),
        axis=1)

    if image_id in train_imgids:
        train_imgids.remove(image_id)
        train_indices[image_id] = train_counter
        train_img_bb[train_counter, :, :] = bboxes
        train_img_features[train_counter, :, :] = item['features'].reshape((item['num_boxes'], -1))
        train_spatial_img_features[train_counter, :, :] = spatial_features
        train_counter += 1
    elif image_id in val_imgids:
        val_imgids.remove(image_id)
        val_indices[image_id] = val_counter
        val_img_bb[val_counter, :, :] = bboxes
        val_img_features[val_counter, :, :] = item['features'].reshape((item['num_boxes'], -1))
        val_spatial_img_features[val_counter, :, :] = spatial_features
        val_counter += 1
    else:
        assert False, 'Unknown image id: %d' % image_id

if len(train_imgids) != 0:
    print('Warning: train_image_ids is not empty')

if len(val_imgids) != 0:
    print('Warning: val_image_ids is not empty')

cPickle.dump(train_indices, open(train_indices_file, 'wb'))
cPickle.dump(val_indices, open(val_indices_file, 'wb'))
h_train.close()
h_val.close()
print("done!")
