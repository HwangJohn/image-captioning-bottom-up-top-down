# vizwiz json
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "vizwiz-caption"))
from vizwiz_api.vizwiz import VizWiz
from vizwiz_eval_cap.eval import VizWizEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import pylab
from tqdm import tqdm
#pylab.rcParams['figure.figsize'] = (8.0, 10.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

annTrainFile = '../mypythia/data/vizwiz/annotations/train.json'
annValFile    = '../mypythia/data/vizwiz/annotations/val.json'

vizwizTrain = VizWiz(annTrainFile, ignore_rejected=True, ignore_precanned=True)
vizwizVal = VizWiz(annValFile, ignore_rejected=True, ignore_precanned=True)

imgIdsTrain = vizwizTrain.getImgIds()

# load and display caption annotations
annIdsTrain = vizwizTrain.getAnnIds();
annsTrain = vizwizTrain.loadAnns(annIdsTrain)

from vizwiz_eval_cap.tokenizer.ptbtokenizer import PTBTokenizer
tokenizer = PTBTokenizer()

try:

    vizwiz_images = []
    gts = dict()
    imgIds = vizwizTrain.getImgIds()
    for img_id in tqdm(imgIds):

        gts[img_id] = vizwizTrain.imgToAnns[img_id]

        # sent ids
        filepath = "../mypythia/data/vizwiz/train/"
        sentids = []    
        filename = vizwizTrain.imgs[img_id]['file_name']
        imgid = gts[img_id]    
        split = "train"
        sentences = []
        for g in gts[img_id]:

            # tokenizer를 사용하기 위한 포멧 맞춤
            tmp = {0:[{'caption':g['caption']}]}
            tokens = tokenizer.tokenize(tmp)
            tokens = tokens[0][0].split(" ")
            raw = g['caption']
            imgid = g['image_id']
            sentid = g['id']

            sentence = {'tokens':tokens,
                       'raw':raw,
                       'imgid':imgid,
                       'sentid':sentid}        

            sentids.append(g['id'])
            sentences.append(sentence)        

        row = {"filepath":filepath,
              "sentids":sentids,
              "filename":filename,
              "imgid":imgid,
              "split":split,
              "sentences":sentences}
        vizwiz_images.append(row)


    vizwiz_anno = {'images':vizwiz_images, 'dataset':'vizwiz'}
    with open("dataset_vizwiz.json", "w") as f:
        json.dump(vizwiz_anno, f)
    
except Exception as e:
    print(e)
    with open("dataset_vizwiz.json", "w") as f:
        json.dump(vizwiz_anno, f)
