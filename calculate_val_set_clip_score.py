#!/usr/bin/env python
from PIL import Image
import requests
import tensorflow as tf

import pandas as pd

from transformers import AutoProcessor, TFCLIPVisionModel, TFCLIPTextModel, TFCLIPModel, CLIPProcessor

model = TFCLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

coco_val_df = pd.read_csv('../coco2014/captions/captions.tsv', sep='\t')

print(f'{"id"}\t{"caption"}\t{"file_name"}\t{"clip_score"}') # outputs['logits_per_image'][0][0].numpy())

# the coco 2014 validation set has 5,000 lable images
for i in range(5000):
    caption, file_name = coco_val_df.iloc[i]['caption'], coco_val_df.iloc[i]['file_name']
    # print(caption)
    image = Image.open(f'../coco2014/val2014/{file_name}')
    inputs = processor(text=[caption], images=image, return_tensors="tf", padding="longest")
    outputs = model(**inputs)
    print(f'{coco_val_df.iloc[i]["id"]}\t{caption}\t{file_name}\t{outputs["logits_per_image"][0][0].numpy()}')
