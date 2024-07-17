from PIL import Image
import requests
import pandas as pd

import tensorflow as tf
from transformers import AutoProcessor, TFCLIPVisionModel, TFCLIPTextModel, TFCLIPModel, CLIPProcessor

from tensorflow.io import TFRecordWriter
from tensorflow.train import Feature
from tensorflow.train import Features
from tensorflow.train import Example

model = TFCLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# I have coco2014 in ../coco2014/
coco_val_df = pd.read_csv('../coco2014/captions/captions.tsv', sep='\t')

caption_bytes = []
filename_bytes = []

to_bytes = lambda x: str.encode(x)

for s in coco_val_df['caption'].to_list():
    caption_bytes.append(to_bytes(s))
for s in coco_val_df['file_name'].to_list():
    filename_bytes.append(to_bytes(s))

id_list = coco_val_df['id'].to_list()

print(f'{"id"}\t{"caption"}\t{"ids_padded"}\t{"file_name"}\t{"clip_score"}') # outputs['logits_per_image'][0][0].numpy())

recordWriter = TFRecordWriter('/tmp/foo.tfrecord', options='ZLIB')
# the coco 2014 validation set has 5,000 labelled images
for i in range(5000):
    caption, file_name = coco_val_df.iloc[i]['caption'], coco_val_df.iloc[i]['file_name']
    # print(caption)
    image = Image.open(f'../coco2014/val2014/{file_name}')
    inputs = processor(text=[caption], images=image, return_tensors="tf", padding="longest")

    ids = inputs['input_ids'][0].numpy().tolist()
    id_len = len(ids)
    ids_padded = ids + ([49407] * (77 - id_len))

    outputs = model(**inputs)
    clip_score = outputs["logits_per_image"][0][0].numpy()
    print(f'{coco_val_df.iloc[i]["id"]}\t{caption}\t{ids_padded}\t{file_name}\t{clip_score}')

    id_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[id_list[i]]))
    caption_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[caption_bytes[i]]))
    tokenized_ids_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=ids_padded))
    filename_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename_bytes[i]]))
    clip_score_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[clip_score]))
    features = Features(feature={
        "id": id_feature,
        "caption": caption_feature,
        "tokenized_ids": tokenized_ids_feature,
        "file_name": filename_feature,
        "clip_score": clip_score_feature,
    })
    example = Example(features=features)
    serializedExample = example.SerializeToString()
    recordWriter.write(serializedExample)
recordWriter.close()
