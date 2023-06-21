import json

import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# load Mask2Former fine-tuned on COCO panoptic segmentation
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
# class_queries_logits = outputs.
# print(class_queries_logits)
# masks_queries_logits = outputs.masks_queries_logits
# print(masks_queries_logits)
# you can pass them to processor for postprocessing
# result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)
# segments_info = result["segments_info"]
# print(segments_info)
# labels = [label_map.get(segment["label_id"], "unknown") for segment in segments_info]
# print(labels)


# load COCO label map
with open("./id2label.json", "r") as f:
    label_map = json.load(f)

# postprocess the segmentation result using COCO labels
result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]], label_map=label_map)[0]

# convert label_ids to labels using category list
categories = result["categories"]
labels = [categories[segment["category_id"] - 1]["name"] for segment in result["segments_info"]]
print(labels)