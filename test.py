import torch
from torchvision import ops
 
# Bounding box coordinates.
ground_truth_bbox = torch.tensor([[1202, 123, 1650, 868]], dtype=torch.float)
prediction_bbox = torch.tensor([[1162.0001, 92.0021, 1619.9832, 694.0033]], dtype=torch.float)

print(ground_truth_bbox.shape)
print(prediction_bbox.shape)
 
# Get iou.
iou = ops.box_iou(ground_truth_bbox, prediction_bbox)
print('IOU : ', iou.numpy()[0][0])
