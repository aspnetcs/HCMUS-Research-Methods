_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/formula_detection.py', #coco_detection
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
