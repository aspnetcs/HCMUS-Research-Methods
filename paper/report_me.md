# 4. Experiment

In this section, we will describe the implementation of our mathematical formula detection system & dataset in detail

**Dataset**: Our data is from [*IBEM dataset*](https://doi.org/10.5281/zenodo.4757864). This originally consists of 600 documents, with a total of 8272 pages. Those documents are parsed from mathetical paper, then each page is annotated with bounding box of 2 types: `isolated` and `embedded` . The dataset is then splited into various sets for *ICDAR 2021 Competition on Mathematical Formula Detection*, includes Training, Validation and Test sets:

Training
- Tr00: 4082 pages
- Tr01: 760 pages
- Tr10: 329 pages

Test
- Ts00: 736 pages
- Ts01: 380 pages
- Ts10: 699 pages
- Ts11: 329 pages

Validation
- Va00: 577 pages
- Va01: 380 pages 

Our experiment uses Tr01, Tr10, Ts01 for training, Va01 for validation and Ts11 for testing with total 2178 pages (~26.33% of original dataset) and approximate ratio 4.47 : 1.16 : 1. The reason for this small subset is for evaluating the ability of model on small sets, and the performance it gives (F1-score) through time (minutes).

**Implement details**: Our baseline model is `Faster RCNN` with `Resnet50` as backbone. We've trained on kaggle with 4 cores CPU, 12GB RAM and 1 Nvidia Tesla P100 GPU[\[*\]](https://www.kaggle.com/docs/notebooks). The image is resized to 1447x2048 with the same ratio. The size of region crops from image is 1200x1120 to fit the limitation of machine. It's also flipped and padded for data augmentation. For the feature aggregation, we use FPN (2-6). Loss function for classifier is `Cross Entropy Loss` and for bounding box is `L1 Loss`. Test images are resize to 1583x2048 due to distribution of test dataset, flip augmentation is also applied. For post processing, Non-Maximum Suppression (NMS) with 0.5 IoU threshold to remove redundant boxes. All models are trained based on MMDetection toolbox and config given by [Yuxiang Zhong](https://github.com/Yuxiang1995/ICDAR2021_MFD/blob/main/configs/_base_/models/faster_rcnn_r50_fpn.py).


