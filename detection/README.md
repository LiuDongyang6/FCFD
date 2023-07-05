## FCFD on COCO detection

### Environment

We verify our code on 
* 1 x Tesla V100 GPU
* CUDA 11.0
* python 3.7
* torch 1.10.0
* torchvision 0.11.1

Other similar envirouments should also work properly.

### Installation
Please refer to [ReviewKD](https://github.com/dvlab-research/ReviewKD/tree/master/Detection#installation).

### Training

Use the following commands to reproduce our results:
```
python train_net.py --config-file configs/FCFD-R18-R101.yaml
python train_net.py --config-file configs/FCFD-R50-R101.yaml
python train_net.py --config-file configs/FCFD-MV2-R50.yaml
```

Use the following command to train with the inheriting strategy:
```
python train_net.py --config-file configs/FCFD-R50-R101-INHERIT.yaml
```


## Acknowledgement
The code is modified from the [ReviewKD](https://github.com/dvlab-research/ReviewKD) project. Thanks for their wonderful work!