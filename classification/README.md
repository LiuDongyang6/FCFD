# Environment
We test the codes with:
>
> + pytorch=1.10.0, torchvision=0.11.1
> + torchsummary=1.5.1
> + timm=0.5.4



# CIFAR-100

### Preparations

+ Download pre-trained teacher checkpoints from the [CRD](https://github.com/HobbitLong/RepDistiller/blob/master/scripts/fetch_pretrained_teachers.sh) repo.

+ Add the downloaded checkpoints to the `./pretrain` directory like this:

  ```
  FCFD/
    classification/
      pretrain/
        cifar_teachers/
          wrn_40_2_vanilla/
            ckpt_epoch_240.pth
          resnet32x4_vanilla/
            ckpt_epoch_240.pth
          ...
    detection/
      ...
    ...
  ```

  Alternatively, your may put the checkpoints anywhere and manually specify the `--teacher_ckpt` argument in your training command.



### Training

To train student models with FCFD, run the following command:

```bash
# . <train script path> <student> <teacher> <random seed (optional)>
. experiments/cifar/train_lr0.05.sh resnet8x4 resnet32x4
. experiments/cifar/train_lr0.01.sh ShuffleV2 resnet32x4

# specify the random seed to 113
. experiments/cifar/train_lr0.05.sh resnet8x4 resnet32x4 113
```
+ Your can modify the `*.sh` and the `*.yaml` files in [experiments/cifar](experiments/cifar) for custumized experiments.
+ See [ConfigExample.yaml](experiments/cifar/ConfigExample.yaml) to know the meaning of each configurable term.

The following is a list of all experiments:

```
. experiments/cifar/train_lr0.05.sh resnet8x4 resnet32x4
. experiments/cifar/train_lr0.05.sh resnet20 resnet56
. experiments/cifar/train_lr0.05.sh wrn_16_2 wrn_40_2
. experiments/cifar/train_lr0.05.sh wrn_40_1 wrn_40_2
. experiments/cifar/train_lr0.05.sh vgg8 vgg13

. experiments/cifar/train_lr0.01.sh ShuffleV1 resnet32x4
. experiments/cifar/train_lr0.01.sh ShuffleV1 wrn_40_2
. experiments/cifar/train_lr0.01.sh MobileNetV2 vgg13
. experiments/cifar/train_lr0.01.sh MobileNetV2 ResNet50
. experiments/cifar/train_lr0.01.sh ShuffleV2 resnet32x4
```



### Evaluation

Run the following command to evaluate:

```bash
# . <evaluate script path> <student> <ckpt_path>
. experiments/cifar/evaluate.sh resnet8x4 output/cifar/resnet8x4-resnet32x4-113/ckpt/model1_model_best.pth.tar
```



# ImageNet

### Preparations

Downland the ImageNet Dataset to `$HOME/data/imagenet/images`. 

+ Alternatively, you may specify dataset directory by the `--data_dir` argument



### Training

To train student models with FCFD, run the following command:

```bash
# . <train script path> <student> <teacher> <random seed (optional)>
. experiments/IN/train.sh resnet18IN resnet34IN
. experiments/IN/train.sh MBIN resnet50IN

# specify the random seed to 113
. experiments/IN/train.sh resnet18IN resnet34IN 113
```



### Evaluation

Run the following command to evaluate:

```bash
# . <evaluate script path> <student> <ckpt_path>
. experiments/IN/evaluate.sh resnet18IN output/IN/resnet18IN-resnet34IN-113/ckpt/model1_model_best.pth.tar
```
