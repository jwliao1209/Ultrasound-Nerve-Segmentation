# Ultrasound Nerve Segmentation

## Introduction of ultrasound nerve dataset
We join this task on [Kaggle [1]](https://github.com/Jia-Wei-Liao/Ultrasound_Nerve_Segmentation/#Reference). Surgery inevitably brings discomfort, and oftentimes involves significant post-surgical pain. In the past, the way to decrease the pain is to inject an anesthetic but bring a bevy of unwanted side effects. This competition's sponsor committed to improving pain management through the use of indwelling catheters that block or mitigate pain at the source. Pain management catheters reduce dependence on narcotics and speed up patient recovery. It is a critical step in finding the exact location 
so can assist to install the device. The task in this competition is to segment a collection of nerves called the Brachial Plexus (BP) in ultrasound images. We are dealing with ultrasound images dataset which contained 5635 training images with 2323 BP and 5508 test images.

![example](https://user-images.githubusercontent.com/93210989/148060792-3c1db347-6c8e-4b84-9ede-e66af46bcb33.png)

## Getting the code
You can download all the files in this repository by cloning this repository:
```
https://github.com/Jia-Wei-Liao/Ultrasound_Nerve_Segmentation.git
```

## Repository structure
```
      .
      ├──checkpoint
      |   └──2021-12-31-02-33-20-best
      ├──dataset
      |   ├──train
      |   |   ├──X_X.tif
      |   |   └──X_X_mask.tif
      |   ├──test
      |   |   └──X.tif
      |   ├──train_masks.csv
      |   ├──clean_masks.csv
      |   ├──Train_X.csv
      |   └──Valid_X.csv
      ├──src
      |   ├──configs.py
      |   ├──dataset.py
      |   ├──transforms.py
      |   ├──model.py
      |   ├──metric.py
      |   ├──logger.py      
      |   ├──trainer.py
      |   ├──RLE.py
      |   └──utils.py
      ├──create_EMS.py
      ├──visualize_mask.py
      ├──train.py
      └──inference.py      
```

## Requirements
```
tqdm==4.55.1
numpy==1.18.5
pandas==1.2.0
matplotlib==3.3.2
pillow==8.1.0
opencv-python==4.5.4.58
torch==1.10.0
torchvision==0.11.1
monai==0.7.0
segmentation_models_pytorch==0.2.1
libtiff=0.4.2
```

## Dataset
You can download the dataset on the Kaggle or our Google Drive:
1. Download the dataset from Kaggle  
<https://www.kaggle.com/c/ultrasound-nerve-segmentation/data>

2. Download the dataset from Google  
<https://drive.google.com/drive/folders/1-mmhwFzC-fS9hthWoyu7zdjmu9sxiUut?usp=sharing>


## Pre-trained weight
You can download the weight and checkpoint of our model and config on the Google Drive:
<https://drive.google.com/drive/folders/1AlgIqtetFxAl9lOGYcZrd96YLTiMTyX2?usp=sharing>

## Data preparation
**1. Split the train validation set**  
After downloading the dataset and put them into right place (see repo. structure above, ```train``` folder,
```test``` folder and ```train_masks.csv```), we first cleaning up the redundant training data by running
```
python clean_dataset.py
```
This step will generate ```clean_masks.csv```, then we can split training and validation dataset by running
```
python split_train_valid.py
```
This step will generate ```Train_X.csv``` and ```Valid_X.csv```.

**2. Erosion Mask Smoothing**  
If you want to implement the erosion mask smoothing, you might run the program ```python create_EMS.py ```.
Then it will create ```train_mask``` folder in dataset.

We also write a code to visualize the train_mask image ```python visualize_mask.py```
![pair](https://user-images.githubusercontent.com/93210989/148061341-555ef08f-35e5-4774-b236-7ddd5208cd74.png)


## Training
To train the model, you can run this command:
```
python train.py \
-bs <batch size of training step> \
-ep <epochs of training step> \
--weight_num <number of save weight> \
```

- dataset: 1, 2, 3, 4, 5
- model: smp_unet, smp_unetpp, deeplabv3pp
- pretrain: resnet{34, 50}, resnext50_32x4d, efficientnet-b{0, 1, 2}, timm-resnest{14, 26, 50}d
- activation: RELU, LRELU, SILU, MISH
- loss: DL, GDL, DCEL, DFL
- optim: sgd, adam, adamw
- scheduler: step, cos
- device: cpu or cuda:0


## Inference
To inference the results, you can run this command:
```
python inference.py \
--adaptive <use adaptive ensemble>
--checkpoint <checkpoint's filename> \
--ensem_num <number of ensemble weight> \
```
This step will output a csv file, which can be found in ```checkpoint/{checkpoint you have used}/answer.csv```

## Reproducing submission
To reproduce our submission, please do the following steps:
1. [Getting the code](https://github.com/Jia-Wei-Liao/Ultrasound_Nerve_Segmentation/#Getting-the-code)
2. [Install the package](https://github.com/Jia-Wei-Liao/Ultrasound_Nerve_Segmentation/#requirements)
3. [Download the dataset](https://github.com/Jia-Wei-Liao/Ultrasound_Nerve_Segmentation/#dataset)
4. [Download the weight of model](https://github.com/Jia-Wei-Liao/Ultrasound_Nerve_Segmentation/#pre-trained-weight)
5. [Inference](https://github.com/Jia-Wei-Liao/Ultrasound_Nerve_Segmentation/#Inference)

## Experiments
| method       | backbone        | private score |
| ------------ | --------------- | ------------- |
| UNet         | ResNet34        | 0.71031       |
| UNet         | ResNet50        | 0.70857       |
| UNet         | EﬀicientNet-b0  | 0.70233       |
| UNet         | EﬀicientNet-b1  | 0.72341       |

## Citation
```
@misc{
    title = {nuclear_dataset_segmentation},
    author = {Jia-Wei Liao, Kuok-Tong Ng, Yi-Cheng Hung},
    url = {https://github.com/Jia-Wei-Liao/Ultrasound_Nerve_Segmentation},
    year = {2022}
}
```

## Reference
[1] Kaggle: https://www.kaggle.com/c/ultrasound-nerve-segmentation  

[2] MONAI: https://github.com/Project-MONAI/MONAI  

[3] Segmentation Models: https://github.com/qubvel/segmentation_models.pytorch  
