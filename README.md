# Ultrasound_Nerve_Segmentation

## Repository structure
```
      .
      ├──checkpoint
      ├──dataset
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
      ├──train.py
      └──inference.py
```


## Training
To train the model, you can run this command:
```
python train.py \
-bs 8 \
-ep 100 \
--dataset 2 \
--model smp_unet \
--pretrain efficientnet-b1 \
--activation RELU \
--loss DFL \
--optim adamw \
--lr 5E-4 \
--weight_decay 1e-4 \
--scheduler cos \
--step_size 1 \
--gamma  0.95 \
--device cuda:0 \
--weight_num 10 \
--fliplr 0.5 \
--flipud 0.25 \
--bright 0.1 \
--noise 0.1
```

--dataset: 1, 2, 3, 4, 5
- model: monai_unet, smp_unet, smp_unetpp
- msih: 0, 1
- pretrain: resnet{34, 50}, resnext50_32x4d, efficientnet-b0, timm-resnest{14, 26, 50}d
- loss: DL, GDL, DCEL, DFL
- optim: sgd, adam, adamw
- scheduler: step, cos
- device: cpu or cuda:{0, 1, 2, 3}


## Inference
To inference the results, you can run this command:
```
python inference.py \
--adaptive <use adaptive ensemble>
--checkpoint <checkpoint's filename> \
--ensem_num <number of ensemble weight> \
```
