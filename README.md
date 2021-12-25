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
--model smp_unet \
--pretrain resnet50 \
--init_filter 32 \
--depth 4 \
--dropout 0.2 \
--mish: 0
--loss DFL \
--optim adamw \
--lr 1e-3 \
--weight_decay 1e-4 \
--scheduler cos \
--step_size 1 \
--gamma  0.95 \
--device cuda:0 \
--weight_num 10
--fliplr 0 \
--flipud 0 \
--rot90 0 \
--bright 0 \
--noise 0
```

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
--checkpoint <checkpoint's filename> \
--weight <model weight's name> \
--test_batch 32
```
