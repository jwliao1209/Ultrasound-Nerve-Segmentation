# Ultrasound_Nerve_Segmentation

## Training
To train the model, you can run this command:
```
python train.py\
--model monai_unet\
--init_filter 32\
--depth 4\
--dropout 0.2\
--loss DL\
--optim adamw\
--lr 1e-3\
--weight_decay 1e-4\
--scheduler step\
--batch_size 8\
--epoch 100\
--device cuda:0\
```

- model: monai_unet
- loss: DL, GDL, DCEL, DFL
- optim: sgd, adam, adamw
- scheduler: step, cos
- device: cpu or cuda:{0, 1, 2, 3}
