# Repo for training, optimizing and deploying MTCNN for mobile devices

## Prepare dataset
1. Download [WIDER FACE]() dataset and put them under `./data` directory.
2. Transform matlab format label train and val format file into text format.
   RUN `pyhon gen_dataset/transform_mat2txt.py`
   Change the mode variable in `transform_mat2txt.py` to `val` to generate val data label.
   
## Train MTNN
### Train Pnet
1. generate pnet training data:
    RUN `python gen_dataset/gen_Pnet_data.py`
    Change the mode variable in `gen_Pnet_data.py` to `val` to generate val data label.
2. Training Pnet:
    RUN `python training/pnet/train.py` to train your model.
3. Save weights:
    We use the validation dataset to help us with choose the best pnet model. The weights are saved in `pretrained_weights/best_pnet.pth`

### Train Rnet
After we trained Pnet, we can use Pnet to generate data for training Rnet.
1. generate Rnet training data:
    RUN `python gen_dataset/gen_Rnet_data.py`
    Change the mode variable in `gen_Pnet_data.py` to `val` to generate val data label.
2. Training Rnet:
    RUN `python training/rnet/train.py` to train your model.
3. 3. Save weights:
    We use the validation dataset to help us with choose the best pnet model. The weights are saved in `pretrained_weights/best_rnet.pth`
    
### Train Onet
After we trained Pnet and Rnet, we can use Pnet and Rnet to generate data for training Onet.
1. generate Onet training data:
    RUN `python gen_dataset/gen_Onet_data.py`
    Change the mode variable in `gen_Pnet_data.py` to `val` to generate val data label.
2. Training Onet:
    RUN `python training/Onet/train.py` to train your model.
3. 3. Save weights:
    We use the validation dataset to help us with choose the best pnet model. The weights are saved in `pretrained_weights/best_onet.pth`
    
### Results

|  WIDER FACE |  Pnet  |  Rnet |  Onet |
| :---------: |:------:|:-----:|:-----:|
|   cls loss  |  0.156 | 0.120| 0.129 |
| offset loss |  0.01  | 0.01 | 0.0063|
|   cls acc   |  0.944 | 0.962| 0.956 |

| PRIVATE DATA|  Pnet  |  Rnet |  Onet |
| :---------: |:------:|:-----:|:-----:|
|   cls loss  |  0.156 | 0.120| 0.129 |
| offset loss |  0.01  | 0.01 | 0.0063|
|   cls acc   |  0.944 | 0.962| 0.956 |

## Optimize MTCNN
### Lighter MTCNN
By combine shufflenet structure and mobilenet structure we can design light weight Pnet, Rnet, and Onet. In this way can can optimize the size of the model and at the same time increase the inference speeed.

### Prune MTCNN

### Knowledge 

### Deploy MTCNN


## Todo
[] Data Augmentation to avoid overfitting

