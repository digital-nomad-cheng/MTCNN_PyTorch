# Repo for training, optimizing and deploying MTCNN for mobile devices

## Prepare dataset
1. Download [WIDER FACE]() dataset and put them under `./data` directory.
2. Transform matlab format label train and val format file into text format.
   RUN `pyhon gen_dataset/transform_mat2txt.py`
   Change the mode variable in `transform_mat2txt.py` to `val` to generate val data label.
   
## Train MTCNN
### Train Pnet
1. generate pnet training data:
    RUN `python gen_dataset/gen_Pnet_data.py`
    Change the mode variable in `gen_Pnet_data.py` to `val` to generate val data label.
2. Training Pnet:
    RUN `python training/pnet/train.py` to train your model.
3. Save weights:
    We use the validation dataset to help us with choose the best pnet model. The weights are saved in `pretrained_weights/mtcnn/best_pnet.pth`

### Train Rnet
After we trained Pnet, we can use Pnet to generate data for training Rnet.
1. generate Rnet training data:
    RUN `python gen_dataset/gen_Rnet_data.py`
    Change the mode variable in `gen_Pnet_data.py` to `val` to generate val data label.
2. Training Rnet:
    RUN `python training/rnet/train.py` to train your model.
3. 3. Save weights:
    We use the validation dataset to help us with choose the best rnet model. The weights are saved in `pretrained_weights/mtcnn/best_rnet.pth`
    
### Train Onet
After we trained Pnet and Rnet, we can use Pnet and Rnet to generate data for training Onet.
1. generate Onet training data:
    RUN `python gen_dataset/gen_Onet_data.py`
    Change the mode variable in `gen_Pnet_data.py` to `val` to generate val data label.
2. Training Onet:
    RUN `python training/Onet/train.py` to train your model.
3. 3. Save weights:
    We use the validation dataset to help us with choose the best onet model. The weights are saved in `pretrained_weights/mtcnn/best_onet.pth`
    
### Results

|  WIDER FACE |  Pnet  |  Rnet |  Onet |
| :---------: |:------:|:-----:|:-----:|
|   cls loss  |  0.156 | 0.120| 0.129 |
| offset loss |  0.01  | 0.01 | 0.0063|
|   cls acc   |  0.944 | 0.962| 0.956 |

| PRIVATE DATA|  Pnet  |  Rnet |  Onet |
| :---------: |:------:|:-----:|:-----:|
|   cls loss  |  0.05  | 0.09 | 0.104 |
| offset loss | 0.0047 | 0.011 | 0.0057|
|   cls acc   |  0.983 | 0.971 | 0.970 |

## Optimize MTCNN
### Lighter MTCNN
By combine shufflenet structure and mobilenet structure we can design light weight Pnet, Rnet, and Onet. In this way can can optimize the size of the model and at the same time decrease the inference speeed.

### Larger Pnet
According to my observation, small pnet brings many false positives which becomes a burden or rnet and onet. By increase the Pnet size, there will be less false positives and improve the overall efficiency.

### Prune MTCNN

Model Prunning is a better strategy than design mobile cnn for such small networks as Pnet, Rnet, and Onet. By iteratively pruning MTCNN models, we can decrease and model size and improve inference speed at the same time.

### Quantization Aware Training

By using quantization aware training library [brevitas](https://github.com/Xilinx/brevitas), I managed to achieve 96.2% accuracy on Pnet which is 2% lower than the original version, but the model size if 4x smaller and the inference speed is to be estimated.

However, when training Rnet and Onet, OOM errors occured. I will figure out why in the future.

| PRIVATE DATA|  Pnet  |  Rnet |  Onet |
| :---------: |:------:|:-----:|:-----:|
|   cls loss  |  0.107  | - | - |
| offset loss | 0.0080 | - | - |
|   cls acc   |  0.962 | - | - |


### Knowledge Distillation

## Deploy MTCNN


## Todo
- [ ] Data Augmentation to avoid overfitting

## References
1. https://github.com/xuexingyu24/MTCNN_Tutorial
2. https://github.com/xuexingyu24/Pruning_MTCNN_MobileFaceNet_Using_Pytorch

