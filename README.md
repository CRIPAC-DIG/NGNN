# NGNN
The code and dataset for our paper in the WebConf2019: Dressing as a Whole: Outfit Compatibility Learning Based on Node-wise Graph Neural Networks [[arXiv version]](https://arxiv.org/abs/1902.08009)

## Paper data and code

This is the code for the WWW-2019 Paper: [Dressing as a Whole: Outfit Compatibility Learning Based on Node-wise Graph Neural Networks](https://arxiv.org/abs/1902.08009). We have implemented our methods in both **Tensorflow**.

Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder `datasets/`:

- YOOCHOOSE: <http://2015.recsyschallenge.com/challenge.html>

- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup>

There is a small dataset `sample` included in the folder `datasets/`, which can be used to test the correctness of the code.

We have also written a [blog](https://sxkdz.github.io/research/SR-GNN) explaining the paper.

## Usage

You need to run the file  `datasets/preprocess.py` first to preprocess the data.

For example: `cd datasets; python preprocess.py --dataset=sample`

```bash
usage: preprocess.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  dataset name: diginetica/yoochoose/sample
```

Then you can run the file `NGNN/main_score.py` to train the model.

You can change parameters according to the usage in `NGNN/Config.py`:

```bash
usage: main.py [-h] [--dataset DATASET] [--batchSize BATCHSIZE]
               [--hiddenSize HIDDENSIZE] [--epoch EPOCH] [--lr LR]
               [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--l2 L2]
               [--step STEP] [--patience PATIENCE] [--nonhybrid]
               [--validation] [--valid_portion VALID_PORTION]

parameters arguments:
    epoch_num           the max epoch number
    train_batch_size    training batch size
    valid_batch_size    validation batch size
    hidden_size = 16    hidden size of the NGNN

    lstm_forget_bias = 0.0
    
    max_grad_norm = 1
    init_scale = 0.05
    learning_rate = 0.01  # 0.001  # 0.2
    decay = 0.5
    decay_when = 0.002  # AUC
    decay_epoch = 200
    sgd_opt = 'RMSProp'
    beta = 0.0001
    GNN_step = 3
    dropout_prob = 0
    adagrad_eps = 1e-5
    gpu = 0
                        
                        
                        
```

## Requirements

- Python 2.7
- Tensorflow 1.5.0

## Citation

Please cite our paper if you use the code:

```
@inproceedings{cui2019dressing,
  title={Dressing as a Whole: Outfit Compatibility Learning Based on Node-wise Graph Neural Networks},
  author={Cui, Zeyu and Li, Zekun and Wu, Shu and Zhang, Xiao-Yu and Wang, Liang},
  booktitle={The World Wide Web Conference},
  pages={307--317},
  year={2019},
  organization={ACM}
}
```

