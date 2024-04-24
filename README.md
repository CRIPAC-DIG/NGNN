# NGNN

<img src="https://github.com/CRIPAC-DIG/NGNN/blob/d433faf82c3bc65de7ea63714828306a900699ae/figures/cmgnn_framework.png" alt="model" style="zoom: 100%;" />

This is the code for the WebConf 2019 Paper: [Dressing as a Whole: Outfit Compatibility Learning Based on Node-wise Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3308558.3313444).

## Usage

### Paper data and code

The original Polyvore dataset we used in our paper is first proposed [here](https://github.com/xthan/polyvore-dataset). After downloaded the datasets, you can put them in the folder `NGNN/data/`:

You can download the preprocessed data here, <https://drive.google.com/open?id=1ibYEw0H9L9O9OLbxCiAlcZkt_IYuwKfd>  and also put them in the folder `NGNN/data/`.

There is a small dataset `sample` included in the folder `NGNN/data/`, which can be used to test the correctness of the code.

**the data preprocess is written in the `./data/README.md`**

### Quick Start

Then you can run the file `NGNN/main_score.py` to train the model.

You can change parameters according to the usage in `NGNN/Config.py`:

```bash

parameters arguments in `NGNN/Config.py`:

    epoch_num           the max epoch number
    train_batch_size    training batch size
    valid_batch_size    validation batch size
    hidden_size         hidden size of the NGNN
    lstm_forget_bias    forget bias in NGNN update
    max_grad_norm       the gradient clip during train
    init_scale          the scale of initialize parameter 0.05
    learning_rate       learning rate  0.01  # 0.001  # 0.2
    decay               the decay of 0.5
    decay_when = 0.002  # AUC
    decay_epoch = 200
    sgd_opt             train strategy can choose: 'RMSProp', 'Adam', 'Momentum', 'RMSProp', 'Adadelta'
    beta                the weight of regulartion
    GNN_step            the number of step of GNN
    dropout_prob        the dropout probability of our model
    adagrad_eps         eps
    gpu = 0             the gpu id            
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



