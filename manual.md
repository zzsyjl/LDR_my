
## Functions
Here we list sample use cases of various functions. The code should be able to run successfully.

### Training
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --cfg experiments/cifar10.yaml
```

### Produce test acc
```bash
CUDA_VISIBLE_DEVICES=1 python test_acc.py --cfg logs/cifar10_LDR_multi_mini_dcgan/config.yaml --ckpt_epochs 45000 EVAL.DATA_SAMPLE 1000
CUDA_VISIBLE_DEVICES=1 python test_acc.py --cfg logs/0514cifar10_1split_try/config.yaml --ckpt_epochs 45000 EVAL.DATA_SAMPLE 1000
```
Acc of CIFAR10 ~73%, 50%

```bash
CUDA_VISIBLE_DEVICES=1 python test_acc.py --cfg logs/mnist_LDR_multi/config.yaml --ckpt_epochs 4500 EVAL.DATA_SAMPLE 1000
```
Acc of MNIST ~97%

### Plot Cosine Similariy Matrix
#### MNIST
```bash
CUDA_VISIBLE_DEVICES=1 python cosSimMatrixFigure.py --dataset mnist
```
Successfully plot the orthogonal subspaces
#### CIFAR10
```bash
CUDA_VISIBLE_DEVICES=1 python cosSimMatrixFigure.py --cfg logs/cifar10_LDR_multi_mini_dcgan/config.yaml --ckpt_epochs 45000
CUDA_VISIBLE_DEVICES=1 python cosSimMatrixFigure.py --cfg logs/0514cifar10_1split_try/config.yaml --ckpt_epochs 45000
```
Accommodate to different model arch
