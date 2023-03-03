
# PUT+: Optimizing Transformer for Large-Hole Image Inpainting

## Introduction

In recent years, leveraging Convolutional Neural Network (CNN) to optimize Transformer (called hybrid model) has received great progress in image inpainting. However, the slow growth of the effective receptive field of CNN in processing large-hole regions significantly limits the overall performance. To alleviate this problem, this paper proposes a new Transformer-CNN-based hybrid framework (termed PUT+) by introducing the fast Fourier convolution (FFC) into the CNN-based refinement network. The proposed framework introduces an improved Patch-based Vector Quantized Variational Auto-Encoder (P-VQVAE+). The encoder transforms the masked region into non-overlapping patch-based unquantized feature vectors as the input of Un-Quantized Transformer (UQ-Transformer). The decoder restores the masked region from the predicted quantized features output by the UQ-Transformer while maintaining the unmasked region unchanged. Many experimental results show that the proposed method outperforms the state-of-the-art by a large margin, especially for image inpainting with large masked areas.

## Dataset Preparation

**Image Dataset.** We evaluate the proposed method on the [Paris Street View](https://github.com/pathak22/context-encoder) and [Places2](http://places2.csail.mit.edu/) datasets, which are widely adopted in the literature.

**Mask Dataset.** We use the mask provided by [PConv](https://nv-adlr.github.io/publication/partialconv-inpainting). Only the testing mask is needed, which can be download from [here](https://www.dropbox.com/s/01dfayns9s0kevy/test_mask.zip?dl=0).

## Environment setup

Clone the repo:
`git clone https://github.com/GZHU-DVL/PUTplus.git`

Preparing the environment:

```
conda create -n ImgSyn python=3.7
conda activate ImgSyn

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

pip install -r requirements.txt

python setup.py build develop --user
```


## Training

Run the training script:

```
cd PUTplus

python train_net --name exp_name --config_file path/to/config.yaml --num_node 1 --tensorboard --auto_resume
```

NOTE: Train P-VQVAE+ first and then train UQ-Transformer.


## Inference
Infer model and run evaluation:

```
python scripts/inference.py --name OUTPUT/transformer_exp_name/checkpoint/last.pth --func inference_complet_sample_in_feature_for_evaluation --gpu 0 --batch_size 1

sh scripts/metrics/cal_metrics.sh path/to/gt path/to/result
```

## Acknowledgments

The code is developed based on [PUT](https://github.com/liuqk3/PUT) and [LaMa](https://github.com/advimman/lama).