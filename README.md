# LLaSE-G1: Incentivizing Generalization Capability for LLaMA-based Speech Enhancement

<p align="center">
  <a href="https://arxiv.org/abs/2503.00493">
    <img src="https://img.shields.io/badge/Paper-ArXiv-red.svg" alt="Paper">
  </a>
  <a href="https://submission-papers.github.io/LLaSE-G1-demo-page/">
    <img src="https://img.shields.io/badge/Demo-Page-blue.svg" alt="Demo">
  </a>
  <a href="https://huggingface.co/ASLP-lab/LLaSE-G1">
    <img src="https://img.shields.io/badge/Model-Hugging%20Face-yellow.svg" alt="Hugging Face">
  </a>
</p>

![LLaSE-G1](LLaSE-G1.png)


## Introduction

LLaSE-G1 is a unified speech enhancement model capable of handling multiple tasks without extra task prompts, including:

- **Noise Suppression (SE)**
- **Target Speaker Extraction (TSE)**
- **Packet Loss Concealment (PLC)**
- **Acoustic Echo Cancellation (AEC)**
- **Speech Separation (SS)**

To mitigate acoustic inconsistency, LLaSE-G1 employs continuous representations from **WavLM** as input and predicts speech tokens using **X-Codec2**, maximizing acoustic preservation. The model surpasses prior task-specific discriminative and generative speech enhancement models, demonstrating scaling effects at test time and emerging capabilities for unseen speech enhancement tasks.

For more details, refer to our paper: [LLaSE-G1 Paper](https://arxiv.org/abs/2503.00493)

## Demo

You can listen to the enhancement results on our [Demo Page](https://submission-papers.github.io/LLaSE-G1-demo-page/).

## Installation

Checkpoints are at [huggingface](https://huggingface.co/ASLP-lab/LLaSE-G1).

### 1. Clone the repository

```bash
git clone https://github.com/your-repo/LLaSE-G1.git
cd LLaSE-G1
```

### 2. Create a Conda environment and install dependencies

```bash
conda create -n llase python=3.10
conda activate llase
pip install -r requirements.txt
```

### 3. Download Pretrained Models

LLaSE-G1 requires three additional pre-trained models and checkpoint of the middle LM on Huggingface to function properly. You can download them using the provided shell script:

```bash
bash ./ckpt/download.sh
```

Alternatively, you can download them manually and place them in the `./ckpt/` directory.

## Inference

The main inference script is **`inference.py`**. The inference process consists of two stages:

1. Extract the 6th-layer features from WavLM.
2. Use the language model (LM) to predict speech tokens, and then decode them into audio using **X-Codec2**.

### Running Inference

To run inference, configure the parameters in `./config/test.yml`:

| Parameter        | Description                                                                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `infer_feat_too` | Whether to extract WavLM features during inference.                                                                                                                    |
| `inference_time` | Number of inference iterations.                                                                                                                                        |
| `feat_dir`       | Directory containing extracted features.                                                                                                                               |
| `wav_dir`        | Directory of processed audio files.                                                                                                                                    |
| `task`           | Task type: `SE` (Noise Suppression), `TSE` (Target Speaker Extraction), `PLC` (Packet Loss Concealment), `AEC` (Acoustic Echo Cancellation), `SS` (Speech Separation). |

Command to run inference:

```bash
bash inference.sh
```

## Results

Samples processed by LLaSE-G1 can be found on our [Demo Page](https://submission-papers.github.io/LLaSE-G1-demo-page/).

## Model Checkpoints

Our pretrained model is available on [Hugging Face](https://huggingface.co/ASLP-lab/LLaSE-G1).

## Hints

Our approach focuses on leveraging the LLM's comprehension capabilities to enable autonomous determination of task types, though this may exhibit instability in certain scenarios. A more stable and robust iteration will be released in the upcoming version.

## Citation

If you find this work useful, please cite our paper:

```
@misc{kang2025llaseg1incentivizinggeneralizationcapability,
      title={LLaSE-G1: Incentivizing Generalization Capability for LLaMA-based Speech Enhancement}, 
      author={Boyi Kang and Xinfa Zhu and Zihan Zhang and Zhen Ye and Mingshuai Liu and Ziqian Wang and Yike Zhu and Guobin Ma and Jun Chen and Longshuai Xiao and Chao Weng and Wei Xue and Lei Xie},
      year={2025},
      eprint={2503.00493},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2503.00493}, 
}
```


## Contact

For any questions, please contact: `beaukang02@gmail.com`

