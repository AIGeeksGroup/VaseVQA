# <img src="https://github.com/AIGeeksGroup/VaseVL/blob/main/images/vasevl_logo-cropped.svg" alt="logo" width="30"/> VaseVQA: Multimodal Agent and Benchmark for Ancient Greek Pottery

This is the repository for the paper:
> **VaseVQA: Multimodal Agent and Benchmark for Ancient Greek Pottery**
> 
> Jinchao Ge\*, Tengfei Cheng\*, Biao Wu\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io)\*†, Shiya Huang, Judith Bishop, Gillian Shepherd, [Meng Fang](https://mengfn.github.io/), [Ling Chen](https://profiles.uts.edu.au/Ling.Chen), [Yang Zhao](https://yangyangkiki.github.io/)\**
>
> \*Equal contribution. †Project lead. \**Corresponding author.
> 
> ### [Paper](https://arxiv.org/abs/2509.17191) | [VaseVQA Dataset](https://huggingface.co/datasets/AIGeeksGroup/VaseVQA) | [VaseVL Model](https://huggingface.co/AIGeeksGroup/VaseVL) | [HF Paper](https://huggingface.co/papers/2509.17191)
https://github.com/user-attachments/assets/8f770bfb-d7f9-4e5d-9c10-54ec15f37163


## Introduction


Analyzing cultural-heritage artifacts remains challenging for MLLMs: general models lack domain expertise, and SFT often overfits superficial patterns, yielding brittle reasoning for authentication and historical attribution. This raises the question of how to equip MLLMs with robust, expert-level reasoning for ancient Greek pottery. We present VaseVL, an SFT-then-RL system that turns evaluation into supervision: we construct a taxonomy of question types, probe the SFT model to localize type-specific performance gaps, and optimize with type-conditioned, compositionality-oriented rewards targeting those gaps. We also release VaseVQA, a comprehensive benchmark of 31,773 images designed to probe deep understanding. Experiments show state-of-the-art results on style classification and historical attribution with marked gains in compositional robustness over SFT-only baselines, validating diagnosis-guided, taxonomy-conditioned reward engineering and providing a reusable resource for future research.
<center class ='img'>
<img title="VaseVL Pipeline" src="https://github.com/AIGeeksGroup/VaseVL/blob/main/images/vasevqa_example.png" width="100%">
</center>


## Installation

### Environment Setup

```bash
# 1. Create and activate environment
conda create -n vase python=3.10 -y
conda activate vase

# 2. Install dependencies
pip install torch==2.5.1 torchvision==0.20.1
pip install git+https://github.com/huggingface/transformers
pip install accelerate qwen-vl-utils==0.0.8
pip install pandas tqdm requests validators metrics pycocoevalcap editdistance

# 3. Clone repository
git clone https://github.com/AIGeeksGroup/VaseVQA.git
cd VaseVQA
```

### Training Environments (Optional)
> ⚠️ We recommend creating separate conda environments for each training setup to avoid version conflicts.

#### 1. SFT Environment

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```
Please refer to the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository for debugging specific errors.

#### 2. RL Environment
Run the following commands to configure the RL environment:
```bash
cd RL
bash setup.sh
```

Please refer to the [VLM-R1](https://github.com/om-ai-lab/VLM-R1)  repository for debugging specific errors.

## Dataset

### Download from HuggingFace

```bash
# export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub<1.0.0"
huggingface-cli login

python download_dataset.py
```



## Inference

### 1. Using VaseVL Locally

Download released checkpoints:

```bash
mkdir Models
cd Models
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Qwen2.5-VL-3B-Instruct
# hf download AIGeeksGroup/VaseVL --repo-type model --local-dir VaseVL
cd ..
```

Run inference:

```bash
cd Eval/evaluation
python vqa_inference.py \
    --model-path ../../Models/VaseVL \  # or zero-shot: ../../Models/Qwen2.5-VL-3B-Instruct \
    --batch-size 64
```

---

### 2. Using GPT / Gemini APIs

```bash
cd Eval/evaluation

# OpenAI (default: gpt-4o-mini, limit to first 10 samples for quick testing)
export OPENAI_API_KEY="sk-xxx"
python vqa_inference_gpt.py --limit 10

# Google Gemini (default: gemini-2.0-flash-exp)
export GOOGLE_API_KEY="xxx"
python vqa_inference_gemini.py
```

Outputs are saved in `Eval/evaluation/output/`.


## 📊 Evaluation

Evaluate the existing inference results using the Accuracy and BLEU@1 metrics.
```bash
python vqa_evaluation.py \
    --infer-file output/gpt_results.jsonl # or jsonl inference results from other models
```

### LLM-as-Judge Support
We also support more comprehensive evaluation using *Prometheus-7B-v2.0*.
```bash
python vqa_evaluation_full.py \
    --infer-file output/gpt_results.jsonl # or jsonl inference results from other models
```



## Training

### SFT Training
Example command:
```bash
cd LLaMA-Factory
export CUDA_VISIBLE_DEVICES=1,2,3
export WANDB_API_KEY="......" # export your WANDB key
export WANDB_PROJECT="vase_sft"
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5vl_full_sft.yaml
```
Or use `examples/train_lora/vase_single.yaml` to launch LoRA training.


### RL Training

From-scratch RL training:
```bash
cd RL
bash run_scripts/run_grpo_vase.sh
```

RL continuing from SFT:
```bash
cd RL
bash run_scripts/run_grpo_vase_sft.sh
```


You can update the scripts in the `run_scripts` directory to adapt to your GPU specifications. The script can be run like this:
```bash
export WANDB_DISABLED=False
export WANDB_PROJECT="vase_rl"
export WANDB_API_KEY="......"

CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 \
  src/open_r1/grpo_jsonl.py \
  --model_name_or_path $model_path \
  --data_file_paths $data_paths \
  --image_folders $image_folders \
  --output_dir checkpoints/rl/${EXP_NAME} \
  --num_train_epochs 2 \
  --num_generations 8 \
  --beta 0.04

echo "Training completed for ${EXP_NAME}"
```

---


## 🚀 Deploy VaseVL Demo UI Locally

To get the full experience of the VaseVL UI, you need to deploy it locally by following the steps below:

1. **Clone the VaseVL repository to your local machine.**
   ```bash
   git clone https://github.com/AIGeeksGroup/VaseVL.git
   ```

2. **Navigate to the `ui` directory which contains the front-end source code.**
   ```bash
   cd ui
   ```

3. **Install all required Node.js dependencies.**
   ```bash
   npm install
   ```

4. **Build the UI project for production.**
   ```bash
   npm run build
   ```

5. **Start the local server to launch the VaseVL Demo UI.**
   ```bash
   npm run start
   ```

Once the server starts, you can access the VaseVL Demo UI in your browser at `http://localhost:1717/projects/1743242682314/playground` by default.

<center class ='img'>
<img title="VaseVL Pipeline" src="https://github.com/AIGeeksGroup/VaseVL/blob/main/images/website_example.png" width="100%">
</center>



## Acknowledgements

We acknowledge the use of the following open-source resources in this work:  
[LLaVA](https://github.com/haotian-liu/LLaVA),  [Vicuna](https://github.com/lm-sys/FastChat),  [MiniCPM](https://github.com/OpenBMB/MiniCPM),  [Qwen-2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) as the base vision–language models;  
[VLM-R1](https://github.com/om-ai-lab/VLM-R1) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) as the training frameworks;  
and the museums and institutions that provided the pottery images.


## Citation

If you use any content of this repo in your work, please cite the following paper:

```
@article{ge2025vasevqa,
  title={VaseVQA: Multimodal Agent and Benchmark for Ancient Greek Pottery},
  author={Ge, Jinchao and Cheng, Tengfei and Wu, Biao and Zhang, Zeyu and Huang, Shiya and Bishop, Judith and Shepherd, Gillian and Fang, Meng and Chen, Ling and Zhao, Yang},
  journal={arXiv preprint arXiv:2509.17191},
  year={2025}
}
```


## License

Our data is under NCND license. No commercial use. Do not modify our data for another dataset.

![license](https://github.com/user-attachments/assets/978cf963-0455-44fa-8027-c859af934753)
