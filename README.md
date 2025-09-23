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



## Citation

If you use any content of this repo for your work, please cite the following our paper:
```
@article{ge2025vasevqa,
  title={VaseVQA: Multimodal Agent and Benchmark for Ancient Greek Pottery},
  author={Ge, Jinchao and Cheng, Tengfei and Wu, Biao and Zhang, Zeyu and Huang, Shiya and Bishop, Judith and Shepherd, Gillian and Fang, Meng and Chen, Ling and Zhao, Yang},
  journal={arXiv preprint arXiv:2509.17191},
  year={2025}
}
```

## Introduction
Analyzing cultural-heritage artifacts remains challenging for MLLMs: general models lack domain expertise, and SFT often overfits superficial patterns, yielding brittle reasoning for authentication and historical attribution. This raises the question of how to equip MLLMs with robust, expert-level reasoning for ancient Greek pottery. We present VaseVL, an SFT-then-RL system that turns evaluation into supervision: we construct a taxonomy of question types, probe the SFT model to localize type-specific performance gaps, and optimize with type-conditioned, compositionality-oriented rewards targeting those gaps. We also release VaseVQA, a comprehensive benchmark of 31,773 images designed to probe deep understanding. Experiments show state-of-the-art results on style classification and historical attribution with marked gains in compositional robustness over SFT-only baselines, validating diagnosis-guided, taxonomy-conditioned reward engineering and providing a reusable resource for future research.
<center class ='img'>
<img title="VaseVL Pipeline" src="https://github.com/AIGeeksGroup/VaseVL/blob/main/images/vasevqa_example.png" width="100%">
</center>


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


## License

Our data is under NCND license. no commerical use. Do not modify our data for another dataset.

![license](https://github.com/user-attachments/assets/978cf963-0455-44fa-8027-c859af934753)
