# VaseVL

This is the repository for the paper:
> **VaseVL: Multimodal Agent and Benchmark for Ancient Greek Pottery**
> 
> Jinchao Ge\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io)\*†, Biao Wu\*, Shiya Huang, Judith Bishop, Scott Mann, Gillian Shepherd, Ruicheng Zhang, Xuan Ren, Ling Chen, Meng Fang, Lingqiao Liu, [Yang Zhao](https://yangyangkiki.github.io/)\**
>
> \*Equal contribution. †Project lead. \**Corresponding author.
> 
> ### [Paper]() | [VaseVQA Dataset]() | [Checkpoints]() | [Papers With Code]() | [HF Paper]()
https://github.com/user-attachments/assets/8f770bfb-d7f9-4e5d-9c10-54ec15f37163



## Citation

If you use any content of this repo for your work, please cite the following our paper:
```

```

## Introduction
We present VaseVL, a pioneering Multi-Modal Large Language Model (MLLM) agent for ancient Greek pottery, capable of understanding and analyzing visual and textual data to enhance cultural heritage preservation. To further support the research community, we introduce VaseVQA, a comprehensive Q&A benchmark for evaluating the reasoning and interpretative capabilities of MLLMs on ancient artifacts. The data has 31,773 multi-view vase images. From these, we select 11,693 as single-view images. The benchmark contains vision-language (VL) tasks of visual question answering. VaseVL achieves state-of-the-art performance in stylistic classification and historical attribution, providing critical tools for authentication, forgery detection, and digital archiving. Our final fine-tuning process for the 7B checkpoint uses 9,354 available vase data and finishes in 3~4 hours. Beyond academic contributions, VaseVL fosters global heritage conservation, mitigating cultural erosion and promoting public engagement with ancient Greek artistry.
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

