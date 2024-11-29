# Situational Scene Graph for Structured Human-centric Situation Understanding

This repository contains the official PyTorch implementation of the paper:

**Situational Scene Graph for Structured Human-centric Situation Understanding**  
Chinthani Sugandhika, Chen Li, Deepu Rajan, Basura Fernando 
WACV 2025 

[![arXiv](https://img.shields.io/badge/arXiv-2307.00586-b31b1b.svg)](https://arxiv.org/abs/2410.22829)


## Abstract
Graph based representation has been widely used in modelling spatio-temporal relationships in video understanding. Although effective, existing graph-based approaches focus on capturing the human-object relationships while ignoring fine-grained semantic properties of the action components. These semantic properties are crucial for understanding the current situation, such as where does the action takes place, what tools are used and functional properties of the objects. In this work, we propose a graph-based representation called Situational Scene Graph (SSG) to encode both human-object relationships and the corresponding semantic properties. The semantic details are represented as predefined roles and values inspired by situation frame, which is originally designed to represent a single action. Based on our proposed representation, we introduce the task of situational scene graph generation and propose a multi-stage pipeline Interactive and Complementary Network (InComNet) to address the task. Given that the existing datasets are not applicable to the task, we further introduce a SSG dataset whose annotations consist of semantic role-value frames for human, objects and verb predicates of human-object relations. Finally, we demonstrate the effectiveness of our proposed SSG representation by testing on different downstream tasks. Experimental results show that the unified representation can not only benefit predicate classification and semantic role-value classification, but also benefit reasoning tasks on human-centric situation understanding.

## Situational Scene Graph
![Situational Scene Graph](image.png)

We split our repository into three sections:
1. SSG Dataset
2. InComNet Model
3. Data Annotation Tool


## Citation
If you use this code for your research, please cite our paper:
```bibtext
@article{sugandhika2024situational,
  title={Situational Scene Graph for Structured Human-centric Situation Understanding},
  author={Sugandhika, Chinthani and Li, Chen and Rajan, Deepu and Fernando, Basura},
  journal={arXiv preprint arXiv:2410.22829},
  year={2024}
}

```


## Acknowledgments
Our code is inspired by [STTran](https://github.com/yrcong/STTran).

This research/project is supported by the National Research Foundation, Singapore, under its NRF Fellowship (Award# NRF-NRFF14-2022-0001) and funding allocation to B.F. by A*STAR under its SERC Central Research Fund (CRF).