# InComNet

## Setup
Run the following commands to install necessary dependencies.

```bash
  conda create -n incomnet python=3.8.2
  conda activate incomnet
  pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
  pip install git+https://github.com/openai/CLIP.git
  pip install open-clip-torch-any-py3
  pip install -r requirements.txt
```

## Data
We use the SSG dataset to train/evaluate our method. SSG is based on Action Genome dataset.
Please process and downloaded Action Genome video frames with the [Toolkit](https://github.com/JingweiJ/ActionGenome) into `data/frames` folder. 
Download the SSG annotations from https://drive.google.com/drive/folders/1cUNQpBM5TBftALfcfA67AgtM7HvxYKNd?usp=drive_link and put them into the `data` folder.

## Fine-tuned CLIP ViT-L-14-336 model on SSG dataset
We provide the fine-tuned CLIP ViT-L-14-336 model on SSG dataset at https://drive.google.com/drive/folders/1eXOJ-HVPlBAc-8bM6_FwcFQ5bO2u2lbh?usp=sharing
Download and put this model into `pre_trained_models` folder. 

## Training
Follow the scripts below to train the InComNet model and InComNet person model.
+ For InComNet model
```
python train    # for training
python test.py  # for evaluating
```
+ For InComNet person model
```
python person_train    # for training
python person_test.py  # for evaluating
```


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
We followed the scripts given in [CLIP finetuning](https://github.com/mlfoundations/open_clip/discussions/812) for finetuning the CLIP model on SSG dataset.
