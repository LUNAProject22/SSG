import time
import copy

import torch
import numpy as np
np.set_printoptions(precision=3)

from model import InComNetPerson
from feature_extraction import FeatureExtractor
from dataset import AG, cuda_collate_fn
from evaluation import EvaluateInComNet
from config import parse_args

import warnings
warnings.filterwarnings('ignore')


def test(args):

    #### Choose CLIP model
    if args.clip_model == 'ViT_B_32':
        model, preprocess = clip.load('ViT-B/32', args.device)
    elif args.clip_model == 'ViT_L_14_336':
        model, preprocess = clip.load('ViT-L/14@336px', args.device)
    elif args.clip_model == 'ViT_L_14_336_sft':
        model_name = "ViT-L-14-336"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name = model_name, pretrained = args.clip_sft_path)
    model.to(args.device)
    
    AG_dataset_test = AG(mode="test", datasize=args.datasize, data_path=args.data_path, frame_path=args.frame_path, filter_nonperson_box_frame=True,
                        filter_small_box=False if args.mode == 'predcls' else True, preprocess = preprocess)
    dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=4,
                                                collate_fn=cuda_collate_fn, pin_memory=False)

    incomnet_person_model = InComNetPerson() 
    incomnet_person_model.to(args.device)
    num_params = sum(p.numel() for p in incomnet_person_model.parameters())
    # print("Number of parameters in InComNetPerson model : ", num_params)

    evaluation = EvaluateInComNet()
    feat_extractor = FeatureExtractor()

    ckpt = torch.load(args.ckpt_person, map_location=args.device)
    incomnet_person_model.load_state_dict(ckpt['state_dict'], strict=False)
    print('*'*50)
    print('CKPT {} is loaded'.format(args.ckpt_person))

    incomnet_person_model.eval()

    person_sr_pred_list, person_sr_gt_list, person_frame_lengths, person_roles = [], [], [], []

    for data_test in dataloader_test:
                
        im_info = copy.deepcopy(data_test[1])
        gt_annotation = AG_dataset_test.gt_annotations[data_test[4]]
        raw_images_pil_test = copy.deepcopy(data_test[5])

        person_features_test  = feat_extractor.get_person_features(im_info, gt_annotation, raw_images_pil_test, model, preprocess)
        person_sr_features = person_features_test['person_sr_features']
        person_sr_gt = person_features_test['person_sr_gt']
        p_frame_lengths = person_features_test['person_frame_lengths']
        p_roles = person_features_test['p_roles']

        person_frame_lengths.extend(p_frame_lengths)
        person_roles.extend(p_roles)

        person_sr_logits, person_sr_q = incomnet_person_model(person_sr_features)
        person_sr_logits = person_sr_logits.view(-1, 364)
        person_sr_gt = person_sr_gt.view(-1)

        pred_person_sr_labels= torch.argmax(person_sr_logits, dim=1)
        pred_person_sr_labels = pred_person_sr_labels.tolist()
        person_sr_gt = person_sr_gt.tolist()
        person_sr_pred_list.extend(pred_person_sr_labels)
        person_sr_gt_list.extend(person_sr_gt)

    person_sr_result = evaluation.evaluate_person_sr(person_sr_gt_list, person_sr_pred_list, person_frame_lengths, person_roles)

    
    print("\nPerson SR result : ")
    for key, value in person_sr_result.items():
        print(f"{key}: {value}")



if __name__ == "__main__":

    torch.cuda.empty_cache()
    t1 = time.time()

    #### Configurations
    args = parse_args()

    print("############### Configurations ###############")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("##############################################")

    test(args)
