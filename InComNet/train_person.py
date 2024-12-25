import time
import os
import copy

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
np.set_printoptions(precision=3)
import wandb
import json
import clip
import open_clip

from model import InComNetPerson
from feature_extraction import FeatureExtractor
from dataset import AG, cuda_collate_fn
from evaluation import EvaluateInComNet
from config import parse_args

import warnings
warnings.filterwarnings('ignore')


def train(args):
    torch.manual_seed(args.seed)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    #### Initialize wandb
    if args.wandb:
        run = wandb.init(
            mode='online',
            project="InComNet",
            name="InComNet_person-ViT-L-14-336-sf",
            config={
                "learning_rate": args.lr,
                "epochs": args.nepochs,
            },
        )

    #### Choose CLIP model
    if args.clip_model == 'ViT_B_32':
        model, preprocess = clip.load('ViT-B/32', args.device)
    elif args.clip_model == 'ViT_L_14_336':
        model, preprocess = clip.load('ViT-L/14@336px', args.device)
    elif args.clip_model == 'ViT_L_14_336_sft':
        model_name = "ViT-L-14-336"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name = model_name, pretrained = args.clip_sft_path)
    model.to(args.device)

    AG_dataset_train = AG(mode="train", datasize=args.datasize, data_path=args.data_path, frame_path=args.frame_path, filter_nonperson_box_frame=True,
                        filter_small_box=False if args.mode == 'predcls' else True, preprocess = preprocess)
    dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=False, num_workers=4,
                                                collate_fn=cuda_collate_fn, pin_memory=False)
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

    criterion = nn.CrossEntropyLoss(ignore_index=365,reduction='none')
    optimizer = torch.optim.Adamax(incomnet_person_model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    best_val_acc = -1e8

    for epoch in range(10):
        b = 0 
        tr = []

        person_roles_train = []

        for data in dataloader_train:
            
            incomnet_person_model.train()
            
            im_info = copy.deepcopy(data[1])
            gt_annotation = AG_dataset_train.gt_annotations[data[4]]
            raw_images_pil = copy.deepcopy(data[5])
            
            person_features  = feat_extractor.get_person_features(im_info, gt_annotation, raw_images_pil, model, preprocess)
            person_sr_features = person_features['person_sr_features']
            person_sr_gt = person_features['person_sr_gt']
            person_frame_lengths = person_features['person_frame_lengths']
            p_roles = person_features['p_roles']

            person_roles_train.extend(p_roles)
       
            person_sr_logits, person_sr_q = incomnet_person_model(person_sr_features)
            person_sr_logits = person_sr_logits.view(-1, 364)
            person_sr_gt = person_sr_gt.view(-1)
            person_sr_loss = criterion(person_sr_logits, person_sr_gt)

            loss = torch.mean(person_sr_loss) 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(incomnet_person_model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()  
            
            tr.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']

            if args.wandb:
                run.log({"loss": loss})

            log_dict = {}
            log_dict['epoch'] = epoch
            log_dict['batch'] = b
            log_dict['learning_rate'] = current_lr
            log_dict['loss'] = loss 

            if b % 6 == 0 and b >= 6:
                formatted = (
                            f"epoch : {log_dict['epoch']}  batch : {log_dict['batch']}  "
                            f"learning rate : {log_dict['learning_rate']:.4f}, "
                            f"loss : {log_dict['loss']:.3f}, "
                        )
                print(formatted)
            b += 1

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

        val_acc = person_sr_result['Value'] + person_sr_result['Value-two'] + person_sr_result['value-all'] + person_sr_result['role_based_accuracy'] + person_sr_result['role_based_f1']
        val_acc= sum(person_sr_result.values()) / len(person_sr_result)
        
        print("avg epoch performance : ",  val_acc) 
        print("*" * 100)
        scheduler.step()

        if val_acc > best_val_acc:
            print("val_acc is: ", epoch, val_acc)
            torch.save({"state_dict": incomnet_person_model.state_dict()}, os.path.join(args.save_path, f"incomnet_person_epoch_{epoch}.tar"))
            print("save the checkpoint after {} epochs".format(epoch))
            best_val_acc = val_acc

        with open("log_incomnet-person.txt", "a") as f:
            f.write(f"epoch_{epoch} Person SR: ")
            f.write(json.dumps(person_sr_result) + "\n")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    t1 = time.time()

    #### Configurations
    args = parse_args()

    print("############### Configurations ###############")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("##############################################")

    train(args)






    





    