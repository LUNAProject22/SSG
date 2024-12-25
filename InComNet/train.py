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

from model import InComNet
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
            name="InComNet-ViT-L-14-336-sft",
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

    #### Load InComNet model
    incomnet_model = InComNet() 
    incomnet_model.to(args.device)
    num_params = sum(p.numel() for p in incomnet_model.parameters())
    # print("Number of parameters in InComNet: ", num_params)

    evaluation = EvaluateInComNet()
    feat_extractor = FeatureExtractor()

    criterion = nn.CrossEntropyLoss(ignore_index=365,reduction='none')
    optimizer = torch.optim.Adamax(incomnet_model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    best_val_acc = -1e8
    grand_loss = []

    for epoch in range(args.nepochs):
        b = 0 
        tr = []

        for data in dataloader_train:

            objsr_loss = []
            vrb_loss = []
            vrbsr_loss = []
            start = time.time()
            
            incomnet_model.train()
            
            im_info = copy.deepcopy(data[1])
            gt_annotation = AG_dataset_train.gt_annotations[data[4]]
            raw_images_pil = copy.deepcopy(data[5])
            
            train_features = feat_extractor.get_features(im_info, gt_annotation, raw_images_pil, model, preprocess)
            obj_sr_features = train_features['obj_sr_features']
            obj_sr_mask = train_features['obj_sr_mask']
            obj_sr_gt = train_features['obj_sr_gt']
            verb_img_features = train_features['verb_img_features']
            verb_cls_features = train_features['verb_cls_features']
            verb_gt = train_features['verb_gt']
            verb_sr_features = train_features['verb_sr_features']
            verb_sr_mask = train_features['verb_sr_mask']
            verb_sr_gt = train_features['verb_sr_gt']
            noun_frame_lengths = train_features['noun_frame_lengths']
            n_roles = train_features['n_roles']
            v_frame_lengths = train_features['v_frame_lengths']
            v_roles = train_features['v_roles']

            loss = 0
            obj_sr_loss = 0
            verb_loss = 0
            verb_sr_loss = 0

            for i in range(10):
                if i == 0:
                    # obj SR
                    obj_sr_flag, verb_flag, verb_sr_flag = True, False, False
                    obj_sr_features = obj_sr_features
                    obj_sr_mask = obj_sr_mask
                    obj_sr_gt = obj_sr_gt
                    obj_sr_logits, obj_sr_q = incomnet_model(obj_sr_flag, verb_flag, verb_sr_flag, obj_sr_features, obj_sr_mask, verb=None, verb_mask=None, verb_sr_frame=None, verb_sr_mask = None)
                    obj_sr_logits = obj_sr_logits.view(-1, 364)
                    obj_sr_gt = obj_sr_gt.view(-1)
                    obj_sr_loss = criterion(obj_sr_logits, obj_sr_gt)
                    
                elif i >=1:
                    obj_sr_flag, verb_flag, verb_sr_flag = True, False, False
                    # print("in train: verb_q: obj_sr_features:  ", verb_q.shape, obj_sr_features.shape)
                    obj_sr2_features = torch.cat([verb_q, obj_sr_features], dim=1)
                    obj_sr_logits, obj_sr_q = incomnet_model(obj_sr_flag, verb_flag, verb_sr_flag, obj_sr2_features, obj_sr_mask, verb=None, verb_mask=None, verb_sr_frame=None, verb_sr_mask = None)
                    obj_sr_logits = obj_sr_logits.view(-1, 364)
                    obj_sr_gt = obj_sr_gt.view(-1)
                    obj_sr_loss = criterion(obj_sr_logits, obj_sr_gt) # TODO change loss name
                    
                # verb
                obj_sr_flag, verb_flag, verb_sr_flag = False, True, False
                verb_img_features = verb_img_features
                verb_cls_features = verb_cls_features
                verb_gt = verb_gt
                verb_feat = torch.cat([verb_img_features.unsqueeze(1), verb_cls_features.unsqueeze(1), obj_sr_q], dim=1)
                verb_logits, verb_q = incomnet_model(obj_sr_flag, verb_flag, verb_sr_flag, obj_sr_features, obj_sr_mask, verb_feat, verb_mask=None, verb_sr_frame=None, verb_sr_mask=None)
                verb_logits = verb_logits.squeeze()
                verb_loss = criterion(verb_logits, verb_gt)
                
                # verb SR
                obj_sr_flag, verb_flag, verb_sr_flag = False, False, True
                verb_sr_features = verb_sr_features
                verb_sr_mask = verb_sr_mask
                verb_sr_gt = verb_sr_gt
                verb_sr_feat = torch.cat([verb_q, verb_sr_features], dim=1)
                verb_mask = None
                verb_sr_logits, verb_sr_q = incomnet_model(obj_sr_flag, verb_flag, verb_sr_flag, obj_sr_features, obj_sr_mask, verb_feat, verb_mask, verb_sr_feat, verb_sr_mask)
                verb_sr_logits = verb_sr_logits.view(-1, 125)
                verb_sr_gt = verb_sr_gt.view(-1)
                verb_sr_loss = criterion(verb_sr_logits, verb_sr_gt)

                iter_loss = torch.mean(obj_sr_loss) + torch.mean(verb_loss) + torch.mean(verb_sr_loss)
                loss += iter_loss

                objsr_loss.append(torch.mean(obj_sr_loss).item())
                vrb_loss.append(torch.mean(verb_loss).item())
                vrbsr_loss.append(torch.mean(verb_sr_loss).item())
   
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(incomnet_model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()  
            
            tr.append(loss.item())

            time_per_batch = (time.time() - start)
            current_lr = optimizer.param_groups[0]['lr']

            if args.wandb:
                run.log({"loss": loss, "objsr_loss": np.mean(objsr_loss), "vrb_loss": np.mean(vrb_loss), "vrbsr_loss":np.mean(vrbsr_loss)})

            log_dict = {}
            log_dict['epoch'] = epoch
            log_dict['batch'] = b
            log_dict['learning_rate'] = current_lr
            log_dict['mean_loss'] = loss # np.mean(tr)
            log_dict['obj_SR_loss'] = np.mean(objsr_loss)
            log_dict['verb_loss'] = np.mean(vrb_loss)
            log_dict['verb_SR_loss'] = np.mean(vrbsr_loss)
            
            if b % 6 == 0 and b >= 6:
                formatted = (
                            f"epoch : {log_dict['epoch']}  batch : {log_dict['batch']}  "
                            f"learning rate : {log_dict['learning_rate']:.4f}, "
                            f"loss : {log_dict['mean_loss']:.3f}, "
                            f"obj SR loss : {log_dict['obj_SR_loss']:.3f}, "
                            f"verb loss : {log_dict['verb_loss']:.3f}, "
                            f"verb SR loss : {log_dict['verb_SR_loss']:.3f}"
                        )
                print(formatted)
                
            b += 1
            start = time.time()

        incomnet_model.eval()
        start = time.time()

        obj_sr_pred_list, obj_sr_gt_list, verb_pred_list, verb_gt_list, verb_sr_pred_list, verb_sr_gt_list, obj_frame_lengths, obj_roles, verb_frame_lengths, verb_roles = [], [], [], [], [], [], [], [], [], []

        for data_test in dataloader_test:
                    
            im_info = copy.deepcopy(data_test[1])
            gt_annotation = AG_dataset_test.gt_annotations[data_test[4]]
            raw_images_pil_test = copy.deepcopy(data_test[5])
        
            test_features = feat_extractor.get_features(im_info, gt_annotation, raw_images_pil_test, model, preprocess)
            obj_sr_features = test_features['obj_sr_features']
            obj_sr_mask = test_features['obj_sr_mask']
            obj_sr_gt = test_features['obj_sr_gt']
            verb_img_features = test_features['verb_img_features']
            verb_cls_features = test_features['verb_cls_features']
            verb_gt = test_features['verb_gt']
            verb_sr_features = test_features['verb_sr_features']
            verb_sr_mask = test_features['verb_sr_mask']
            verb_sr_gt = test_features['verb_sr_gt']
            noun_frame_lengths = test_features['noun_frame_lengths']
            n_roles = test_features['n_roles']
            v_frame_lengths = test_features['v_frame_lengths']
            v_roles = test_features['v_roles']

            obj_frame_lengths.extend(noun_frame_lengths)
            obj_roles.extend(n_roles)
            verb_frame_lengths.extend(v_frame_lengths)
            verb_roles.extend(v_roles)

            obj_sr_logits = 0
            verb_sr_logits = 0
            verb_logits = 0

            for i in range(args.iterations):
                if i == 0:
                    # obj SR
                    obj_sr_flag, verb_flag, verb_sr_flag = True, False, False
                    obj_sr_features = obj_sr_features
                    obj_sr_mask = obj_sr_mask
                    obj_sr_logits, obj_sr_q = incomnet_model(obj_sr_flag, verb_flag, verb_sr_flag, obj_sr_features, obj_sr_mask, verb=None, verb_mask=None, verb_sr_frame=None, verb_sr_mask = None)

                elif i >=1:
                    # Iter2 onward
                    # obj SR
                    obj_sr_flag, verb_flag, verb_sr_flag = True, False, False
                    obj_sr2_features = torch.cat([verb_q, obj_sr_features], dim=1)
                    obj_sr_logits, obj_sr_q = incomnet_model(obj_sr_flag, verb_flag, verb_sr_flag, obj_sr2_features, obj_sr_mask, verb=None, verb_mask=None, verb_sr_frame=None, verb_sr_mask = None)

                # verb
                obj_sr_flag, verb_flag, verb_sr_flag = False, True, False
                verb_img_features = verb_img_features
                verb_cls_features = verb_cls_features
                verb_feat = torch.cat([verb_img_features.unsqueeze(1), verb_cls_features.unsqueeze(1), obj_sr_q], dim=1)
                verb_logits, verb_q = incomnet_model(obj_sr_flag, verb_flag, verb_sr_flag, obj_sr_features, obj_sr_mask, verb_feat, verb_mask=None, verb_sr_frame=None, verb_sr_mask=None)
                
                # verb SR
                obj_sr_flag, verb_flag, verb_sr_flag = False, False, True
                verb_sr_features = verb_sr_features
                verb_sr_mask = verb_sr_mask
                verb_sr_feat = torch.cat([verb_q, verb_sr_features], dim=1)
                verb_mask = None
                verb_sr_logits, verb_sr_q = incomnet_model(obj_sr_flag, verb_flag, verb_sr_flag, obj_sr_features, obj_sr_mask, verb_feat, verb_mask, verb_sr_feat, verb_sr_mask)

            obj_sr_gt = obj_sr_gt
            obj_sr_logits = obj_sr_logits.view(-1, 364)
            obj_sr_gt = obj_sr_gt.view(-1)
            pred_obj_sr_labels= torch.argmax(obj_sr_logits, dim=1)
            pred_obj_sr_labels = pred_obj_sr_labels.tolist()
            obj_sr_gt = obj_sr_gt.tolist()
            obj_sr_pred_list.extend(pred_obj_sr_labels)
            obj_sr_gt_list.extend(obj_sr_gt)

            verb_gt = verb_gt
            verb_logits = verb_logits.squeeze()
            pred_verb_labels= torch.argmax(verb_logits, dim=1)
            verb_pred_list.extend(pred_verb_labels.tolist())
            verb_gt_list.extend(verb_gt.tolist())
            
            verb_sr_gt = verb_sr_gt
            verb_sr_logits = verb_sr_logits.view(-1, 125)
            verb_sr_gt = verb_sr_gt.view(-1)
            pred_verb_sr_labels= torch.argmax(verb_sr_logits, dim=1)
            verb_sr_pred_list.extend(pred_verb_sr_labels.tolist())
            verb_sr_gt_list.extend(verb_sr_gt.tolist())

        obj_result, verb_result, verb_sr_result = evaluation.evaluate(obj_sr_gt_list, obj_sr_pred_list, obj_frame_lengths, obj_roles, verb_gt_list, verb_pred_list, verb_sr_gt_list, verb_sr_pred_list, verb_frame_lengths, verb_roles)

        
        print("\nObject SR result:")
        for key, value in obj_result.items():
            print(f"{key}: {value}")
        
        print("\nVerb result:")
        for key, value in verb_result.items():
            print(f"{key}: {value}")
        
        print("\nVerb SR result:") 
        for key, value in verb_sr_result.items():
            print(f"{key}: {value}")

        val_acc = obj_result['Value'] + verb_result['Accuracy'] + verb_sr_result['Value']
        
        print("avg epoch performance : ",  val_acc) 
        print("*" * 100)
        scheduler.step()

        if val_acc > best_val_acc:
            print("val_acc is: ", epoch, val_acc)
            torch.save({"state_dict": incomnet_model.state_dict()}, os.path.join(args.save_path, f"incomnet_epoch_{epoch}.tar"))
            print("save the checkpoint after {} epochs".format(epoch))
            best_val_acc = val_acc

        with open("log_incomnet.txt", "a") as f:
            f.write(f"epoch_{epoch} Object SR: ")
            f.write(json.dumps(obj_result) + "\n")
        with open("log_incomnet.txt", "a") as f:
            f.write(f"epoch_{epoch} Verb predicate: ")
            f.write(json.dumps(verb_result) + "\n")
        with open("log_incomnet.txt", "a") as f:
            f.write(f"epoch_{epoch} Verb predicate SR: ")
            f.write(json.dumps(verb_sr_result) + "\n")
                      

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
