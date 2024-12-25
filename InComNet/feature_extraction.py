import os
import gc

import numpy as np
np.set_printoptions(precision=4)
import torch
import torch.nn.functional as F
import clip
import open_clip

from utils import get_visually_prompted_features, get_visually_prompted_features_person
from config import parse_args

args = parse_args()

class FeatureExtractor: 

    def organize_input(self, im_info, gt_annotation):

        bbox_num = 0
        im_idx = [] 
        pair = []
        a_rel = []
        s_rel = []
        c_rel = []
        rel_sr = [] # rel sem roles
        noun_sr = [] # obj sem roles
        noun_sr_values = []
        rel_sr_values = []
        nouns=[]
        relationships = []
        person_roles = []
        person_role_values = []
        video_frames = []
        person_box = []
        
        for i in gt_annotation:
            bbox_num += len(i)
            
        FINAL_BBOXES = torch.zeros([bbox_num,5], dtype=torch.float32).to(args.device)
        FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64).to(args.device)
        FINAL_SCORES = torch.ones([bbox_num], dtype=torch.float32).to(args.device)
        HUMAN_IDX = torch.zeros([len(gt_annotation),1], dtype=torch.int64).to(args.device)
                    
        bbox_idx = 0

        for i, j in enumerate(gt_annotation):
            for m in j:
                if 'person_roles' in m.keys():
                    person_roles.append(m['person_roles'])
                elif 'person_role_values' in m.keys():
                    person_role_values.append(m['person_role_values'])
                elif 'person_bbox' in m.keys():
                    person_box.append(m['person_bbox'])
                    FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['person_bbox'][0])
                    FINAL_BBOXES[bbox_idx, 0] = i
                    FINAL_LABELS[bbox_idx] = 1
                    HUMAN_IDX[i] = bbox_idx
                    bbox_idx += 1
                else:
                    FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['bbox'])
                    FINAL_BBOXES[bbox_idx, 0] = i
                    FINAL_LABELS[bbox_idx] = m['class']
                    im_idx.append(i)
                    pair.append([int(HUMAN_IDX[i]), bbox_idx])
                    a_rel.append(m['attention_relationship'].tolist())
                    s_rel.append(m['spatial_relationship'].tolist())
                    c_rel.append(m['contacting_relationship'].tolist())
                    noun_sr.append(m['noun_roles'])
                    noun_sr_values.append(m['noun_role_values'])
                    nouns.append(m['nouns'])
                    rel_sr.append(m['relation_roles'])
                    rel_sr_values.append(m['relation_role_values'])
                    relationships.append(m['relationships'])
                    # affordance.append(m['affordance'])
                    # aff_available.append(m['aff_available'])
                    bbox_idx += 1

        pair = torch.tensor(pair).to(args.device)
        im_idx = torch.tensor(im_idx, dtype=torch.float).to(args.device)

        # if self.mode == 'predcls':
        FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
        
        entry = {
            'boxes': FINAL_BBOXES,
            'labels': FINAL_LABELS,
            'scores': FINAL_SCORES,
            'im_idx': im_idx,
            'pair_idx': pair,
            'human_idx': HUMAN_IDX,
            'attention_gt': a_rel,
            'spatial_gt': s_rel,
            'contacting_gt': c_rel,
            'noun_sr': noun_sr,
            'noun_sr_values': noun_sr_values,
            'relation_sr':rel_sr,
            'relation_sr_values':rel_sr_values,
            'relationships' : relationships,
            'nouns' : nouns,
            'person_roles' : person_roles,
            'person_role_values' : person_role_values,
            # 'video_frames' : video_frames, 
            'person_box':person_box   
        }

        return entry   

    def get_features(self, im_info, gt_annotation, raw_images_pil, model, preprocess): 

        entry = self.organize_input(im_info, gt_annotation)

        max_dim_0 = 7 
        gt_padding_value = 365

        #### Create object SR features
        noun_frame_lengths = []
        n_roles = []
        v_frame_lengths = []
        v_roles = []
        obj_sr_mask = []
        verb_sr_mask = []

        class_names = entry['nouns']
        noun_roles = entry['noun_sr']
        noun_role_values = entry['noun_sr_values']
    
        noun_frame_lengths.extend([len(sublist) for sublist in noun_role_values])
        n_roles.extend(frame for frame in noun_roles)

        # Object SR mask
        frame_lengths = [len(sublist) for sublist in noun_role_values]
        obj_sr_mask = [[1] * (length) + [0] * (max_dim_0 - (length)) for length in frame_lengths] # +2 = img_feat, obj name
        obj_sr_mask = [torch.tensor(sublist) for sublist in obj_sr_mask]
        obj_sr_mask = torch.stack(obj_sr_mask, dim=0).cpu() 

        # Visually prompted image features for object sr
        boxes=entry['boxes']
        pair_idx = entry['pair_idx']
        im_idx = entry['im_idx']
        obj_box_tensor=torch.cat((im_idx[:, None], boxes[:, 1:5][pair_idx[:, 1]]), 1) # object features   
        obj_box_tensor=torch.floor(obj_box_tensor)
        obj_box_tensor=obj_box_tensor.type(torch.int64)
        
        image_features = get_visually_prompted_features(raw_images_pil, obj_box_tensor, model, preprocess)        
        
        with torch.no_grad():
            # features for object class names 
            class_input = clip.tokenize(class_names).to(args.device)
            class_features = model.encode_text(class_input)

            # features for object roles
            obj_role_features = []
            for item in noun_roles:
                role_input = clip.tokenize(item).to(args.device)
                role_inputs = model.encode_text(role_input)
                obj_role_features.append(role_inputs)

        obj_role_features = [F.pad(tensor, (0, 0, 0, max_dim_0 - tensor.shape[0])) for tensor in obj_role_features]

        # Object SR features
        obj_sr_features = []
        for img, cls, obj_roles in zip(image_features, class_features,  obj_role_features):
            obj_roles1 = list(torch.unbind(obj_roles, dim=0))
            obj_roles1.insert(0, img)
            obj_roles1.insert(1, cls)
            feat = torch.stack(obj_roles1)
            obj_sr_features.append(feat)

        obj_sr_features = torch.stack(obj_sr_features).cpu().to(torch.float32)

        # Object sr gt
        obj_sr_gt = entry['noun_sr_values']
        obj_sr_gt = [sublist + [gt_padding_value] * (7 - len(sublist)) if len(sublist) < 7 else sublist[:7] for sublist in obj_sr_gt]
        obj_sr_gt = [torch.tensor(sublist) for sublist in obj_sr_gt]
        obj_sr_gt = torch.stack(obj_sr_gt)
            
        #### Create verb predicate features
        # Visuualy prompted verb predicate features
        boxes=entry['boxes']
        pair_idx = entry['pair_idx']
        im_idx = entry['im_idx']
        union_box_tensor=torch.cat((im_idx[:, None], torch.min(boxes[:, 1:3][pair_idx[:, 0]], boxes[:, 1:3][pair_idx[:, 1]]),
                                         torch.max(boxes[:, 3:5][pair_idx[:, 0]], boxes[:, 3:5][pair_idx[:, 1]])), 1)

        union_box_tensor=torch.floor(union_box_tensor)
        union_box_tensor=union_box_tensor.type(torch.int64)

        verb_image_features = get_visually_prompted_features(raw_images_pil, union_box_tensor, model, preprocess)
        
        verb_img_features = torch.unbind(verb_image_features, dim=0)
        verb_cls_features = torch.unbind(class_features, dim=0)

        verb_img_features = torch.stack(verb_img_features, dim=0).cpu().to(torch.float32)
        verb_cls_features = torch.stack(verb_cls_features, dim=0).cpu().to(torch.float32)

        # Verb predicate gt
        verb_gt = entry['contacting_gt']
        verb_gt = [sublist[0] for sublist in verb_gt] # one verb per object
        verb_gt = torch.tensor(verb_gt, dtype=torch.long)

        #### Create verb predicate SR features
        r_roles = entry['relation_sr']
        rel_role_values = entry['relation_sr_values']
        
        v_frame_lengths.extend([len(sublist[0]) for sublist in rel_role_values]) # takes only one verb SR per obj when >1 verb sr for same p-o pair
        v_roles.extend(frame[0] for frame in r_roles)

        # verb predicate SR mask
        verb_sr_frame_lengths = [len(sublist[0]) for sublist in rel_role_values]
        verb_sr_mask = [[1] * (length) + [0] * (max_dim_0 - (length)) for length in verb_sr_frame_lengths] # +2 = img_feat, obj name
        verb_sr_mask = [torch.tensor(sublist) for sublist in verb_sr_mask]
        verb_sr_mask = torch.stack(verb_sr_mask, dim=0).cpu() 

        # Create visually prompted verb predicate SR features
        boxes=entry['boxes']
        pair_idx = entry['pair_idx']
        im_idx = entry['im_idx']
        union_box_tensor=torch.cat((im_idx[:, None], torch.min(boxes[:, 1:3][pair_idx[:, 0]], boxes[:, 1:3][pair_idx[:, 1]]),
                                         torch.max(boxes[:, 3:5][pair_idx[:, 0]], boxes[:, 3:5][pair_idx[:, 1]])), 1)

        union_box_tensor=torch.floor(union_box_tensor)
        union_box_tensor=union_box_tensor.type(torch.int64)

        verb_sr_image_features = get_visually_prompted_features(raw_images_pil, union_box_tensor, model, preprocess)

        with torch.no_grad():
            # verb roles
            rel_role_features = []
            for frames in r_roles:
                role_input = clip.tokenize(frames[0]).to(args.device)
                role_inputs = model.encode_text(role_input)
                rel_role_features.append(role_inputs)

        rel_role_features = [F.pad(tensor, (0, 0, 0, max_dim_0 - tensor.shape[0])) for tensor in rel_role_features]
        verb_sr_features = []
        for img, cls, verb_roles in zip(verb_sr_image_features, class_features,  rel_role_features):
            verb_roles1 = list(torch.unbind(verb_roles, dim=0))
            verb_roles1.insert(0, img)
            verb_roles1.insert(1, cls)
            feat = torch.stack(verb_roles1)
            verb_sr_features.append(feat)

        # Verb SR features
        verb_sr_features = torch.stack(verb_sr_features).cpu().to(torch.float32)

        # Verb predicate SR gt
        verb_sr_gt = rel_role_values
        verb_sr_gt = [sublist[0] for sublist in verb_sr_gt]
        verb_sr_gt = [sublist + [gt_padding_value] * (7 - len(sublist)) if len(sublist) < 7 else sublist[:7] for sublist in verb_sr_gt]
        verb_sr_gt = [torch.tensor(sublist) for sublist in verb_sr_gt]
        verb_sr_gt = torch.stack(verb_sr_gt)

        features = {}
        features['obj_sr_features'] = obj_sr_features.to(args.device)
        features['obj_sr_mask'] = obj_sr_mask.to(args.device)
        features['obj_sr_gt'] = obj_sr_gt.to(args.device)
        features['verb_img_features'] = verb_img_features.to(args.device)
        features['verb_cls_features'] = verb_cls_features.to(args.device)
        features['verb_gt'] = verb_gt.to(args.device)
        features['verb_sr_features'] = verb_sr_features.to(args.device)
        features['verb_sr_mask'] = verb_sr_mask.to(args.device)
        features['verb_sr_gt'] = verb_sr_gt.to(args.device)
        features['noun_frame_lengths'] = noun_frame_lengths
        features['n_roles'] = n_roles
        features['v_frame_lengths'] = v_frame_lengths
        features['v_roles'] = v_roles

        return features               


    def get_person_features(self, im_info, gt_annotation, raw_images_pil, model, preprocess): 

        entry = self.organize_input(im_info, gt_annotation)

        person_roles = entry['person_roles']
        person_role_values = entry['person_role_values']

        person_cls = ["person" for _ in range(len(person_role_values))]
        
        person_frame_lengths = []
        p_roles = []
    
        person_frame_lengths.extend([len(sublist) for sublist in person_role_values])
        p_roles.extend(frame for frame in person_roles)

        # person sr gt
        person_sr_gt = [torch.tensor(sublist) for sublist in person_role_values]
        person_sr_gt = torch.stack(person_sr_gt)
        
        person_box_tensor = torch.tensor(entry['person_box'])
        person_box_tensor = person_box_tensor.squeeze()
          
        image_features = get_visually_prompted_features_person(raw_images_pil, person_box_tensor,model, preprocess)   
        
        with torch.no_grad():
            class_input = clip.tokenize(person_cls).to(args.device)
            class_features = model.encode_text(class_input)

            person_role_features = []
            for item in person_roles:
                role_input = clip.tokenize(item).to(args.device)
                role_inputs = model.encode_text(role_input)
                person_role_features.append(role_inputs)

        person_sr_features = []
        for img, cls, person_roles in zip(image_features, class_features,  person_role_features):
            person_roles = list(torch.unbind(person_roles, dim=0))
            person_roles.insert(0, img)
            person_roles.insert(1, cls)
            feat = torch.stack(person_roles)
            person_sr_features.append(feat)

        person_sr_features = torch.stack(person_sr_features).to(torch.float32)

        person_features = {}
        person_features['person_sr_features'] = person_sr_features.to(args.device)
        person_features['person_sr_gt'] = person_sr_gt.to(args.device)
        person_features['person_frame_lengths'] = person_frame_lengths
        person_features['p_roles'] = p_roles
 
        return person_features   