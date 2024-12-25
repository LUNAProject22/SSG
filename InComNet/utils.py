import gc

import torch
import numpy as np
import cv2
import clip
from PIL import Image, ImageDraw

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from config import parse_args

args = parse_args()

def organize_input(im_info, gt_annotation):

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
    affordance = []
    aff_available = []
    
    for i in gt_annotation:
        bbox_num += len(i)
        
    FINAL_BBOXES = torch.zeros([bbox_num,5], dtype=torch.float32).to(args.device)
    FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64).to(args.device)
    FINAL_SCORES = torch.ones([bbox_num], dtype=torch.float32).to(args.device)
    HUMAN_IDX = torch.zeros([len(gt_annotation),1], dtype=torch.int64).to(args.device)
                
    bbox_idx = 0

    for i, j in enumerate(gt_annotation):
        for m in j:
            if 'person_bbox' in m.keys():
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
                affordance.append(m['affordance'])
                aff_available.append(m['aff_available'])
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
        'affordance' : affordance,
        'aff_available' : aff_available   
    }

    return entry

#### Translucent background prompt
def get_visually_prompted_features(raw_images_pil, box_tensor, model, preprocess):

    box_features = []

    for i in range(len(raw_images_pil)): 
        image_pil=raw_images_pil[i] 
        image_pil_copy = image_pil.copy()
        
        for k in range(len(box_tensor)):
            if(i==box_tensor[k][0]):
                box_tensor=box_tensor.detach().cpu()
    
                x_min_u=box_tensor[k][1]
                y_min_u=box_tensor[k][2]
                x_max_u=box_tensor[k][3]
                y_max_u=box_tensor[k][4]
                
                image_pil_copy=image_pil_copy.convert('RGBA')
                
                mask = Image.new('L', image_pil_copy.size, 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle((x_min_u, y_min_u, x_max_u, y_max_u), fill=255)
                
                overlay_color = (255, 105, 180, 128)  # Pink color with transparency
                overlay = Image.new('RGBA', image_pil_copy.size, overlay_color)
                
                # Apply the mask to the overlay image
                overlay.paste(image_pil_copy, (0, 0), mask=mask)
                blended_image = Image.alpha_composite(image_pil_copy, overlay)
    
                prompted_pil_image_u=blended_image.convert("RGB")
                preprocessed_u=preprocess(prompted_pil_image_u)
                preprocessed_u = preprocessed_u.cpu()
                box_features.append(preprocessed_u)
    
        del image_pil_copy
    
    image_input_u = torch.tensor(np.stack(box_features)).cpu()
    
    with torch.no_grad():
        image_features = model.encode_image(image_input_u.to(args.device))
        return image_features
    

#### Translucent background prompt for person bboxes
def get_visually_prompted_features_person(raw_images_pil, box_tensor, model, preprocess):

    box_features = []
    for i in range(len(raw_images_pil)): 
        image_pil=raw_images_pil[i] 
        image_pil_copy = image_pil.copy()
    
        x_min_u=box_tensor[i][0]
        y_min_u=box_tensor[i][1]
        x_max_u=box_tensor[i][2]
        y_max_u=box_tensor[i][3]
        
        image_pil_copy=image_pil_copy.convert('RGBA')
        
        mask = Image.new('L', image_pil_copy.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle((x_min_u, y_min_u, x_max_u, y_max_u), fill=255)
        
        overlay_color = (255, 105, 180, 128)  # Pink color with transparency
        overlay = Image.new('RGBA', image_pil_copy.size, overlay_color)
        overlay.paste(image_pil_copy, (0, 0), mask=mask)
        blended_image = Image.alpha_composite(image_pil_copy, overlay)

        prompted_pil_image_u=blended_image.convert("RGB")
        # prompted_pil_image_u.save("personbox.png")
        
        preprocessed_u=preprocess(prompted_pil_image_u)
        box_features.append(preprocessed_u)

        del image_pil_copy
    
    image_input_u = torch.tensor(np.stack(box_features)).to(args.device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input_u).to(args.device)

        return image_features


# def get_ag_kb(classes):
#     class_names=[]
#     kb=[]
#     for i in range(len(classes)):
#         if classes[i]==1: # background take a print and see the indexes
#             continue
#         elif classes[i]==2:
#             class_names.append('bag')
#             kb.append(['holding', 'have it on the back', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==3:
#             class_names.append('bed')
#             kb.append(['leaning on', 'holding', 'lying on', 'sitting on', 'touching', 'not contacting'])
#         elif classes[i]==4:
#             class_names.append('blanket')
#             kb.append(['holding', 'lying on', 'covered by', 'sitting on', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==5:
#             class_names.append('book')
#             kb.append(['holding', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==6:
#             class_names.append('box')
#             kb.append(['holding', 'sitting on', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==7:
#             class_names.append('broom')
#             kb.append(['holding', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==8:
#             class_names.append('chair')
#             kb.append(['leaning on', 'holding', 'lying on', 'sitting on', 'standing on', 'touching', 'not contacting'])
#         elif classes[i]==9:
#             class_names.append('closet')
#             kb.append(['leaning on', 'holding', 'touching', 'not contacting'])
#         elif classes[i]==10:
#             class_names.append('clothes')
#             kb.append(['holding', 'wearing', 'covered by', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==11:
#             class_names.append('cup')
#             kb.append(['wiping', 'holding', 'drinking from', 'touching', 'not contacting'])
#         elif classes[i]==12:
#             class_names.append('dish')
#             kb.append(['wiping', 'holding', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==13:
#             class_names.append('door')
#             kb.append(['leaning on', 'holding', 'touching', 'not contacting'])
#         elif classes[i]==14:
#             class_names.append('doorknob')
#             kb.append(['holding', 'twisting', 'touching', 'not contacting'])
#         elif classes[i]==15:
#             class_names.append('doorway')
#             kb.append(['leaning on', 'holding', 'lying on', 'sitting on', 'standing on', 'touching', 'not contacting'])
#         elif classes[i]==16:
#             class_names.append('floor')
#             kb.append(['lying on', 'sitting on', 'standing on', 'touching', 'not contacting'])
#         elif classes[i]==17:
#             class_names.append('food')
#             kb.append(['holding', 'eating', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==18:
#             class_names.append('groceries')
#             kb.append(['holding', 'eating', 'standing on', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==19:
#             class_names.append('laptop')
#             kb.append(['holding', 'carrying', 'touching', 'looking at', 'not contacting'])
#         elif classes[i]==20:
#             class_names.append('light')
#             kb.append(['holding', 'touching', 'not contacting'])
#         elif classes[i]==21:
#             class_names.append('medicine')
#             kb.append(['holding', 'eating', 'twisting', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==22:
#             class_names.append('mirror')
#             kb.append(['wiping', 'holding', 'carrying', 'touching', 'looking at', 'not contacting'])
#         elif classes[i]==23:
#             class_names.append('paper')
#             kb.append(['holding', 'writing on', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==24:
#             class_names.append('phone')
#             kb.append(['holding', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==25:
#             class_names.append('picture')
#             kb.append(['holding', 'touching', 'looking at', 'not contacting'])
#         elif classes[i]==26:
#             class_names.append('pillow')
#             kb.append(['leaning on', 'holding', 'lying on', 'sitting on', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==27:
#             class_names.append('refrigerator')
#             kb.append(['holding', 'touching', 'not contacting'])
#         elif classes[i]==28:
#             class_names.append('sandwich')
#             kb.append(['holding', 'eating', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==29:
#             class_names.append('shelf')
#             kb.append(['leaning on', 'holding', 'touching', 'not contacting'])
#         elif classes[i]==30:
#             class_names.append('shoe')
#             kb.append(['holding', 'wearing', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==31:
#             class_names.append('sofa')
#             kb.append(['holding', 'lying on', 'sitting on', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==32:
#             class_names.append('table')
#             kb.append(['wiping', 'holding', 'sitting on', 'touching', 'not contacting'])
#         elif classes[i]==33:
#             class_names.append('television')
#             kb.append(['holding', 'touching', 'looking at', 'not contacting'])
#         elif classes[i]==34:
#             class_names.append('towel')
#             kb.append(['holding', 'covered by', 'twisting', 'carrying', 'touching', 'not contacting'])
#         elif classes[i]==35:
#             class_names.append('vacuum')
#             kb.append(['holding', 'touching', 'not contacting'])
#         else:    
#             class_names.append('window')
#             kb.append(['wiping', 'leaning on', 'holding', 'touching', 'not contacting'])

#     return class_names, kb

# def compute_similarity(relationship_embeddings, affordance_embeddings):
#     result = []
#     for rel, aff in zip(relationship_embeddings, affordance_embeddings):
#         max_similarity = -1
#         best_match = None
        
#         for rel_embedding in rel:
#             if (aff.shape == torch.Size([512])):
#                 aff = aff.unsqueeze(0)
#             if (rel_embedding.shape == torch.Size([512])):
#                 rel_embedding = rel_embedding.unsqueeze(0)
#             similarities = torch.nn.functional.cosine_similarity(aff, rel_embedding)
#             similarity = similarities.item()
                
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 best_match = rel_embedding
            
#         result.append(best_match)
            
#         return result

def remove_elements_by_indexes(input_list, indexes_to_remove):
    indexes_to_remove.sort(reverse=True)  # Sort indexes in descending order
        
    for index in indexes_to_remove:
        if 0 <= index < len(input_list):
            input_list.pop(index)
        
    return input_list

# # value metric
# def value_metric(predicted_noun_lists, ground_truth_noun_lists, total):
#     total_pairs = len(predicted_noun_lists)
#     correct_pairs = 0.0
        
#     for predicted_nouns, ground_truth_nouns in zip(predicted_noun_lists, ground_truth_noun_lists):
#         if any(predicted_noun == ground_truth_noun for predicted_noun, ground_truth_noun in zip(predicted_nouns, ground_truth_nouns)):
#             correct_pairs += 1

#     value = (correct_pairs / total) * 100 
#     print("correct_pairs, total_pairs :", correct_pairs, total_pairs,  total)
#     print("correct_pairs, total_pairs wo uneqaul:", correct_pairs, len(predicted_noun_lists))
#     return value

# # value all metric
# def value_all_metric(predicted_noun_lists, ground_truth_noun_lists, total):
#     total_pairs = len(predicted_noun_lists)
#     correct_pairs = 0.0
#     matched_indices = []
    
#     for idx, (predicted_nouns, ground_truth_nouns) in enumerate(zip(predicted_noun_lists, ground_truth_noun_lists)):
#         if all(predicted_noun == ground_truth_noun for predicted_noun, ground_truth_noun in zip(predicted_nouns, ground_truth_nouns)):
#             correct_pairs += 1
#             matched_indices.append(idx)
#             print(idx, ground_truth_nouns)
    
#     # value_all = (correct_pairs / total_pairs) * 100
#     value = (correct_pairs / total) * 100 
#     print("correct_pairs, total_pairs :", correct_pairs, total_pairs,  total)
#     print("correct_pairs, total_pairs wo unequal:", correct_pairs, len(predicted_noun_lists))
#     return value

# # value-at-least two
# def  value_metric_at_least_two(predicted_noun_lists, ground_truth_noun_lists, total):
#     total_verbs = len(predicted_noun_lists)
#     verbs_with_at_least_two_correct = 0.0
        
#     for predicted_nouns, ground_truth_nouns in zip(predicted_noun_lists, ground_truth_noun_lists):
#         num_correct_pairs = sum(predicted_noun == ground_truth_noun for predicted_noun, ground_truth_noun in zip(predicted_nouns, ground_truth_nouns))
            
#         if num_correct_pairs >= 2:
#             verbs_with_at_least_two_correct += 1
    
#         # value = (verbs_with_at_least_two_correct / total_verbs) * 100
#     value = (verbs_with_at_least_two_correct / total) * 100 
#     print("correct_pairs, total_pairs :", verbs_with_at_least_two_correct, total_verbs, total)
#     print("correct_pairs, total_pairs wo uneqaul:", verbs_with_at_least_two_correct, len(predicted_noun_lists))
#     return value

# def role_based_metrics(pred, gt, roles):
    
#     pred = [[int(x) for x in sublist] for sublist in pred]
#     gt = [[int(x) for x in sublist] for sublist in gt]
    
#     pred = [item for sublist in pred for item in sublist]
#     gt = [item for sublist in gt for item in sublist]
    
#     print(len(gt), len(pred), len(roles))
    
#     # Find unique classes
#     unique_roles = set(roles)
    
#     # Initialize lists to store metrics for each role class
#     accuracies = []
#     macro_precisions = []
#     macro_recalls = []
#     macro_f1_scores = []
    
#     # Iterate over unique role classes
#     for role in unique_roles:
#         # Get indices of instances belonging to current role class
#         role_indices = [i for i, r in enumerate(roles) if r == role]
#         # print(role, role_indices)
        
#         # Extract ground truth and predicted values for current role class
#         role_gt = [gt[i] for i in role_indices]
#         role_pred = [pred[i] for i in role_indices]
    
#         # Calculate accuracy
#         accuracy = accuracy_score(role_gt, role_pred)
#         accuracies.append(accuracy*100)
    
#         # Calculate macro precision
#         macro_precision = precision_score(role_gt, role_pred, average='macro')
#         macro_precisions.append(macro_precision*100)
    
#         # Calculate macro recall
#         macro_recall = recall_score(role_gt, role_pred, average='macro')
#         macro_recalls.append(macro_recall*100)
    
#         # Calculate macro F1-score
#         macro_f1_score = f1_score(role_gt, role_pred, average='macro')
#         macro_f1_scores.append(macro_f1_score*100)
    
#     # # Print metrics for each role class
#     # for i, role in enumerate(unique_roles):
#     #     print(f"\nMetrics for Role: {role}")
#     #     print(f"Accuracy: {accuracies[i]}")
#     #     print(f"Macro Precision: {macro_precisions[i]}")
#     #     print(f"Macro Recall: {macro_recalls[i]}")
#     #     print(f"Macro F1-score: {macro_f1_scores[i]}")
    
#     # Calculate average metrics
#     avg_accuracy = np.mean(accuracies)
#     avg_macro_precision = np.mean(macro_precisions)
#     avg_macro_recall = np.mean(macro_recalls)
#     avg_macro_f1_score = np.mean(macro_f1_scores)
    
#     # Print average metrics
#     print("\n----Average Metrics----:")
#     print(f"Average Accuracy: {avg_accuracy}")
#     print(f"Average Macro Precision: {avg_macro_precision}")
#     print(f"Average Macro Recall: {avg_macro_recall}")
#     print(f"Average Macro F1-score: {avg_macro_f1_score}")
#     print("\n")

# def role_based_metrics_top1_setting(splitted_gt, splitted_pred, roles, incorrect_pred_index):
#     # 1 - replace splitted_gt's (after remove empty sublists if any-should not be available for SSG) role values with 5000 (value that does not exist to penalize them if their top-1 predicted verb is icnorrrect)
#         gt_list = splitted_gt
#         for idx in incorrect_pred_index:
#             if idx < len(gt_list):
#                 gt_list[idx] = [5000 for _ in gt_list[idx]]
        
#         # print("incorrect_pred_index: ", incorrect_pred_index)
#         # print("after replace with incorrect_pred_index: ", gt_list)

#         # 2: Obtain gt, pred, role names (use the sr code to to obtain roles)
#         pred = splitted_pred
#         gt = gt_list
#         roles = roles
        
        
#         pred = [[int(x) for x in sublist] for sublist in pred]
#         gt = [[int(x) for x in sublist] for sublist in gt]
        
#         pred = [item for sublist in pred for item in sublist]
#         gt = [item for sublist in gt for item in sublist]
        
#         print("count 5000 :", gt.count(5000))
        
#         print(len(gt), len(pred), len(roles))
        
#         # Find unique classes
#         unique_roles = set(roles)
        
#         # Initialize lists to store metrics for each role class
#         accuracies = []
#         macro_precisions = []
#         macro_recalls = []
#         macro_f1_scores = []
        
#         # Iterate over unique role classes
#         for role in unique_roles:
#             # Get indices of instances belonging to current role class
#             role_indices = [i for i, r in enumerate(roles) if r == role]
#             # print(role, role_indices)
            
#             # Extract ground truth and predicted values for current role class
#             role_gt = [gt[i] for i in role_indices]
#             role_pred = [pred[i] for i in role_indices]
        
#             # Calculate accuracy
#             accuracy = accuracy_score(role_gt, role_pred)
#             accuracies.append(accuracy*100)
        
#             # Calculate macro precision
#             macro_precision = precision_score(role_gt, role_pred, average='macro')
#             macro_precisions.append(macro_precision*100)
        
#             # Calculate macro recall
#             macro_recall = recall_score(role_gt, role_pred, average='macro')
#             macro_recalls.append(macro_recall*100)
        
#             # Calculate macro F1-score
#             macro_f1_score = f1_score(role_gt, role_pred, average='macro')
#             macro_f1_scores.append(macro_f1_score*100)
        
#         # # Print metrics for each role class
#         # for i, role in enumerate(unique_roles):
#         #     print(f"\nMetrics for Role: {role}")
#         #     print(f"Accuracy: {accuracies[i]}")
#         #     print(f"Macro Precision: {macro_precisions[i]}")
#         #     print(f"Macro Recall: {macro_recalls[i]}")
#         #     print(f"Macro F1-score: {macro_f1_scores[i]}")
        
#         # Calculate average metrics
#         avg_accuracy = np.mean(accuracies)
#         avg_macro_precision = np.mean(macro_precisions)
#         avg_macro_recall = np.mean(macro_recalls)
#         avg_macro_f1_score = np.mean(macro_f1_scores)
        
#         # Print average metrics
#         print("\nAverage Metrics:")
#         print(f"Average Accuracy: {avg_accuracy}")
#         print(f"Average Macro Recall: {avg_macro_recall}")
#         print(f"Average Macro Precision: {avg_macro_precision}")
#         print(f"Average Macro F1-score: {avg_macro_f1_score}")


def split_list(lst, sublist_lengths):
    result = []
    index = 0

    for length in sublist_lengths:
        result.append(lst[index:index + length])
        index += length

    return result

def remove_elements_by_indexes(input_list, indexes_to_remove):
    indexes_to_remove.sort(reverse=True)  # Sort indexes in descending order
    
    for index in indexes_to_remove:
        if 0 <= index < len(input_list):
            input_list.pop(index)
    
    return input_list


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    # im = im[:, :, ::-1]
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    # im = imresize(im, im_scale)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale























    

    