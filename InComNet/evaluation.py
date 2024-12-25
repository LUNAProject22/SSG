import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from utils import split_list, remove_elements_by_indexes
from config import parse_args

args = parse_args()


def value_metric(predicted_noun_lists, ground_truth_noun_lists, total):

    correct_pairs = 0.0
        
    for idx, (predicted_nouns, ground_truth_nouns) in enumerate(zip(predicted_noun_lists, ground_truth_noun_lists)):
        found = False
        for predicted_noun, ground_truth_noun in zip(predicted_nouns, ground_truth_nouns):
            if predicted_noun == ground_truth_noun:
                found = True
        if found:    
            correct_pairs += 1
    value = (correct_pairs / total) * 100 

    return value


def value_all_metric(predicted_noun_lists, ground_truth_noun_lists, total):

    correct_pairs = 0.0
    
    for idx, (predicted_nouns, ground_truth_nouns) in enumerate(zip(predicted_noun_lists, ground_truth_noun_lists)):
        found = False
        if predicted_nouns == ground_truth_nouns:
            found = True
        if found:    
            correct_pairs += 1
            print("idx: ", idx, predicted_nouns, ground_truth_nouns)
    value_all = (correct_pairs / total) * 100 

    return value_all


def value_metric_at_least_two(predicted_noun_lists, ground_truth_noun_lists, total):
    verbs_with_at_least_two_correct = 0.0
        
    for idx, (predicted_nouns, ground_truth_nouns) in enumerate(zip(predicted_noun_lists, ground_truth_noun_lists)):
        num_correct_pairs = 0
        for predicted_noun, ground_truth_noun in zip(predicted_nouns, ground_truth_nouns):
            if predicted_noun == ground_truth_noun:
                num_correct_pairs += 1
        if num_correct_pairs >= 2:
            verbs_with_at_least_two_correct += 1
    value_two = (verbs_with_at_least_two_correct / total) * 100 
    
    return value_two

def role_based_metrics(pred, gt, roles, incorrect_pred_index):

    if incorrect_pred_index is not None:
        for idx in incorrect_pred_index:
            if idx < len(gt):
                gt[idx] = [5000 for _ in gt[idx]]

    roles = [item for sublist in roles for item in sublist]
    pred = [[int(x) for x in sublist] for sublist in pred]
    gt = [[int(x) for x in sublist] for sublist in gt]
    
    pred = [item for sublist in pred for item in sublist]
    gt = [item for sublist in gt for item in sublist]

    assert len(gt) == len(pred) == len(roles), f"Lengths mismatch: gt={len(gt)}, pred={len(pred)}, roles={len(roles)}"
    
    # Find unique classes
    unique_roles = set(roles)
    
    # Initialize lists to store metrics for each role class
    accuracies = []
    macro_precisions = []
    macro_recalls = []
    macro_f1_scores = []
    
    # Iterate over unique role classes
    for role in unique_roles:

        # Get indices of instances belonging to current role class
        role_indices = [i for i, r in enumerate(roles) if r == role]
        
        # Extract ground truth and predicted values for current role class
        role_gt = [gt[i] for i in role_indices]
        role_pred = [pred[i] for i in role_indices]
    
        # Calculate accuracy
        accuracy = accuracy_score(role_gt, role_pred)*100.
        accuracies.append(accuracy)
    
        # Calculate macro precision
        macro_precision = precision_score(role_gt, role_pred, average='macro')*100.
        macro_precisions.append(macro_precision)
    
        # Calculate macro recall
        macro_recall = recall_score(role_gt, role_pred, average='macro')*100.
        macro_recalls.append(macro_recall)
    
        # Calculate macro F1-score
        macro_f1_score = f1_score(role_gt, role_pred, average='macro')*100.
        macro_f1_scores.append(macro_f1_score)
    
    # Calculate average metrics
    avg_accuracy = np.mean(accuracies)
    avg_macro_precision = np.mean(macro_precisions)
    avg_macro_recall = np.mean(macro_recalls)
    avg_macro_f1_score = np.mean(macro_f1_scores)
    
    return avg_accuracy, avg_macro_precision, avg_macro_recall, avg_macro_f1_score

# def role_based_metrics_top1_setting(splitted_gt, splitted_pred, roles, incorrect_pred_index):

#     # Replace splitted_gt's (after remove empty sublists if any-should not be available for SSG) role values with 5000 (value that does not exist to penalize them if their top-1 predicted verb is icnorrrect)
#     roles = [item for sublist in roles for item in sublist]
#     gt_list = splitted_gt
#     for idx in incorrect_pred_index:
#         if idx < len(gt_list):
#             gt_list[idx] = [5000 for _ in gt_list[idx]]

#     #  Obtain gt, pred, role names (use the sr code to to obtain roles)
#     pred = splitted_pred
#     gt = gt_list
#     roles = roles
    
#     pred = [[int(x) for x in sublist] for sublist in pred]
#     gt = [[int(x) for x in sublist] for sublist in gt]
    
#     pred = [item for sublist in pred for item in sublist]
#     gt = [item for sublist in gt for item in sublist]
    
#     assert len(gt) == len(pred) == len(roles), f"Lengths mismatch: gt={len(gt)}, pred={len(pred)}, roles={len(roles)}"
    
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
    
#     # Calculate average metrics
#     avg_accuracy = np.mean(accuracies)
#     avg_macro_precision = np.mean(macro_precisions)
#     avg_macro_recall = np.mean(macro_recalls)
#     avg_macro_f1_score = np.mean(macro_f1_scores)
    
#     return avg_accuracy, avg_macro_precision, avg_macro_recall, avg_macro_f1_score


class EvaluateInComNet:

    def evaluate(self, obj_sr_gt_list, obj_sr_pred_list, obj_frame_lengths, obj_roles, verb_gt_list, verb_pred_list, verb_sr_gt_list, verb_sr_pred_list, verb_frame_lengths, verb_roles):

        # Remove padded elements (pad_value=365)
        obj_sr_pad_indices = [i for i, x in enumerate(obj_sr_gt_list) if x == 365]
        obj_sr_gt_list = remove_elements_by_indexes(obj_sr_gt_list, obj_sr_pad_indices)
        obj_sr_pred_list = remove_elements_by_indexes(obj_sr_pred_list, obj_sr_pad_indices)
        
        verb_pad_indices = [i for i, x in enumerate(verb_gt_list) if x == 365]
        verb_gt_list = remove_elements_by_indexes(verb_gt_list, verb_pad_indices)
        verb_pred_list = remove_elements_by_indexes(verb_pred_list, verb_pad_indices)

        verb_sr_pad_indices = [i for i, x in enumerate(verb_sr_gt_list) if x == 365]
        verb_sr_gt_list = remove_elements_by_indexes(verb_sr_gt_list, verb_sr_pad_indices)
        verb_sr_pred_list = remove_elements_by_indexes(verb_sr_pred_list, verb_sr_pad_indices)
    

        #### Verb evaluation
        verb_gt_list = np.array(verb_gt_list)
        verb_pred_list = np.array(verb_pred_list)
        
        acc=accuracy_score(verb_gt_list, verb_pred_list)*100.
        unique_classes = np.unique(np.concatenate((verb_gt_list, verb_pred_list)))
        per_class_accuracy = {}

        for cls in unique_classes:
            true_mask = (verb_gt_list == cls)
            pred_mask = (verb_pred_list == cls)
            class_accuracy = accuracy_score(verb_gt_list[true_mask], verb_pred_list[true_mask])*100
            per_class_accuracy[cls] = class_accuracy

        mP=precision_score(verb_gt_list, verb_pred_list, average='macro')*100.
        mR=recall_score(verb_gt_list, verb_pred_list, average='macro')*100. # zero_division=1   
        f1 = f1_score(verb_gt_list, verb_pred_list, average='macro')*100
        # verb_results = {"Accuracy":acc, "mR": mR, "mP": mP, "F1": f1, "per_clss_accuracy": per_class_accuracy}
        verb_results = {"Accuracy":acc, "mR": mR, "mP": mP, "F1": f1}


        #### Object SR evaluation        
        acc=accuracy_score(obj_sr_gt_list, obj_sr_pred_list)*100.
        mP=precision_score(obj_sr_gt_list, obj_sr_pred_list, average='macro')*100.
        mR=recall_score(obj_sr_gt_list, obj_sr_pred_list, average='macro')*100. # zero_division=1   
        f1 = f1_score(obj_sr_gt_list, obj_sr_pred_list, average='macro')*100.

        splitted_pred = split_list(obj_sr_pred_list, obj_frame_lengths)
        splitted_gt = split_list(obj_sr_gt_list, obj_frame_lengths)

        value = value_metric(splitted_pred, splitted_gt, len(splitted_gt))    
        value_two = value_metric_at_least_two(splitted_pred, splitted_gt, len(splitted_gt))
        value_all = value_all_metric(splitted_pred, splitted_gt, len(splitted_gt))
        
        incorrect_verb_pred_index = None
        role_based_acc, role_based_mP, role_based_mR, role_based_f1 = role_based_metrics(splitted_pred, splitted_gt, obj_roles, incorrect_verb_pred_index)
        obj_sr_results = {"Accuracy":acc, "mR": mR, "mP": mP, "F1": f1, "Value":value, "Value-two": value_two, "value-all":value_all, "role_based_accuracy": role_based_acc, "role_based_mP" : role_based_mP, "role_based_mR": role_based_mR, "role_based_f1":role_based_f1 }
    

        #### Verb SR evaluation        
        acc=accuracy_score(verb_sr_gt_list, verb_sr_pred_list)*100.
        mP=precision_score(verb_sr_gt_list, verb_sr_pred_list, average='macro')*100.
        mR=recall_score(verb_sr_gt_list, verb_sr_pred_list, average='macro')*100. # zero_division=1   
        f1 = f1_score(verb_sr_gt_list, verb_sr_pred_list, average='macro')*100.

        splitted_pred = split_list(verb_sr_gt_list, verb_frame_lengths)
        splitted_gt = split_list(verb_sr_pred_list, verb_frame_lengths)
        total_frames = len(splitted_gt)

        verb_gt_list = verb_gt_list.tolist()
        verb_pred_list = verb_pred_list.tolist()

        if args.top1:
            incorrect_verb_pred_index = [i for i, (x, y) in enumerate(zip(verb_gt_list, verb_pred_list)) if x != y]
            # role_based_acc, role_based_mP, role_based_mR, role_based_f1 = role_based_metrics_top1_setting(splitted_gt, splitted_pred, verb_roles, incorrect_verb_pred_index)
            role_based_acc, role_based_mP, role_based_mR, role_based_f1 = role_based_metrics(splitted_pred, splitted_gt, verb_roles, incorrect_verb_pred_index)

            splitted_pred = remove_elements_by_indexes(splitted_pred, incorrect_verb_pred_index) # take this forl calculating role-based mp, mr and f1 in top-1 setting
            splitted_gt = remove_elements_by_indexes(splitted_gt, incorrect_verb_pred_index) # take this forl calculating role-based mp, mr and f1 in top-1 setting
            verb_roles = remove_elements_by_indexes(verb_roles, incorrect_verb_pred_index) # take this forl calculating role-based mp, mr and f1 in top-1 setting
            verb_gt_list = remove_elements_by_indexes(verb_gt_list, incorrect_verb_pred_index)

            value = value_metric(splitted_pred, splitted_gt, total_frames)
            value_two = value_metric_at_least_two(splitted_pred, splitted_gt, total_frames)
            value_all = value_all_metric(splitted_pred, splitted_gt, total_frames)

        else:
            incorrect_verb_pred_index = None
            role_based_acc, role_based_mP, role_based_mR, role_based_f1 = role_based_metrics(splitted_pred, splitted_gt, verb_roles, incorrect_verb_pred_index)
            value = value_metric(splitted_pred, splitted_gt, total_frames)
            value_two = value_metric_at_least_two(splitted_pred, splitted_gt, total_frames)
            value_all = value_all_metric(splitted_pred, splitted_gt, total_frames)
   
        verb_sr_results = {"Accuracy":acc, "mR": mR, "mP": mP, "F1": f1, "Value":value, "Value-two": value_two, "value-all":value_all, "role_based_accuracy": role_based_acc, "role_based_mP" : role_based_mP, "role_based_mR": role_based_mR, "role_based_f1":role_based_f1 }

        return obj_sr_results, verb_results, verb_sr_results
    
    def evaluate_person_sr(self, person_sr_gt_list, person_sr_pred_list, person_frame_lengths, person_roles):
            
            acc=accuracy_score(person_sr_gt_list, person_sr_pred_list)*100.
            mP=precision_score(person_sr_gt_list, person_sr_pred_list, average='macro')*100.
            mR=recall_score(person_sr_gt_list, person_sr_pred_list, average='macro')*100. 
            f1 = f1_score(person_sr_gt_list, person_sr_pred_list, average='macro')*100.

            splitted_pred = split_list(person_sr_pred_list, person_frame_lengths)
            splitted_gt = split_list(person_sr_gt_list, person_frame_lengths)

            value = value_metric(splitted_pred, splitted_gt, len(splitted_gt))
            value_two = value_metric_at_least_two(splitted_pred, splitted_gt, len(splitted_gt))
            value_all = value_all_metric(splitted_pred, splitted_gt, len(splitted_gt))

            incorrect_pred_index = None
            role_based_acc, role_based_mP, role_based_mR, role_based_f1 = role_based_metrics(splitted_pred, splitted_gt, person_roles, incorrect_pred_index)

            person_sr_result = {"Value":value, "Value-two": value_two, "value-all":value_all, "role_based_accuracy": role_based_acc, "role_based_mP" : role_based_mP, "role_based_mR": role_based_mR, "role_based_f1":role_based_f1 }
            
            return person_sr_result

















