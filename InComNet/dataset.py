import os
import gc
import pickle

import torch
import numpy as np
import imageio
import clip
import open_clip

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from utils import prep_im_for_blob, im_list_to_blob
from config import parse_args

args = parse_args()

torch.cuda.empty_cache()
gc.collect()


class AG(Dataset):

    def __init__(self, mode, datasize, data_path=None, frame_path=None, filter_nonperson_box_frame=True, filter_small_box=False, preprocess=None):

        self.data_path = data_path
        self.frames_path = frame_path
        self. preprocess = preprocess

        #### collect the object classes
        self.object_classes = ['__background__']
        with open(os.path.join(self.data_path, 'object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

        #### collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(self.data_path, 'relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()

        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        #### collect semantic roles of objects
        self.sr_classes = []
        with open(os.path.join(self.data_path, 'roles.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.sr_classes.append(line)
        f.close()

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]

        self.noun_sr_classes = self.sr_classes[0:21]
        self.rel_sr_classes = self.sr_classes[21:]

        #### collect values of relation semantic roles 
        self.rel_sr_val = []
        with open(os.path.join(self.data_path, 'relation_role_values.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.rel_sr_val.append(line)
        f.close()

        #### collect values of object semantic roles 
        self.noun_sr_val = []
        with open(os.path.join(self.data_path, 'noun_role_values.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.noun_sr_val.append(line)
        f.close()

        self.noun_sr_values = self.noun_sr_val[0:]
        self.rel_sr_values = self.rel_sr_val[0:]

        # print("self.attention_relationships :", self.attention_relationships)
        # print("self.spatial_relationships :", self.spatial_relationships)
        # print("self.contacting_relationships :", self.contacting_relationships)
        # print("self.sr_classes :", self.sr_classes, len(self.sr_classes))
        # print("self.noun_sr_classes :", self.noun_sr_classes, len(self.noun_sr_classes))
        # print("self.rel_sr_classes :", self.rel_sr_classes, len(self.rel_sr_classes))

        print('-------loading annotations---------slowly-----------')

        if filter_small_box:
            with open(self.data_path + 'ag_ssg_combined_person.pkl', 'rb') as f: 
                person_bbox = pickle.load(f)
            f.close()
            with open(self.data_path+'ssg_dataset.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
            f.close()
        else:
            with open(self.data_path + 'ag_ssg_combined_person.pkl', 'rb') as f: 
                person_bbox = pickle.load(f)
            f.close()
            with open(self.data_path+'ssg_dataset.pkl', 'rb') as f:
                object_bbox = pickle.load(f)  # sr.pkl
            f.close()

        if datasize == 'mini':
            small_person = {}
            small_object = {}
            for i in list(person_bbox.keys())[:80000]:
                small_person[i] = person_bbox[i]
                small_object[i] = object_bbox[i]
            person_bbox = small_person
            object_bbox = small_object

        #### collect valid frames
        video_dict = {}
        for i in object_bbox.keys():
            if object_bbox[i][0]['metadata']['set'] == mode: #train or testing?
                frame_valid = False
                for j in object_bbox[i]: # the frame is valid if there is visible bbox
                    if j['visible']:
                        frame_valid = True
                if frame_valid:
                    video_name, frame_num = i.split('/')
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]

        frames_to_remove = ["004QE.mp4/000217.png", "004QE.mp4/000264.png", "004QE.mp4/000276.png", "004QE.mp4/000312.png", "00NN7.mp4/000820.png"] # these frames do not have person SR annotations
        for key, value in video_dict.items():
            video_dict[key] = [v for v in value if v not in frames_to_remove]
            
        self.video_list = []
        self.video_size = [] 
        self.gt_annotations = []
        self.non_gt_human_nums = 0
        self.non_heatmap_nums = 0
        self.non_person_video = 0
        self.one_frame_video = 0
        self.valid_nums = 0

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''

        videos_exclude=['5G9SV.mp4', '651VO.mp4', 'FFYL6.mp4', '1C6P3.mp4', '1K0SU.mp4', 'WF7TY.mp4', '4JOAD.mp4', '83XF0.mp4', 'A4VK8.mp4', '7RXMM.mp4', 'AHLVF.mp4', 'UCDL4.mp4','0UBYY.mp4', '10ND1.mp4', 'XNXW6.mp4', 'F75LG.mp4', 'Y1HGC.mp4', '00607.mp4'] # '01KML.mp4', '028CE.mp4', '069GJ.mp4' (issues in sr.json)
              
        videos_500 = ['001YG.mp4', '004QE.mp4', '00607.mp4', '00HFP.mp4', '00MFE.mp4', '00N38.mp4', '00NN7.mp4', '00T1E.mp4', '00T4B.mp4', '00X3U.mp4', '00YZL.mp4', '00ZCA.mp4', '013SD.mp4', '015XE.mp4', '01K8X.mp4', '01KM1.mp4', '01KML.mp4' '01O27.mp4', '01THT.mp4', '01ZWG.mp4','024PD.mp4', '028CE.mp4', '02CYP.mp4', '02DPI.mp4', '02GMI.mp4', '02SK4.mp4', '02SKC.mp4', '02V54.mp4', '02XLP.mp4', '038WZ.mp4', '03AA8.mp4', '03D66.mp4', '03EW0.mp4', '03M0K.mp4', '03OQS.mp4', '03PRW.mp4', '03TL7.mp4', '03XSP.mp4', '04LAX.mp4', '04MTP.mp4', '05124.mp4', '1DYYP.mp4', 'CGNBJ.mp4', 'BOC1T.mp4']
                   
        for i in video_dict.keys():
            video = []
            if ((i in videos_500) and (i not in videos_exclude)): # ((i in videos_500) and (i not in videos_exclude)) # testing dataset
                gt_annotation_video = []
                for j in video_dict[i]:
                    if filter_nonperson_box_frame:
                        if person_bbox[j]['bbox'].shape[0] == 0: 
                            self.non_gt_human_nums += 1
                            continue
                        else:
                            video.append(j)
                            self.valid_nums += 1
                        
                    gt_annotation_frame = []
                    for k in object_bbox[j]:
                        
                        if k['visible'] and k['contacting_relationship']!=['other_relationship']:
                            assert k['bbox'] != None, 'warning! The object is visible without bbox'
                            k['nouns'] = k['class']
                            k['class'] = self.object_classes.index(k['class'])
                            
                            k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
                            k['attention_relationship'] = torch.tensor([self.attention_relationships.index(r) for r in k['attention_relationship']], dtype=torch.long)
                            k['spatial_relationship'] = torch.tensor([self.spatial_relationships.index(r) for r in k['spatial_relationship']], dtype=torch.long)
                            k['contacting_relationship'] = torch.tensor([self.contacting_relationships.index(r) for r in k['contacting_relationship']], dtype=torch.long)

                            noun_roles=[]
                            noun_role_values=[]
                            relation_roles=[]
                            relation_role_values=[]
                            attributes = k.get("attributes", {})
                            roles = attributes.keys()
                            vals = attributes.values()
                            noun_roles.extend(roles)
                            noun_role_values.extend(vals)
                            noun_role_values = [self.noun_sr_val.index(r) for r in noun_role_values]
                           
                            # Skip roles 'state' and 'purpose' as the dataset contains these roles only for a subset of the annotations.
                            if ('state' in noun_roles):
                                index_of_state = noun_roles.index('state')
                                noun_roles.pop(index_of_state)
                                noun_role_values.pop(index_of_state)

                            if ('purpose' in noun_roles):
                                index_of_purpose = noun_roles.index('purpose')
                                noun_roles.pop(index_of_purpose)
                                noun_role_values.pop(index_of_purpose)

                            k['noun_roles'] = noun_roles
                            k['noun_role_values'] = noun_role_values

                            available_rels = []
                            contact_rel = k.get("contacting_relationship_semantic_role_frame", {})
                                                                                                                                                
                            for dict in contact_rel:
                                rel_dict = dict.get("contacting_relationship")
                                if rel_dict == None:
                                    continue
                                else:
                                    available_rels.append(rel_dict)
                                    frame_dict=dict.get("frame", {})
                                    r_roles = []
                                    r_vals = []
                                    roles = frame_dict.keys()
                                    vals = frame_dict.values()
                                    r_roles.extend(roles)
                                    r_vals.extend(vals)
                                    r_vals = [self.rel_sr_val.index(r) for r in r_vals]
                                    relation_roles.append(r_roles)
                                    relation_role_values.append(r_vals)
                            k['relationships'] =  available_rels
                            k['relation_roles'] = relation_roles
                            k['relation_role_values']  = relation_role_values
                            k['contacting_relationship'] = torch.tensor([self.contacting_relationships.index(r) for r in available_rels], dtype=torch.long)
                            person = person_bbox[j]
                            gt_annotation_frame.append(k)
                                                
                    if gt_annotation_frame != []:
                        
                        gt_annotation_frame.append({'person_bbox': person_bbox[j]['bbox']})
                        
                        person_roles = []
                        person_role_values = []
                        atts = person_bbox[j]['attributes']
                        person_roles.extend(atts.keys())
                        person_role_values.extend(atts.values())
                        person_role_values = [self.noun_sr_val.index(r) for r in person_role_values]
                        gt_annotation_frame.append({'person_roles': person_roles})
                        gt_annotation_frame.append({'person_role_values': person_role_values})

                        gt_annotation_video.append(gt_annotation_frame)
                    else:
                        print("Removed {} as it has one contacting relationship and it is 'other relationship'".format(j))
                        video.remove(j)
            else:
                continue

            if len(video) > 2:
                self.video_list.append(video)
                self.video_size.append(person_bbox[j]['bbox_size'])
                self.gt_annotations.append(gt_annotation_video)
            elif len(video) == 1:
                self.one_frame_video += 1
            else:
                self.non_person_video += 1

        print('x'*60)
        if filter_nonperson_box_frame:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} videos are invalid (no person), remove them'.format(self.non_person_video))
            print('{} videos are invalid (only one frame), remove them'.format(self.one_frame_video))
            print('{} frames have no human bbox in GT, remove them!'.format(self.non_gt_human_nums))
        else:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} frames have no human bbox in GT'.format(self.non_gt_human_nums))
            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(non_heatmap_nums))
        print('x' * 60)

    def __getitem__(self, index):

        frame_names = self.video_list[index]
        processed_ims = []
        im_scales = []
        raw_images_pil = []
        raw_images_np = []

        for idx, name in enumerate(frame_names):

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(os.path.join(self.frames_path, name)) #.convert("RGB") # channel h,w,3
            raw_images_pil.append(img) 
            im= self.preprocess(img)
                
            im = np.array(im)
            im=imageio.core.util.Array(im)
            im = im[:, :, ::-1] # rgb -> bgr
            im = np.transpose(im, [1, 2, 0])
            im, im_scale = prep_im_for_blob(im, [[[0.48145466, 0.4578275, 0.40821073]]], 600, 1000) #cfg.PIXEL_MEANS, target_size, 

            im_scales.append(im_scale)
            processed_ims.append(im)
            raw_images_np.append(name)
        
        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

        return img_tensor, im_info, gt_boxes, num_boxes, index, raw_images_pil, raw_images_np 

    def __len__(self):
        return len(self.video_list)

def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]