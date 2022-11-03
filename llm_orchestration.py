import sys
sys.path.insert(0, "/notebooks/pipenv")
sys.path.insert(0, "/notebooks/nebula3_database")
sys.path.insert(0, "/notebooks/nebula3_experiments")
sys.path.insert(0, "/notebooks/nebula3_videoprocessing")
sys.path.insert(0, "/notebooks/")

from abc import ABC, abstractmethod
from enum import Enum
import requests
# import visual_genome.local as vg
import json
import copy
import operator
import itertools
import subprocess
import time
import typing

import numpy as np
import torch
import spacy
import nltk
import openai

from typing import NamedTuple
from database.arangodb import DatabaseConnector
from config.config import NEBULA_CONF

from videoprocessing.vlm_factory import VlmFactory
from videoprocessing.vlm_interface import VlmInterface
from videoprocessing.vlm_implementation import VlmChunker, BlipItcVlmImplementation


IPC_PATH = '/storage/ipc_data/paragraphs_v1.json'
GLOBAL_TOKENS_COLLECTION = 's3_global_tokens'
LOCAL_TOKENS_COLLECTION = 's3_local_yolo'
# LOCAL_TOKENS_COLLECTION = 's3_local_tokens'
# LOCAL_TOKENS_COLLECTION = 's3_local_rpn'
VISUAL_CLUES_COLLECTION = 's4_visual_clues_'
MOVIES_COLLECTION = "Movies"
LLM_OUTPUT_COLLECTION = "s4_llm_output"
KEY_COLLECTION = "llm_config"
FS_GPT_MODEL = 'text-davinci-002'
FS_SAMPLES = 5                   # Samples for few-shot gpt
LocalSource = Enum('LocalSource', 'GT RETRIEVAL')
IPCImageId = int

class MovieImageId(NamedTuple):
    movie_id: str
    frame_num: int

ImageId = typing.Union[IPCImageId,MovieImageId]

def image_id_as_dict(id: ImageId):
    if type(id) == IPCImageId:
        return {'image_id': id}
    else:
        return {'movie_id': id.movie_id,
                'frame_num': id.frame_num}


def flatten(lst): return [x for l in lst for x in l]

class NEBULA_DB:
    def __init__(self):
        config = NEBULA_CONF()
        self.db_host = config.get_database_host()
        self.pg_database = config.get_playground_name()
        self.database = config.get_database_name()
        self.gdb = DatabaseConnector()
        self.db = self.gdb.connect_db(self.database)
        self.pg_db = self.gdb.connect_db(self.pg_database)

    def get_image_id_from_collection(self, id: IPCImageId,collection=GLOBAL_TOKENS_COLLECTION):
        results = {}
        query = 'FOR doc IN {} FILTER doc.image_id == {} RETURN doc'.format(collection,id)
        cursor = self.pg_db.aql.execute(query)
        for doc in cursor:
            results.update(doc)
        return results
    
    def get_image_url(self, id: MovieImageId) -> str:
        return 'no url for now'

    def get_movie_structure(self, movie_id: str):
        rc = {}
        query = 'FOR doc IN {} FILTER doc._id == "{}" RETURN doc'.format(MOVIES_COLLECTION,movie_id)
        cursor = self.db.aql.execute(query)
        for doc in cursor:
            rc.update(doc)
        return dict(zip(flatten(rc['mdfs']),rc['mdfs_path']))

    def get_movie_frame_from_collection(self, mid: MovieImageId, collection=VISUAL_CLUES_COLLECTION):
        results = {}
        query = 'FOR doc IN {} FILTER doc.movie_id == "{}" AND doc.frame_num == {} RETURN doc'.format(collection,mid.movie_id, mid.frame_num)
        cursor = self.db.aql.execute(query)
        for doc in cursor:
            results.update(doc)
        return results

    def write_movie_frame_doc_to_collection(self, mid: MovieImageId, mobj: dict, collection: str, check_exists=False):
        if check_exists:
            rc = self.get_movie_frame_from_collection(mid,collection)
            if rc:
                print("write_movie_frame_doc_to_collection: Document with id {} already exists in collection {}".format(mid,collection))
                return
        query = "INSERT {} INTO {}".format(mobj,collection)
        cursor = self.db.aql.execute(query)  

    def get_llm_key(self):
        results = {}
        query = 'FOR doc IN {} FILTER doc.keyname == "openai" RETURN doc'.format(KEY_COLLECTION,)
        cursor = self.db.aql.execute(query)
        for doc in cursor:
            results.update(doc)
        return results['keyval']




def gpt_execute(prompt_template, *args, **kwargs):            
    prompt = prompt_template.format(*args)   
    response = openai.Completion.create(prompt=prompt, max_tokens=256, **kwargs)   
    # return response
    return [x['text'].strip() for x in response['choices']]

def get_size_text(rect: [int, int, int, int], size: [int, int]):
    rect_size = (rect[2]-rect[0])*(rect[3]-rect[1])
    img_size = size[0]*size[1]
    rel_size = rect_size / img_size
    if rel_size < 0.15:
        return "small"
    elif rel_size < 0.6:
        return "moderate"
    else:
        return "large"

def get_location_text(rect: [int, int, int, int], size: [int, int]):
    center_x = (rect[0]+rect[2])/2
    center_y = (rect[1]+rect[3])/2
    x_1 = size[0]/3
    x_2 = 2*x_1
    y_1 = size[1]/3
    y_2 = 2*y_1
    if center_x < x_1:
        x_text = 'left'
    elif center_x < x_2:
        x_text = 'middle'
    else:
        x_text = 'right'
    if center_y < y_1:
        y_text = 'upper'
    elif center_y < y_2:
        y_text = ''
    else:
        y_text = 'lower'
    
    return '{} {}'.format(y_text,x_text).strip()

def get_uncertainty_prompt(objects, attributes):
    uncertaint_prompt = '''Find the  most likely combination of an object and its attribute:

Objects: girl, sky, car, leaf, bubble
Attributes: fast, irridiscent, illuminated
Likely combination: irridiscent bubble

Objects: computer, cup, hair, boy
Attributes: lonely, long, chatty, blue
Likely combination: long hair

Objects: grass, dog, frizbee, roof, solar panel, city, window
Attributes: transparent, crowded, green, unkempt, round, cute
Likely combination: crowded city

Objects {}
Attributes: {}
Likely combination:'''

    return uncertaint_prompt.format(', '.join(objects), ', '.join(attributes))

# returns (attribute, object), best of n

def get_likey_obj_attr_combinations(objects, attributes, n=1, **kwargs):
    prompt = get_uncertainty_prompt(objects, attributes)
    rc = gpt_execute(prompt, model=FS_GPT_MODEL, n=n, **kwargs)
    if n>1:
        rc = sorted([(x[0],len(list(x[1]))) for x in itertools.groupby(sorted(rc)) if len(x[0].split())==2], key=lambda x:-x[1])
        try:
            rc = rc[0][0].split()
        except Exception as e:
            print('gah, exception:')
            print(rc)
            return [attributes[0], objects[0]]
    else:
        rc = rc[0].split()
    return [[x] for x in rc]
    # rc = rc[0].split(', ')
    # return tuple(zip(*[x.split() for x in rc]))

class ICandidatesFilter(ABC):
    def __init__(self):
        super().__init__() 

    @abstractmethod
    def candidates_from_paragraph(self, paragraph: str, vlm: VlmInterface, image_url: str) -> list[str]:
        pass

class SubsetCandidatesFilter(ICandidatesFilter):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load('en_core_web_lg')

    def candidates_from_paragraph(self, paragraph: str, vlm: VlmInterface, image_url: str) -> list[str]:
        senter = self.nlp.get_pipe("senter")
        sentences = [str(x) for x in senter(self.nlp(paragraph)).sents]
        n = len(sentences)
        cands = []
        for i in range(3,n+1):
            for comb in itertools.combinations(range(n),i):
                cands.append(' '.join(operator.itemgetter(*comb)(sentences)))
        scores = vlm.compute_similarity_url(image_url,cands)
        cand = cands[np.argmax(scores)]
        return cand    

class FixedThresholdCandidatesFilter(ICandidatesFilter):
    def __init__(self, threshold):
        self.threshold = threshold
        self.nlp = spacy.load('en_core_web_lg')

    def candidates_from_paragraph(self, paragraph: str, vlm: VlmInterface, image_url: str) -> list[str]:
        senter = self.nlp.get_pipe("senter")
        sentences = [str(x) for x in senter(self.nlp(paragraph)).sents]
        scores = vlm.compute_similarity_url(image_url,sentences)
        # print(scores)
        return ' '.join([x for (x,y) in zip(sentences,scores) if y>self.threshold])


class GTBaseGenerator:
    def __init__(self, ipc_path=IPC_PATH, num_objects=1):
        self.pipeline = NEBULA_DB()
        self.ipc_data = json.load(open(ipc_path,'r'))
        self.global_captioner = 'blip'
        self.global_tagger = 'blip'
        self.local_tagger = 'yolo'
        self.places_source = 'blip'        
        self.num_objects = num_objects
        # self.local_captions = local_captions
        # self.local_spatial_text = local_spatial_text
        self.local_prefix = "Objects in this image:\n"
        self.local_prompt_with_attr = "{}. Attributes: {}."
        self.local_prompt = "{}."
        self.global_prompt1 = '''Caption of image: {}
This image is taking place in: {}
Tags: This image is about {}
Describe this image in detail:'''

    
    def get_image_structure(self, id: IPCImageId):
        global_doc = self.pipeline.get_image_id_from_collection(id)
        if not global_doc:
            print("Couldn't find global tokens for id {}".format(id))
            return
        rc_doc = {
            'image_id': id,
            'url': global_doc['url']    
        }
        for (k,v) in global_doc.items():
            if k.startswith('global'):
                rc_doc[k]=copy.copy(v)
        rois = []        
        local_doc = self.pipeline.get_image_id_from_collection(id,collection=LOCAL_TOKENS_COLLECTION)
        if not local_doc:
            print("Couldn't find local tokens for id {}".format(id))
            return
        for roi in local_doc['roi']:
            objects = list(list(zip(*roi['local_objects'][self.local_tagger][:self.num_objects]))[0])
            attrs = list(list(zip(*roi['local_attributes'][self.local_tagger][:self.num_objects]))[0]) if 'local_attributes' in roi.keys() else []
            bbox = roi['bbox']
            if type(bbox)==str:
                bbox = eval(bbox)
            caption = roi['local_captions'][self.local_tagger] if 'local_captions' in roi.keys() else ''
            obj_doc = {
                'objects': objects, 
                'attributes': attrs, 
                'bbox': [bbox[0],bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]],
                'caption': caption                 
            }           
            rois.append(obj_doc)
        rc_doc['rois']=rois

        return rc_doc

    def get_prompt(self, id: IPCImageId, include_local=False, include_attributes=False, include_answer=False, local_captions=False, local_spatial_text = False, reduce_uncertainty = False):
        base_doc = self.get_image_structure(id)
        if base_doc == None:
            return
        caption = base_doc['global_captions'][self.global_captioner]
        all_objects = base_doc['global_objects'][self.global_tagger]
        all_persons = base_doc['global_persons'][self.global_tagger]
        all_places = base_doc['global_scenes'][self.places_source]
        # print("Caption: {}".format(caption))
        # print("Objects: ")
        # print(all_objects[:5])
        # print("Places:")
        # print(all_places[:5])
        # print("Persons:")
        # print(all_persons[:5])
        objects = '; '.join([x['label'] for x in all_objects[:8]])
        persons = '; '.join([x['label'] for x in all_persons[:5]])
        places = ' or '.join([x['label'] for x in all_places[:3]])
        local_prompt = ""
        if include_local:
            roi_prompts = []
            for roi in base_doc['rois']:
                local_objects = roi['objects']
                local_attributes = roi['attributes']                    
                if include_attributes and local_attributes:
                    if reduce_uncertainty:
                        local_attributes, local_objects = get_likey_obj_attr_combinations(local_objects,local_attributes,n=6)            
                    roi_prompt = self.local_prompt_with_attr.format(', '.join(local_objects),', '.join(local_attributes))
                else:
                    roi_prompt = self.local_prompt.format(', '.join(local_objects))
                if local_captions:
                    roi_prompt = "{}. {}".format(roi['caption'],roi_prompt)
                roi_prompts.append(roi_prompt)
            local_prompt = self.local_prefix+'\n'.join(roi_prompts)+'\n'
               
        prompt_before_answer = self.global_prompt1.format(caption,places,objects)
        if include_answer:                 
            [answer] = [x['paragraph'] for x in self.ipc_data if x['image_id']==id]
            final_prompt = prompt_before_answer+" "+answer
        else:
            final_prompt = prompt_before_answer
        return local_prompt+final_prompt


    # Ilan changed the format in the database, so this is the respective change in the code parsing the object. This is for YOLO

    def get_movie_frame_prompt(self, mid: MovieImageId, include_local=False, include_attributes=False, local_captions=False, local_spatial_text = False, reduce_uncertainty = False, **kwargs):
        base_doc = self.pipeline.get_movie_frame_from_collection(mid)
        caption = base_doc['global_caption'][self.global_captioner]
        all_objects = base_doc['global_objects'][self.global_tagger]
        all_persons = base_doc['global_persons'][self.global_tagger]
        all_places = base_doc['global_scenes'][self.places_source]
        objects = '; '.join([x[0] for x in all_objects[:8]])
        persons = '; '.join([x[0] for x in all_persons[:5]])
        places = ' or '.join([x[0] for x in all_places[:3]])
        print(objects)
        print(persons)
        print(places)
        print(caption)
        local_prompt = ""
        if include_local:
            roi_prompts = []
            for roi in base_doc['roi']:
                local_object = roi['bbox_object']
                roi_prompt = self.local_prompt.format(local_object)
                roi_prompts.append(roi_prompt)
            local_prompt = self.local_prefix+'\n'.join(roi_prompts)+'\n'
               
        prompt_before_answer = self.global_prompt1.format(caption,places,objects)
        final_prompt = prompt_before_answer
        return local_prompt+final_prompt

    def generate_prompt(self, ids: list[IPCImageId], target_id: ImageId = None, **kwargs):
        rc = []
        for id in ids:  
            rc.append(self.get_prompt(id,include_answer=True, **kwargs))
        if target_id:  
            if type(target_id) == MovieImageId:
                rc.append(self.get_movie_frame_prompt(target_id,include_answer=False, **kwargs))
            else:
                rc.append(self.get_prompt(target_id,include_answer=False, **kwargs))
        return '\n'.join(rc)

    def few_shot_process_target_id(self, fs_ids: list[IPCImageId],target_id: ImageId, n=5, debug_print_prompt=False, **kwargs):
        fs_prompt = self.generate_prompt(fs_ids, target_id=target_id, **kwargs)
        if debug_print_prompt:
            print("Prompt:\n---------------------------------\n")
            print(fs_prompt)
            print("\n---------------------------------\n")
        results = gpt_execute(fs_prompt, model=FS_GPT_MODEL, n=n)
        # results = opt_execute(fs_prompt, num_return_sequences=n)
        return results

class LlmTaskInternal:
    def __init__(self):
        self.config = NEBULA_CONF
        self.nebula_db = NEBULA_DB()
        self.prompt_obj = GTBaseGenerator()
        self.vlm = VlmChunker(VlmFactory().get_vlm("blip_itc"), chunk_size=50)
        self.cand_filter = SubsetCandidatesFilter()

        try:
            with open('/storage/keys/openai.key','r') as f:
                OPENAI_API_KEY = f.readline().strip()
            openai.api_key = OPENAI_API_KEY
        except:
            openai.api_key = self.nebula_db.get_llm_key()

        with open('s3_ids.json','r') as f:
            self.s3_ids = json.load(f)

    def get_all_s3_ids(self):
        query = 'FOR doc IN {} RETURN doc.image_id'.format(GLOBAL_TOKENS_COLLECTION)
        cursor = self.nebula_db.db.aql.execute(query)
        return [doc for doc in cursor]

    def process_target_id(self, target_id: ImageId, image_url=None, fs_samples=FS_SAMPLES, cand_filter=SubsetCandidatesFilter(),  **kwargs):
        if image_url == None:
            rc = self.nebula_db.get_movie_frame_from_collection(target_id)
            assert(rc)
            image_url = rc['url']
        print("Processing target_id {}, url: {}".format(target_id,image_url))
        train_ids = np.random.choice(self.s3_ids,fs_samples)
        rc = self.prompt_obj.few_shot_process_target_id(train_ids, target_id, **kwargs)
        candidates = [self.cand_filter.candidates_from_paragraph(x,self.vlm,image_url) for x in rc]
        scores = self.vlm.compute_similarity_url(image_url,candidates)
        cand = candidates[np.argmax(scores)]
        rc = {
            'url': image_url,
            'paragraphs': rc,
            'candidate': cand
        }
        return {**image_id_as_dict(target_id), **rc}

    def process_movie(self, movie_id: str, **kwargs):
        mdfs = self.nebula_db.get_movie_structure(movie_id)
        for frame in mdfs.keys():
            mid = MovieImageId(movie_id,frame)
            print('Processing movie {}, frame #{}'.format(movie_id,frame))
            rc = self.process_target_id(mid, **kwargs)
            if rc:
                self.nebula_db.write_movie_frame_doc_to_collection(mid,rc,LLM_OUTPUT_COLLECTION)
            else:
                return False,1
        return True, None


    