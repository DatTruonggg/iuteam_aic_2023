from PIL import Image
import faiss
import matplotlib.pyplot as plt
import math
import numpy as np 
import clip
from langdetect import detect
from nlp_processing import Translation
from sentence_transformers import SentenceTransformer, util
import glob
import os
import torch
import json 
class MyFaiss:
    def __init__(self,root_database:str,  bin_file: str, features: str, id2img_fps, device, translater, model):
        self.root_database = root_database
        self.index = self.load_bin_file(bin_file)
        self.id2img_fps = self.load_json_file(id2img_fps)
        self.features = features
        self.device = device
        self.translater = translater
        self.model = model
    def load_json_file(self, json_path: str):
        js = json.load(open(json_path, 'r'))
        return {int(k):v for k,v in js.items()}
    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)
    
    def show_images(self, image_paths):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))

        for i in range(1, columns*rows + 1):
            if i - 1 < len(image_paths):
                img = plt.imread(image_paths[i - 1])
                ax = fig.add_subplot(rows, columns, i)
                #ax.set_title('/'.join(image_paths[i - 1].split('/')[-1:]))
                title_parts = image_paths[i - 1].split("\\")[-2:]
                title = " - ".join(title_parts)
                ax.set_title(title)
                plt.imshow(img)
                plt.axis("off")

        plt.show()
    def get_path_frame(self, value):
        return self.id2img_fps[int(value)][1]
    def get_frame_info(self,value):
        map_key_path = 'D:\\DatTruong\\All\\2025\\HCM_AI\\Data\\Map_Chung'
        key = self.id2img_fps[int(value)][0]
        temp = key.split('-')
        #df = pd.read_csv(map_key_path +temp[-2]+'.csv')
        temp1 = (temp[-1])
        #temp1 = [temp[-2], (temp[-1])]
        return temp1

    def image_search(self, id_query, k):
        query_feats = self.index.reconstruct(id_query).reshape(1,-1)

        scores, idx_image = self.index.search(query_feats, k=k)
        idx_image = idx_image.flatten()

        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        infos = [info[0].split('-')[1] for info in infos_query]
        image_paths = [info[1] for info in infos_query]

        
        return scores, infos, image_paths
        #return scores, path_imgs, info_imgs
    def text_search(self, text, k):
        #if detect(text) == 'vi':
            #text = self.translater(text)
      
        ###### TEXT FEATURES EXTRACTING ######
        text = self.model.encode([text])
        id_list = []
        score_list = []
        ###### SEARCHING #####
        cos_scores = util.semantic_search(torch.tensor(text),
                                           torch.tensor(self.features, dtype=torch.float32),
                                           top_k=k)
        list_of_dictionaries = cos_scores[0]
        for item in list_of_dictionaries:
            id_list.append(item['corpus_id'])
            score_list.append(item['score'])
        path_imgs = []
        info_imgs =[]
        for i in id_list:
            path_imgs.append(self.get_path_frame(i))
            info_imgs.append(self.get_frame_info(i))

        return cos_scores, path_imgs, info_imgs

    def show_segment(self, id_query_path):
        #id_query_path = id_query_path.replace('/', '\\')
        stt = None  # Khởi tạo stt bằng None trước vòng lặp
        for i, image_path in self.id2img_fps.items():
            if image_path[1] == id_query_path:
                stt = int(i)
                break  # Tìm thấy id_query_path, thoát khỏi vòng lặp

        if stt is not None:
            start = int(stt - 3)
            end = int(stt + 196)
            path_imgs = []
            infos = []
            for i in range(start, end + 1):
                path_imgs.append(self.get_path_frame(i))
                infos.append(self.get_frame_info(i))
            return path_imgs, infos





