import torchvision
import faiss
import numpy as np
import torch.nn.functional as F
import json

from PIL import Image
import torch

class DynamicKnowledgeBase:
    def __init__(self, cfg, device="cuda"):
        self.text_embedder, self.image_model, self.preprocess = None, None, None
        text_model_name = cfg.text.split("/")[-1]
        visual_model_name = cfg.visual.split("/")[-1]
        if text_model_name == visual_model_name and visual_model_name in ["clip-vit-large-patch14"]:
            from transformers import CLIPProcessor, CLIPModel
            self.text_embedder = self.image_model = CLIPModel.from_pretrained(cfg.text).to(device)
            self.preprocess = CLIPProcessor.from_pretrained(cfg.text)
        elif text_model_name == visual_model_name and visual_model_name in ["blip2-opt-2.7b"]:
            from transformers import Blip2Processor, Blip2Model
            self.text_embedder = self.image_model = Blip2Model.from_pretrained(cfg.text, torch_dtype=torch.float16).to(device)
            self.preprocess = Blip2Processor.from_pretrained(cfg.text)
        else:
            from sentence_transformers import SentenceTransformer
            import clip
            self.text_embedder = SentenceTransformer(cfg.text, device=device)
            self.image_model, self.preprocess = clip.load(cfg.visual, device=device)

        self.index = faiss.IndexFlatL2(cfg.dim)
        self.data = []  # 存储文本和图像的信息

    def update_memory(self, text, image, topk, device="cuda"):
        memories = self.search(text, image, topk, device)

        # 得到当前观察的特征用于与历史观察计算相似度
        image_inputs = self.preprocess(images=image, return_tensors="pt", padding=True, truncation=True).to(device)
        current_obs = self.image_model.get_image_features(**image_inputs).cpu().detach().numpy()

        for memory in memories:
            history_obs = memory["image_vector"]
            descript = memory["text"]

            # 计算当前观察和历史观察的相似度
            current_obs = F.normalize(torch.tensor(current_obs), p=2, dim=-1).numpy()
            history_obs = F.normalize(torch.tensor(history_obs), p=2, dim=-1).numpy()
            obs_similarity = np.dot(current_obs, history_obs.T)

            objs = json.loads(descript.split("Objects: ")[-1])
            for obj in objs:
                cls = obj["cls"]
                cap = obj["cap"]
                pos = obj["pos"]

            if obs_similarity > 0.8:
                # 删除记忆索引
                self.index.remove(memory["combined_vector"])
                # 从数据中删除
                self.data.remove(memory)
                # 添加新的记忆
                self.add_to_knowledge_base(text, image, device)

    def add_to_knowledge_base(self, text=None, image=None, device='cuda'):
        # 更新记忆
        self.update_memory(text, image, 5, device)

        image_vector = text_vector = None
        if text:
            text_inputs = self.preprocess(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
            text_vector = self.text_embedder.get_text_features(**text_inputs).cpu().detach().numpy()
        if image:
            image_inputs = self.preprocess(images=image, return_tensors="pt", padding=True, truncation=True).to(device)
            image_vector = self.image_model.get_image_features(**image_inputs).cpu().detach().numpy()

        if image is not None and text is not None:
            combined_vector = np.concatenate([image_vector, text_vector], axis=1)
        elif image is not None:
            combined_vector = image_vector
        else:
            combined_vector = text_vector
        
        # 添加到FAISS索引
        self.index.add(combined_vector)
        # 存储原始数据
        self.data.append({
            "image": image,  # 原始图像
            "text": text,    # 原始文本
            "image_vector": image_vector,  # 图像向量
            "text_vector": text_vector,    # 文本向量
            "combined_vector": combined_vector  # 拼接后的向量
        })

    def search(self, query_text, image, top_k=5, device='cuda'):
        # 编码文本查询
        text_inputs = self.preprocess(text=query_text, return_tensors="pt", padding=True, truncation=True).to(device)
        text_vector = self.text_embedder.get_text_features(**text_inputs).cpu().detach().numpy()

        image_inputs = self.preprocess(images=image, return_tensors="pt", padding=True).to(device)
        image_vector = self.image_model.get_image_features(**image_inputs).cpu().detach().numpy()

        combined_vector = np.concatenate([image_vector, text_vector], axis=1)

        # 在FAISS索引中检索最相似的向量
        distances, indices = self.index.search(combined_vector, top_k)

        # 返回检索到的图像文本对
        retrieved_pairs = [self.data[i] for i in indices[0]]
        return retrieved_pairs

    def clear(self):
        # 清空FAISS索引和数据
        self.index.reset()
        self.data = []