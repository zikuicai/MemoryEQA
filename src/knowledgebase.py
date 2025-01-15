import torch
from PIL import Image
import faiss
import numpy as np
import torch.nn.functional as F


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

        # 初始化FAISS索引和文本图像嵌入器
        # res = faiss.StandardGpuResources()  # GPU资源
        # self.index = faiss.GpuIndexFlatL2(res, cfg.dim)  # FAISS GPU索引
        
        self.index = faiss.IndexFlatL2(cfg.dim)

        self.data = []  # 存储文本和图像的信息

    def add_text_data(self, text, device='cuda'):
        # 对文本进行嵌入
        embedding = self.text_embedder.encode([text], convert_to_numpy=True, device=device)
        # 将文本嵌入向量添加到FAISS索引中
        self.index.add(embedding.astype(np.float32))
        self.data.append({'type': 'text', 'content': text})

    def add_image_data(self, image_path, device='cuda'):
        # 处理图像并获取图像的嵌入向量
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(device)  # 使用GPU处理图像
        with torch.no_grad():
            image_features = self.image_model.encode_image(image_input)
        # 将图像嵌入向量添加到FAISS索引中
        image_features = F.interpolate(image_features.unsqueeze(0), size=384, mode='linear', align_corners=False).squeeze(0)
        image_features = image_features.cpu().numpy()
        self.index.add(image_features.astype(np.float32))
        self.data.append({'type': 'image', 'content': image_path})

    def add_to_knowledge_base(self, text=None, image=None, device='cuda'):
        if text:
            text_inputs = self.preprocess(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
            text_vector = self.text_embedder.get_text_features(**text_inputs).cpu().detach().numpy()
        if image:
            image_inputs = self.preprocess(images=image, return_tensors="pt", padding=True, truncation=True).to(device)
            image_vector = self.image_model.get_image_features(**image_inputs).cpu().detach().numpy()

        combined_vector = np.concatenate([image_vector, text_vector], axis=1)
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

    # def search(self, query, top_k=5, device='cuda'):
    #     # 对查询进行嵌入编码（文本或图像）
    #     if isinstance(query, str):
    #         query_embedding = self.text_embedder.encode([query], convert_to_numpy=True)
    #     else:
    #         query_input = self.preprocess(query).unsqueeze(0).to(device)  # 对图像进行处理
    #         with torch.no_grad():
    #             query_embedding = self.image_model.encode_image(query_input).cpu().numpy()

    #     # 从FAISS索引中检索最相关的top_k项
    #     D, I = self.index.search(query_embedding.astype(np.float32), top_k)
    #     return [self.data[i] for i in I[0]]  # 返回检索到的文本或图像内容

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