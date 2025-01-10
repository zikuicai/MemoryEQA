import torch
from PIL import Image
import faiss
import numpy as np
import torch.nn.functional as F

class DynamicKnowledgeBase:
    def __init__(self, text_embedder, image_model, dim=384, gpu_id=0):
        # 初始化FAISS索引和文本图像嵌入器
        res = faiss.StandardGpuResources()  # GPU资源
        # index_cpu = faiss.IndexFlatL2(dim)
        # self.index = faiss.index_cpu_to_all_gpus(index_cpu)
        self.index = faiss.GpuIndexFlatL2(res, dim)  # FAISS GPU索引
        
        self.text_embedder = text_embedder
        self.image_model, self.preprocess = image_model
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

    def search(self, query, top_k=5, device='cuda'):
        # 对查询进行嵌入编码（文本或图像）
        if isinstance(query, str):
            query_embedding = self.text_embedder.encode([query], convert_to_numpy=True)
        else:
            query_input = self.preprocess(query).unsqueeze(0).to(device)  # 对图像进行处理
            with torch.no_grad():
                query_embedding = self.image_model.encode_image(query_input).cpu().numpy()

        # 从FAISS索引中检索最相关的top_k项
        D, I = self.index.search(query_embedding.astype(np.float32), top_k)
        return [self.data[i] for i in I[0]]  # 返回检索到的文本或图像内容

    def clear(self):
        # 清空FAISS索引和数据
        self.index.reset()
        self.data = []