from openai import OpenAI
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()

# 建议这里存的是你 proxy 的 key，比如：

api_key = os.getenv('EMBEDDING_API_KEY')
if not api_key:
    raise ValueError("No API key found. Please check your .env file.")

# ⭐ 关键修改：指向你自己的代理，而不是官方 OpenAI
client = OpenAI(
    api_key=api_key,
    base_url="http://10.248.10.54:5000/v1"  # 注意带 /v1
)

def get_embedding(text, model="jina-embeddings-zh"):
    # 你的 proxy 的 embeddings 接口已经是 OpenAI 兼容格式
    text = text.replace("\n", " ")
    resp = client.embeddings.create(
        model=model,
        input=[text],   # 这里用 list，拿 data[0] 就行
    )
    return resp.data[0].embedding

# Sample documents（你也可以换成中文句子测试）
documents = [
    "The sky is blue and beautiful.",
    "Love this blue and beautiful sky!",
    "The quick brown fox jumps over the lazy dog.",
    "A king's breakfast has sausages, ham, bacon, eggs, toast, and beans",
    "I love green eggs, ham, sausages and bacon!",
    "The brown fox is quick and the blue dog is lazy!",
    "The sky is very blue and the sky is very beautiful today",
    "The dog is lazy but the brown fox is quick!"
]

# Generate embeddings for each document
embeddings = [get_embedding(doc) for doc in documents]

# Convert embeddings to a numpy array for PCA
embeddings_array = np.array(embeddings)

print("embeddings shape:", embeddings_array.shape)  # 预期是 (8, 768)

# Applying PCA to reduce dimensions to 3
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings_array)

# Creating a 3D plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_embeddings[:, 0],
    y=reduced_embeddings[:, 1],
    z=reduced_embeddings[:, 2],
    mode='markers+text',
    text=documents,          # Hover 显示文本
    hoverinfo='text',
    marker=dict(
        size=12,
        color=list(range(len(documents))),
        opacity=0.8
    )
)])

fig.update_layout(
    title="3D Plot of Document Embeddings (jina-embeddings-zh)",
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='PCA Component 3'
    )
)

fig.show()
