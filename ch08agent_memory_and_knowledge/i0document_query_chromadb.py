from openai import OpenAI
import chromadb

from dotenv import load_dotenv
import os
# ========= 1. 使用本地 Embedding API =============

# 这里假设你的本地服务地址是这个：
load_dotenv()


api_key = os.getenv('EMBEDDING_API_KEY')
print('api_key=', api_key)
if not api_key:
    raise ValueError("No API key found. Please check your .env file.")

# ⭐ 关键修改：指向你自己的代理，而不是官方 OpenAI
client = OpenAI(
    api_key=api_key,
    base_url="http://10.248.10.54:5000/v1"  # 注意带 /v1
)
def get_embedding(text, model="jina-embeddings-zh"):
    """
    调用本地的 embedding 接口。
    如果你的本地模型名字不同，比如 'bge-m3'、'mxbai-embed-large' 等，
    把上面的默认 model 改掉就可以。
    """
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text],
        model=model,
    )
    return response.data[0].embedding

# ========= 2. 示例文档 & 生成向量 =============

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

embeddings = [get_embedding(doc) for doc in documents]
ids = [f"id{i}" for i in range(len(documents))]

# ========= 3. 创建 Chroma 集合 =============

# 内存模式（进程结束就没了）
chroma_client = chromadb.Client()

# 如果你想落盘持久化，可以用：
# import chromadb
# chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.create_collection(name="documents")

collection.add(
    embeddings=embeddings,
    documents=documents,    
    ids=ids
)

# ========= 4. 查询函数 =============

def query_chromadb(query, top_n=2):
    """返回 ChromaDB 中相似度最高的 top_n 条文本"""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )
    # 注意： Chroma 的 distances 默认是距离，数值越小越相似（如果用的是 cosine 距离）
    return [
        (id, score, text)
        for id, score, text in zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0]
        )
    ]

# ========= 5. 交互式查询循环 =============

if __name__ == "__main__":
    while True:
        query = input("Enter a search query (or 'exit' to stop): ")
        if query.lower() == "exit":
            break
        top_n = int(input("How many top matches do you want to see? "))
        search_results = query_chromadb(query, top_n)
        
        print("Top Matched Documents:")
        for id, score, text in search_results:
            print(f"ID:{id} TEXT: {text} DISTANCE: {round(score, 4)}")
        print("\n")
