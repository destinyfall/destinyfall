import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from sentence_transformers import SentenceTransformer

# chroma run --path /db_path
# 加载环境变量
# 初始化OpenAI客户端
_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEYO")
)
model = SentenceTransformer('all-MiniLM-L6-v2')
def get_completion(prompt, documents, model="gpt-3.5-turbo-1106"):
    # 将 documents 扁平化
    documents = [doc for sublist in documents for doc in sublist]
    context = ' '.join(documents)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def get_embeddings(texts):
    #print(f"Input texts: {texts}")
    embeddings = model.encode(texts)
    embeddings = embeddings.tolist() # 将 numpy 数组转换为列表
    #print(f"Output embeddings: {embeddings}")
    return embeddings

def extract_text(pdf_path, page_numbers=None, min_line_length=1):
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(pdf_path)):
        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' '+text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    return paragraphs
# 创建向量数据库的实例

class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        # 为了演示，实际不需要每次 reset()
        chroma_client.reset()

        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(name="demo")
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        '''向 collection 中添加文档与向量'''
        self.collection.add(
            embeddings=self.embedding_fn(documents),
            documents=documents,
            ids=[f"id{i}" for i in range(len(documents))]
        )
    def search(self, query, top_n):
        '''检索向量数据库'''
        print(f"Input query: {query}")
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        print(f"Output results: {results}")
        print(f"Retrieved documents: {results['documents']}")
        return results

class RAG_Bot:
    def __init__(self, paragraphs, llm_api, vector_db, n_results=3):
        self.vector_db = vector_db
        self.vector_db.add_documents(paragraphs)
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        # 1. Retrieve search results
        search_results = self.vector_db.search(user_query, top_n=3)
        #print(f"User query: {user_query}, type: {type(user_query)}")
        # 2. Build prompt
        prompt_template = "您想知道关于 '{query}' 的什么信息？"
        prompt = self.build_prompt(
            prompt_template, info=search_results['documents'][0], query=user_query)
        # 检查 prompt 和 documents 的类型
        #print(f"Prompt: {prompt}, type: {type(prompt)}")
        #print(f"Documents: {search_results['documents']}, type: {type(search_results['documents'])}")
        # 3. Call LLM
        response = self.llm_api(prompt, search_results['documents'])
        return response

    def build_prompt(self, prompt_template, info, query):
        # 根据查询构建提示
        prompt = prompt_template.format(query=query)
        return prompt

pdf_path = 'test1.pdf'
#page_numbers = [0, 1]
paragraphs = extract_text(pdf_path, page_numbers=None)

# 创建一个向量数据库对象
vector_db = MyVectorDBConnector("demo", get_embeddings)

# 创建RAG_Bot实例
bot = RAG_Bot(paragraphs, llm_api=get_completion, vector_db=vector_db)
#bot.vector_db.add_documents(paragraphs)

# 打印搜索结果
#print(results)
while True:
    # 进行对话
    user_query = input("主人您想知道什么：")
    if user_query.lower() == "再见":
        break
    results = vector_db.search(user_query, top_n=4)
    # 选择最相关的文档作为答案
    best_document = results['documents'][0][0][0][0]
    response = bot.chat(user_query)
    print("小Q:", response)
