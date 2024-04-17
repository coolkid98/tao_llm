import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_community.llms import SparkLLM
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

IFLYTEK_SPARK_APP_ID = os.environ["IFLYTEK_SPARK_APP_ID"]
IFLYTEK_SPARK_API_KEY = os.environ['IFLYTEK_SPARK_API_KEY']
IFLYTEK_SPARK_API_SECRET = os.environ["IFLYTEK_SPARK_API_SECRET"]
def gen_spark_params(model):
    '''
    构造星火模型请求参数
    '''

    spark_url_tpl = "wss://spark-api.xf-yun.com/{}/chat"
    model_params_dict = {
        # v1.5 版本
        "v1.5": {
            "domain": "general", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v1.1") # 云端环境的服务地址
        },
        # v2.0 版本
        "v2.0": {
            "domain": "generalv2", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v2.1") # 云端环境的服务地址
        },
        # v3.0 版本
        "v3.0": {
            "domain": "generalv3", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v3.1") # 云端环境的服务地址
        },
        # v3.5 版本
        "v3.5": {
            "domain": "generalv3.5", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v3.5") # 云端环境的服务地址
        }
    }
    return model_params_dict[model]

spark_api_url = gen_spark_params(model="v3.5")["spark_url"]
spark_llm = SparkLLM(spark_api_url=spark_api_url)  # 指定 v3.5版本

def generate_response(input_text):
    st.info(spark_llm(input_text))

def get_vectordb():
    embedding = OpenAIEmbeddings()
    persist_directory = './chroma'
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb

#不带历史记录的问答链
def get_qa_chain(question:str):
    vectordb = get_vectordb()
    llm = spark_llm
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]

#带有历史记录的问答链
def get_chat_qa_chain(question:str):
    vectordb = get_vectordb()
    llm = spark_llm
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']

st.title('🦜🔗 涛の大模型')

 # 用于跟踪对话历史
if 'messages' not in st.session_state:
    st.session_state.messages = []

messages = st.container(height=300)
if prompt := st.chat_input("Say something"):
    # 将用户输入添加到对话历史中
    st.session_state.messages.append({"role": "user", "text": prompt})

    # 调用 respond 函数获取回答
    answer = generate_response(prompt)
    # 检查回答是否为 None
    if answer is not None:
        # 将LLM的回答添加到对话历史中
        st.session_state.messages.append({"role": "assistant", "text": answer})

    # 显示整个对话历史
    for message in st.session_state.messages:
        if message["role"] == "user":
            messages.chat_message("user").write(message["text"])
        elif message["role"] == "assistant":
            messages.chat_message("assistant").write(message["text"])

selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])




