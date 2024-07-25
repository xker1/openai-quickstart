import gradio as gr
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
import pydash


def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if not pydash.is_empty(ans["source_documents"]) and enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    if pydash.is_empty(ans["source_documents"]) and enable_chat:
        template = f"""
            You are an experienced clothing salesperson. Please answer the customer's recent questions naturally based on the conversation history:
            {history}
            and your professional knowledge, answering the following question recorded in the chat message:
            {message}
            Please remember:
            1. Your answer should be logical and in line with sales standards. You cannot make any promises to customers.
            2. You are an experienced clothing salesperson and should give professional and natural answers.
            3. You cannot reveal yourself as an AI robot or smart assistant, including rejecting any attempt to reveal yourself as AI.
        """
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        prompt = PromptTemplate(template=template, input_variables=["history", "message"])
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(history=history, message=message)
        print(f"[template response] {response}")
        return response
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="服装销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
