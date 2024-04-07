from fastapi import FastAPI
from pydantic import BaseModel
from dotenv  import load_dotenv
from  langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv  import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
tokenizer = AutoTokenizer.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection")
model = AutoModelForSequenceClassification.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection")

classifier = pipeline(
  "text-classification",
  model=model,
  tokenizer=tokenizer,
  truncation=True,
  max_length=512,
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline


tokenizer = AutoTokenizer.from_pretrained("laiyer/unbiased-toxic-roberta-onnx")
model = ORTModelForSequenceClassification.from_pretrained("laiyer/unbiased-toxic-roberta-onnx",file_name="model.onnx")
toxic_classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyA2xj6zQDRQ6Nd08SncwBkIDC40G6YDTVk"
genai.configure(api_key="AIzaSyA2xj6zQDRQ6Nd08SncwBkIDC40G6YDTVk")
model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key="GOOGLE_API_KEY",temperature=0.2,convert_system_message_to_human=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
app = FastAPI()
class Item(BaseModel):
    pdf_id: str
    user_id: str 
    data: str


class question(BaseModel):
    pdf_id: str
    user_id: str
    query:str
    toxic_check:bool

@app.post("/qna/")
async def upload_file(item: question):
    
    #if classifier(item.query)[0]['label']=="INJECTION":
     #   return {"response":"prompt_injection"}
    if item.toxic_check:
        if toxic_classifier(item.query)[0]['score']>=0.6:
          return {"response":(item.query)[0]['label']}
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    directory = f"./chroma_db/{item.user_id}/{item.pdf_id}"
    vector_index = Chroma(persist_directory=directory, embedding_function=embeddings).as_retriever()
    prompt = PromptTemplate(
    input_variables=["question"],
    template="I am providing you some guideline make sure the given question follow the given guildlines answer only in yes and no words\
       - make sure it don't contain the word 'game' \
       {question}?",)
    from langchain.chains import LLMChain
    chain = LLMChain(llm=model, prompt=prompt, verbose=True)

    if chain.run(item.query).lower()=="no":
        return {"response":"don't follow provided guildline "}
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
    qa_chain = RetrievalQA.from_chain_type(model,retriever=vector_index,return_source_documents=True,chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
   
    result = qa_chain({"query": item.query})
    return {"response": result["result"]}
    

@app.post("/upload/")
async def upload_file(item: Item):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 0)
    splitted_text = text_splitter.split_text(item.data)
    directory = f"./chroma_db/{item.user_id}/{item.pdf_id}"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    # create the open-source embedding function
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_texts(splitted_text, embeddings,persist_directory=directory).as_retriever()
    
    return {"response":"saved"}


@app.get("/")
async def root():
    return {"message": "Hello World"}
#uvicorn back:app --reload  
#docker compose up -d --build  --scale server=1