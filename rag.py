from langchain_community.document_loaders import PyPDFLoader
loader =PyPDFLoader("./1.pdf")
docs=loader.load()

tables=[]
texts=[d.page_content for d in docs]
print(texts[0])
print(len(texts))
