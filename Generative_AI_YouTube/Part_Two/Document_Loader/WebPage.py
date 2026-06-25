from langchain_community.document_loaders import WebBaseLoader

url = "https://arxiv.org/abs/1706.03762"

data = WebBaseLoader(url)

docs  = data.load()

print(docs[0].page_content)