from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 10,
    chunk_overlap= 1
)

loader = TextLoader("notes.txt")

documents = loader.load()

chunks = splitter.split_documents(documents)

for i in chunks:
    print(i.page_content)
    print()
    print()
