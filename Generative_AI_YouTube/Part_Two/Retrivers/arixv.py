from langchain_community.retrievers import ArxivRetriever

# Create the retriever
retriever = ArxivRetriever(
    load_max_docs=2,          # Number of papers to retrieve
    load_all_available_meta=True
)

# Query arXiv
docs = retriever.invoke("large language models")

# Print results
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("-" * 50)
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Published:", doc.metadata.get("Published"))
    print("Summary:")
    print(doc.page_content[:1000])   # Print first 1000 characters