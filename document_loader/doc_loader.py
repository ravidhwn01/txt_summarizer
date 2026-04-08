# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader('pdfs/r1.pdf')

# docs = loader.load()

# print(len(docs))

# print("page content",   docs[0].page_content)
# print("metadata", docs[1].metadata)



from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='pdfs',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)