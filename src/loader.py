import time
from typing import List

import dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from utils import url_to_md

dotenv.load_dotenv()

def load_urls(filename):
    lines = []
    with open(f'web_urls/{filename}', 'r') as file:
        # Read all lines of the file into a list
        lines = [line.strip() for line in file.readlines()]
    return lines

class URLtoMDLoader(BaseLoader):
    def __init__(self, url: str):
        self.url = url
    def load(self) -> List[Document]:
        text = url_to_md(self.url)
        metadata = {"source": self.url}
        return [Document(page_content=text, metadata=metadata)]
    
def load_url_file(filename: str, force_reload:bool = False):
    docembeddings = None
    if not force_reload:
        try:
            docembeddings = FAISS.load_local(f'vectorstore/llm_{filename}_index',OpenAIEmbeddings())
        except Exception as e:
            pass
    if not docembeddings:
        urls = load_urls(filename)
        external_docs = []
        for url in urls:
            loader = URLtoMDLoader(url)
            external_docs.extend(loader.load())
            time.sleep(3)
        chunk_size_value = 1000
        chunk_overlap = 100
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_value, chunk_overlap=chunk_overlap,length_function=len)
        texts = text_splitter.split_documents(external_docs)
        docembeddings = FAISS.from_documents(texts, OpenAIEmbeddings())
        docembeddings.save_local(f'vectorstore/llm_{filename}_index')
    return docembeddings

def load_external_links(force_reload:bool = False):
    return load_url_file('external.txt', force_reload)

def load_internal_links(force_reload:bool = False):
    return load_url_file('internal.txt', force_reload)

if __name__ == "__main__":
    # urls = load_urls('external.txt')
    # for url in urls:
    #     # url_to_md(url)
    #     print(URLtoMDLoader(url).load())

    docembeddings = load_external_links()
    docembeddings = load_internal_links()

    query = "What's the Classroom Conduct of MPCS?"
    relevant_chunks = docembeddings.similarity_search_with_score(query, k=3)
    chunk_docs = [chunk[0] for chunk in relevant_chunks]
    doc_text = '> '+'\n\n> '.join([doc.page_content for doc in chunk_docs])+'\n\n'+\
        'Source:\n- '+'\n- '.join(set([doc.metadata.get('source') for doc in chunk_docs]))
    print(doc_text)
    





