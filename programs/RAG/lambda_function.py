import os
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, SpacyTextSplitter,NLTKTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import json
from utils import get_secret

def lambda_handler(event, context):
    query = event['query']
    try:
        os.environ['OPENAI_API_KEY']=get_secret("openAPI_secret")['OPENAI_API_KEY']
        embeddings = OpenAIEmbeddings()
        # Load vectorised data from local DB
        docsearch_local = FAISS.load_local("faiss_index", embeddings)
        # The database has been created locally, querying on local DB.
        chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        docs = docsearch_local.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        return {
            'statusCode': 200,
            'body': json.dumps('{}'.format(response))
        }
    except Exception as e :
        return {
            'statusCode': 520,
            'body': json.dumps('{}'.format(e))
        }

