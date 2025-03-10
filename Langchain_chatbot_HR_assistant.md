```python
## HR chatbot assistant using Langchain and Python functions
```


```python
# setup Enviornment
__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules['pysqlite3']
```


```python
#Import necessary libraries
import os
import openai
import sys
```


```python
#Using PyPDF
from langchain.document_loaders import PyPDFLoader

Doc_loader = PyPDFLoader("Nestle_HR_policies.pdf")
extracted_text = Doc_loader.load()
```


```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter  = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
splitted_text=text_splitter.split_documents(extracted_text)
```


```python
# Embed the text and save it in vector stores
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```


```python
from langchain.vectorstores import Chroma
persist_directory = "chroma_vector_nestle"

vectordb = Chroma.from_documents(
    documents=splitted_text,
    embedding=embeddings,
    persist_directory=persist_directory
)
```


```python
#creat retrieval function
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

```


```python
from langchain.chains import RetrievalQA
Retriever_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       )
```


```python
import gradio as gr

import time
with gr.Blocks() as demo:
    
    def close():
            demo.clear()
            demo.close()

    def Enter_query (query,history):

        if query.strip() == "":
            return 'Please enter your question'
            # Get the answer from the chain
        start = time.time()

        res=Retriever_chain(query)
        end = time.time()

        return "\n\n> Question:" + query + f"\n> Answer (took {round(end - start, 2)} s.):"+res['result']
    chatbot=gr.ChatInterface(fn=Enter_query,type="messages")
    btn=gr.Button('Exit')     
    btn.click(fn=close, inputs=None, outputs=None)
    

demo.launch(share=True)


```

    * Running on local URL:  http://127.0.0.1:7860
    * Running on public URL: https://f315082068b3cd45f8.gradio.live
    
    This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)
    


<div><iframe src="https://f315082068b3cd45f8.gradio.live" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    




```python

```
