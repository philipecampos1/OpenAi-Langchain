import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


OPEN_AI_API = os.environ['OPEN_AI_API']

model = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPEN_AI_API
)

# loader = TextLoader('test_chat_01.txt')
# documents = loader.load()

# loader = PyPDFLoader('Philipe Campos CV.pdf')
# documents = loader.load()

loader = CSVLoader('machine-readable-business-employment-data-Jun-2024-quarter.csv')
documents = loader.load()


prompt_knowledge_base = PromptTemplate(
    input_variables=['context', 'question'],
    template='''
        Using only the context to answer a question.
        Do not use infromation from other sources:
        context : {context}
        question : {question}
        '''
)

chain = prompt_knowledge_base | model | StrOutputParser()

response = chain.invoke(
    {
        'context': '\n'.join(doc.page_content for doc in documents),
        'question': 'Tell what is this file about'
    }
)


print(response)
