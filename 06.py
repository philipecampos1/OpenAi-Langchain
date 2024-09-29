import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


OPEN_AI_API = os.environ['OPEN_AI_API']

model = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPEN_AI_API
)


classification_chain = (
    PromptTemplate.from_template(
        '''
        Classify a question from the user for one of this sectors:
        - Finance
        - IT Support
        - Other information

        question: {question}
        '''
    )
    | model
    | StrOutputParser()
)


financial_chain = (
    PromptTemplate.from_template(
        '''
    You are an speacilyst in finance.
    Always anser a question starting with a "Welcome to the finance Sector".
    Answer user question:
    question: {question}
    '''
    )
    | model
    | StrOutputParser()
)


tech_support_chain = (
    PromptTemplate.from_template(
        '''
        You are an speacilyst in IT support.
        Always anser a question starting with a "Welcome to the IT support Sector".
        Answer user question:
        question: {question}
        '''
    )
    | model
    | StrOutputParser()
)

other_info_chain = (
    PromptTemplate.from_template(
        '''
        You are an speacilyst in General information.
        Always anser a question starting with a "Welcome to the General information".
        Answer user question:
        question: {question}
        '''
    )
    | model
    | StrOutputParser()
)


def route(classification):
    classification = classification.lower()
    if 'finance' in classification:
        return financial_chain
    elif 'support' in classification:
        return tech_support_chain
    else:
        return other_info_chain


question = input('What is your question ? ')

classification = classification_chain.invoke(
    {'question': question}
)

response_chain = route(classification=classification)

response = response_chain.invoke(
    {'question': question}
)

print(response)
