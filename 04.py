import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


OPEN_AI_API = os.environ['OPEN_AI_API']

model = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPEN_AI_API
)

chate_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content='YOu need to answer based in geografic data from regions of Brazil'),
        HumanMessagePromptTemplate.from_template('Please, tell me about this region {region}.'),
        AIMessage(content='Of course, I will be collecting information about this region and analising the data'),
        HumanMessage(content='Make sure to include demographic'),
        AIMessage(content='Got it. This is the data:')
    ],
)

prompt = chate_template.format_messages(region='South')

response = model.invoke(prompt)

print(response.content)
