import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


OPEN_AI_API = os.environ['OPEN_AI_API']

model = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPEN_AI_API
)

prompt_template = PromptTemplate.from_template(
    'Tell me about this car {car}.'
)

# runnable_sequence = prompt_template | model | StrOutputParser()

# response = runnable_sequence.invoke({'car': 'Marea 20v 1999'})

# print(response)


runnable_sequence = (
    PromptTemplate.from_template(
        'Tell me about this car {car}.'
    )
    | model
    | StrOutputParser()
)

response = runnable_sequence.invoke({'car': 'Marea 20v 1999'})

print(response)
