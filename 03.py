import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

OPEN_AI_API = os.environ['OPEN_AI_API']

model = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPEN_AI_API,
)

template = '''
Transalate text from {language1} to {language2}:
{text}
'''

prompt_template = PromptTemplate.from_template(
    template=template,
)

prompt = prompt_template.format(
    language1='portugues',
    language2='english',
    text='Bom dia!'
)

response = model.invoke(prompt)

print(response.content)
