from langchain_openai import OpenAI, ChatOpenAI
import os

OPEN_AI_API = os.environ['OPEN_AI_API']

# model = OpenAI(
#     api_key=OPEN_AI_API,
# )

# response = model.invoke(
#     input='Who was alan Turning?',
#     temperature=1,
#     max_tokens=500,

# )

# print(response)


model = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPEN_AI_API,
)


messages = [
    {'role': 'system', 'content': 'You are going to give information about historical figures'},
    {'role': 'user', 'content': 'Who was alan turning?'}
]

response = model.invoke(messages)

print(response)
print(response.content)
