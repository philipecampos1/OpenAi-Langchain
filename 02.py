import os
from langchain_openai import OpenAI
from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache


OPEN_AI_API = os.environ['OPEN_AI_API']


model = OpenAI(
    api_key=OPEN_AI_API,
)

'''
With this would be able to avoid spending tokens for the same
question but it only works until the your application is working
'''
# set_llm_cache(InMemoryCache())

'''Same idea as before but this time will be saving in a data base so even
if you stop your application ti will be saved to be use another time'''
set_llm_cache(
    SQLiteCache(database_path='openai_cache.db')
              )


prompt = 'Who was albert einstein'

response1 = model.invoke(prompt)
print(f'Chamada 1: {response1}')

response2 = model.invoke(prompt)
print(f'Chamda 2: {response2}')
