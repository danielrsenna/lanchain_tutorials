# Baseado nos vídeos da playlist https://www.youtube.com/watch?v=ekpnVh-l3YA&list=PL4HikwTaYE0GEs7lvlYJQcvKhq0QZGRVn&ab_channel=LeonvanZyl
from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

memory 

chain = prompt | model

msg = {
    "input":"Hello"
}

response = chain.invoke(msg)
print(response.content)
