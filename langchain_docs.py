# Baseado em https://python.langchain.com/v0.2/docs/tutorials/chatbot/

# Pegando chaves
from dotenv import load_dotenv
load_dotenv()

#Importando e definindo modelo
from langchain_openai import ChatOpenAI #OpenAI chat model integration.
model = ChatOpenAI(
    model="gpt-3.5-turbo", #Name of OpenAI model to use.
    temperature=0.7 #Sampling temperature
)

#Importando bibliotecas
from langchain_core.messages import HumanMessage #Message from a human.
from langchain_core.messages import AIMessage #Message from an AI.
from langchain_core.chat_history import( #Chat message history stores a history of the message interactions in a chat.
    BaseChatMessageHistory, #Abstract base class for storing chat message history.
    InMemoryChatMessageHistory,# In memory implementation of chat message history. Stores messages in an in memory list.
)
from langchain_core.runnables.history import RunnableWithMessageHistory #Runnable that manages chat message history for another Runnable. A chat message history is a sequence of messages that represent a conversation. RunnableWithMessageHistory wraps another Runnable and manages the chat message history for it; it is responsible for reading and updating the chat message history.

store ={}

#session_id is used to distinguish between separate conversations, and should be passed in as part of the config when calling the new chain
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

from langchain_core.prompts import ChatPromptTemplate #Prompt template for chat models. Use to create flexible templated prompts for chat models.
from langchain_core.prompts import MessagesPlaceholder #Prompt template that assumes variable is already list of messages. A placeholder which can be used to pass in a list of messages.


from langchain_core.messages import SystemMessage, trim_messages

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer.invoke(messages)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc20"}}

response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)

print(response.content)