from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv

load_dotenv() 

# Initialize the OpenAI LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Setup memory (shared globally across chains)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


#Prompt 1 --> Extract Intent
intent_template = PromptTemplate(
    input_variables=["input_text"],
    template="""
                    Extract the core intent from the user input below.
                    Text: {input_text}
                    Intent:
            """
)
intent_chain = LLMChain(llm=llm, prompt=intent_template)

##Promot 2 --> Summarize user input
summary_template = PromptTemplate(
    input_variables=["input_text"],
    template= """
                    Summarize the following message in 1-2 lines for quick understanding:
                    {input_text}
                    Summary:
              """
)
summary_chain = LLMChain(llm=llm, prompt=summary_template)

##Prompt 3---> Create final reply using both intent and summary
final_template = PromptTemplate(
    input_variables=["intent", "summary"],
    template="""
                    You are a helpful assistant in a multi-turn chat.

                    Previous Conversation:
                    {chat_history}

                    Based on:
                    - Intent: {intent}
                    - Summary: {summary}

                    Respond concisely and helpfully.

                    Response:
            """
)
final_chain = LLMChain(llm=llm, prompt=final_template)



def chat_with_memory(user_input):
    # Store the message into memory manually
    memory.chat_memory.add_user_message(user_input)

    # Step 1 ---> Intent
    intent = intent_chain.run({"input_text": user_input})

    # Step 2 ---> Summary
    summary = summary_chain.run({"input_text": user_input})

    # Step 3 ---> Final response with memory
    final_response = final_chain.run({
        "intent": intent,
        "summary": summary,
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })

    # Add AI response to memory
    memory.chat_memory.add_ai_message(final_response)

    return final_response