import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, tool, AgentExecutor
from langchain.prompts import PromptTemplate
from tools import rag_tool, calculator_tool, joke_tool

load_dotenv()

# Define prompt with BOTH 'input' and 'agent_scratchpad'
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
                You are a helpful AI assistant. You can use the available tools to answer the user's question.
                When you need to solve a problem, reason step-by-step and call the correct tool.

                User question: {input}

                {agent_scratchpad}
            """
)

# LLM
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    model="gpt-4o-mini"
)

# Tools list
tools = [rag_tool, calculator_tool, joke_tool]

# Create agent
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_agent(user_input: str):
    """Run the AI agent with a given user input."""
    return agent_executor.invoke({"input": user_input})
