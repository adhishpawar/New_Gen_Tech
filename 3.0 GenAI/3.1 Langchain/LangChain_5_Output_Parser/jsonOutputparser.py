from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatOpenAI()

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age, city of a Fictional person \n {format_intruction}',
    input_variables=[],
    partial_variables={'format_intruction':parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({})   #req to send the dict --> else " RunnableSequence.invoke() missing 1 required positional argument: 'input'"
print(result)

 