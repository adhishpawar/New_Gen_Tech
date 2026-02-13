from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI

model2 = ChatAnthropic(model_name='claude-3')

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the Following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answer from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into the a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
After high school, I have many aspirations I want to achieve. I aim to attend both college and law school. Once I gain some experience in my career, I hope to settle down, marry, and start a family. While I desire a fulfilling career, raising a family is equally important to me. I’m uncertain about the future or whether these plans will materialize because life can be unpredictable. Many challenges may arise as I pursue my goals, particularly the demands of college and law school, which require extensive studying, hard work, and commitment to become a lawyer. However, with diligence and resolve, I intend to reach my aspirations, whatever they may be, and realize my full potentia
"""

result = chain.invoke({'text':text})




