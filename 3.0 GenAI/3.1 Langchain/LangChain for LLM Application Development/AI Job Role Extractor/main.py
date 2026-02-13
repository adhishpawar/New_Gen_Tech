from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


# 1. Define output schema
response_schemas = [
    ResponseSchema(name="job_title", description="The title of the job"),
    ResponseSchema(name="skills", description="List of required skills"),
    ResponseSchema(name="experience", description="Years of experience required"),
    ResponseSchema(name="location", description="Job location"),
    ResponseSchema(name="salary", description="Salary range if available")
]


parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()


# 2. Prompt
template = """Extract the following information from the job description: {format_instructions} Job Description: {job_description} """

prompt = PromptTemplate(
    template=template,
    input_variables=["job_description"],
    partial_variables={"format_instructions": format_instructions}
)

formatted_prompt = prompt.format(job_description="""
We are looking for a Senior Backend Engineer to join our team in Pune. The role requires 5+ years of experience in Java, Spring Boot, and SQL. Remote work is allowed. Compensation is ₹15–18 LPA.
""")

## Model setup
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

response = llm.invoke(formatted_prompt)

result = parser.parse(response.content)

print("Result--->",result)