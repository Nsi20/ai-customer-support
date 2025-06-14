# nodes.py

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize Groq Chat model
llm = ChatGroq(model="mixtral-8x7b-32768")  # or "llama3-8b-8192"

# Common parser
parser = StrOutputParser()

# Categorize function
def categorize_node():
    prompt = ChatPromptTemplate.from_template(
        "Classify the user message into one of the categories: [Product, Billing, Technical, General].\n\nMessage: {message}"
    )
    return prompt | llm | parser

# Answer product questions
def product_support_node():
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant answering product-related queries. Question: {question}"
    )
    return prompt | llm | parser

# Answer billing questions
def billing_support_node():
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant answering billing-related queries. Question: {question}"
    )
    return prompt | llm | parser

# Answer technical questions
def technical_support_node():
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant answering technical support questions. Question: {question}"
    )
    return prompt | llm | parser

# General fallback
def general_support_node():
    prompt = ChatPromptTemplate.from_template(
        "Answer the user's question as clearly as possible. Question: {question}"
    )
    return prompt | llm | parser
