from typing import Tuple

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from output_parsers import summary_parser, Summary
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

def ice_break_with(name:str)-> Tuple[Summary, str]:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username, mock=True)

    summary_template = """
    You are an assistant that outputs structured JSON.

    ONLY respond with a valid JSON object using the structure:
    {{
      "summary": "A brief summary of the person.",
      "facts": ["Fact 1", "Fact 2"]
    }}

    Do not include any explanations, headers, or notes.

    Here is the LinkedIn data:

    {information}

    REMEMBER: Your entire output must be valid JSON. No markdown, no code blocks, no extra text.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
    )

    # llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    llm = ChatOllama(model="llama3.2", temperature=0)
    chain = summary_prompt_template | llm | summary_parser

    res: Summary = chain.invoke(input={"information": linkedin_data})

    return res, linkedin_data.get("profile_pic_url")

if __name__ == "__main__":
    load_dotenv()
    print("ICE BREAKER")
    ice_break_with(name="Eden Marco Udemy")



