from typing import Union, List
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, Tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


from callbacks import AgentCallbackHandler
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
load_dotenv()

# 🔧 Tools 정의
@tool
def get_text_length(text: str) -> int:
    """Return the number of characters in a given text."""
    return len(text.strip("'\n").strip('"'))

@tool
def get_word_count(text: str) -> int:
    """Return the number of words in a given text."""
    return len(text.split())

@tool
def reverse_text(text: str) -> str:
    """Return the text in reverse order."""
    return text[::-1]

# 🔍 Tool 찾기
def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} is not found")

# 🚀 실행
if __name__ == "__main__":
    tools = [get_text_length, get_word_count, reverse_text]

    # 📜 프롬프트 (llama3에 맞게 \n 없이 정리)
    template = """You are a helpful assistant that can use tools.

You have access to the following tools:

{tools}

When you are asked a question, think step-by-step and decide what to do.

Use the following format:

Question: the input question you must answer
Thought: your reasoning about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
{agent_scratchpad}
"""

    # 🧠 프롬프트 생성
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # OpenAI LLM 설정
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        stop=["\nObservation:"],
        timeout=60,
        callbacks=[AgentCallbackHandler()],
    )

    # 🧠 Agent 구성
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # 💬 질문 입력
    question = "What is the text length of ORANGE in characters?"
    intermediate_steps = []
    agent_step = None

    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke({
            "input": question,
            "agent_scratchpad": intermediate_steps
        })

        print(f"\n🧠 Agent Step:\n{agent_step}")

        if isinstance(agent_step, AgentAction):
            tool = find_tool_by_name(tools, agent_step.tool)
            observation = tool.func(agent_step.tool_input)
            print(f"🔧 Observation: {observation}")
            intermediate_steps.append((agent_step, str(observation)))

    # ✅ 최종 결과 출력
    if isinstance(agent_step, AgentFinish):
        print(f"\n✅✅✅✅✅ Final Answer: {agent_step.return_values}")
