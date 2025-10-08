from secret import OPENAI_API_KEY
from openai import OpenAI
import asyncio
import os

def basic_test(client):
    response = client.responses.create(
        model="gpt-5-nano",
        input="안녕"
    )
    print(response.output_text)

def websearch_test(client):
    response = client.responses.create(
        model="gpt-5-nano",
        tools=[{"type": "web_search"}],
        input="k8s에서 cpu request를 limit 없이 쓰는 경우, 어떤 전략을 사용하는 것이 좋을지 알려줘"
    )
    print(response.output_text)
'''
# It needs AVX/AVX2 instruction set
from agents import Agent, Runner
spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)

async def triage_agent_test():

    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)
'''

if __name__ == '__main__':
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    print('Open API test')
    #client = OpenAI()
    # basic_test(client)
    websearch_test(client)
    # asyncio.run(triage_agent_test())

