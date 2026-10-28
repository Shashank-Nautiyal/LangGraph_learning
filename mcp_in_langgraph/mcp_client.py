from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()  

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

# MCP client for local FastMCP server
client = MultiServerMCPClient(
    {
        "demo_server": {
            "transport": "stdio",
            "command": "C:/Users/shash/.local/bin/uv.exe",
            "args": [
                "run",
                "--project",
                "D:/Ai-engineer-learning/MCP_learning/MCP_server/demo_MCP_server",
                "main.py",
            ],
            "cwd": "D:/Ai-engineer-learning/MCP_learning/MCP_server/demo_MCP_server",
        },
    }
)


# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():

    tools = await client.get_tools()

    print(tools)

    llm_with_tools = llm.bind_tools(tools)

    # nodes
    async def chat_node(state: ChatState):

        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)

    # defining graph and nodes
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # defining graph connections
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()

    return chatbot

async def main():

    chatbot = await build_graph()

    # running the graph
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="roll 3 dice ")]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())