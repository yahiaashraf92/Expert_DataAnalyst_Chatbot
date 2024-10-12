import os
from tavily import TavilyClient
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.tools.tavily_search import TavilySearchResults


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

search = TavilySearchAPIWrapper()
tavily_response_tool = TavilySearchResults(api_wrapper=search)
