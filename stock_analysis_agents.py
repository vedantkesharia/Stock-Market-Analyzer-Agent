from crewai import Agent
from langchain_openai import ChatOpenAI
import os
from langchain_openai import AzureChatOpenAI
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from tools.sec_tools import SECTools

# from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

from dotenv import load_dotenv
load_dotenv()
class StockAnalysisAgents():
    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_version = "2024-05-01-preview",
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            
        )

    def financial_analyst(self):
        return Agent(
            model=os.getenv("MODEL"),
            role='The Best Financial Analyst',
            goal="""Impress all customers with your financial data 
      and market trends analysis""",
            backstory="""The most seasoned financial analyst with 
      lots of expertise in stock market analysis and investment
      strategies that is working for a super important customer.""",
            verbose=True,
            tools=[
                SearchTools.search_internet,
                CalculatorTools.calculate,
                SECTools.search_10q,
                SECTools.search_10k
            ],
            llm=self.llm
        )

    def research_analyst(self):
        return Agent(
            model=os.getenv("MODEL"),
            role='Staff Research Analyst',
            goal="""Being the best at gather, interpret data and amaze
      your customer with it""",
            backstory="""Known as the BEST research analyst, you're
      skilled in sifting through news, company announcements, 
      and market sentiments. Now you're working on a super 
      important customer""",
            verbose=True,
            tools=[
                SearchTools.search_internet,
                SearchTools.search_news,
                YahooFinanceNewsTool(),
                SECTools.search_10q,
                SECTools.search_10k
            ],
            llm=self.llm
        )

    def investment_advisor(self):
        return Agent(
            model=os.getenv("MODEL"),
            role='Private Investment Advisor',
            goal="""Impress your customers with full analyses over stocks
      and completer investment recommendations""",
            backstory="""You're the most experienced investment advisor
      and you combine various analytical insights to formulate
      strategic investment advice. You are now working for
      a super important customer you need to impress.""",
            verbose=True,
            tools=[
                SearchTools.search_internet,
                SearchTools.search_news,
                CalculatorTools.calculate,
                YahooFinanceNewsTool()
            ],
            llm=self.llm
        )
