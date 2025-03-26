from elsai_core.model import AzureOpenAIConnector
from elsai_core.llm_services import SummarizationService
from langchain.schema import HumanMessage, SystemMessage
import json
import feedparser
import urllib.parse
from datetime import datetime, timedelta
from newspaper import Article
import asyncio
import re
from pydantic import BaseModel, Field
from typing import List, Optional
from gpt_researcher import GPTResearcher
from googlenewsdecoder import new_decoderv1
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_SUBSCRIPTION_KEY = os.getenv("AZURE_SUBSCRIPTION_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

chat = AzureOpenAIConnector().connect_azure_open_ai("gpt-4o-mini")

class ContactInformation(BaseModel):
    email: Optional[str] = Field(None, description="Company email address")
    phone: Optional[str] = Field(None, description="Company phone number")
    website: Optional[str] = Field(None, description="Company website URL")
    
    def validate_email(cls, v):
        return v if v and "@" in v and "." in v else None
 
    def validate_website(cls, v):
        return v if v and ("http://" in v or "https://" in v) else None
 
class CompanyInformation(BaseModel):
    primary_address: str
    registration_number: Optional[str] = "Information not available"
    legal_form: str
    country: str
    town: str
    registration_date: str
    contact_information: ContactInformation
    general_details: str
    ubo: Optional[str]
    directors_shareholders: List[str]
    subsidiaries: Optional[str]
    parent_company: Optional[str]
    last_reported_revenue: str
 
async def generate_evidence(query: str):

    try:
        researcher = GPTResearcher(query=query, report_type="research_report", config_path=None)
        await researcher.conduct_research()
        report = await researcher.write_report()
        return report.split("## References")[0].strip()

    except Exception as e:
        return f"Error during research: {str(e)}"
 

def final_output_generation(llm, report):
    try:
        structured_llm = llm.with_structured_output(CompanyInformation)
        return structured_llm.invoke(report)

    except Exception as e:
        print(f"Error processing company information: {str(e)}")
        return None

def fetch_company_data(company_name: str, country: str):
    
    if country == "":
        query = f"Provide details for {company_name} including registration, address, legal form, UBO, shareholders, subsidiaries, and revenue."
    else:
        query = f"Provide details for {company_name} in {country} including registration, address, legal form, UBO, shareholders, subsidiaries, and revenue."
    print("Fetching details, please wait...")
    report = asyncio.run(generate_evidence(query=query))
    result = final_output_generation(chat, report)
    return result

def generate_search_queries(company, country, data_dir):
    try:
        if country == "":
            prompt = f"""
            For the given company, {company}, use {data_dir["Details"]} to generate search queries such that it can be used in TAVILY to generate detailed information about the company for adverse media screening and generate corporate actions keywords and adverse media keywords related to this company as separate lists which will be used for classification
            OUTPUT in JSON format:\n"""  + """
            {
                'search_queries': ['query1', 'query2'],
                'corporate_actions':['keyword1', 'keyword2'],
                'adverse_media':['keyword1', 'keyword2']
            }
            """
        else:
            prompt = f"""
            For the given company, {company} in {country}, use {data_dir["Details"]} to generate search queries such that it can be used in TAVILY to generate detailed information about the company for adverse media screening and generate corporate actions keywords and adverse media keywords related to this company as separate lists which will be used for classification
            OUTPUT in JSON format:\n"""  + """
            {
                'search_queries': ['query1', 'query2'],
                'corporate_actions':['keyword1', 'keyword2'],
                'adverse_media':['keyword1', 'keyword2']
            }
            """
        messages = [
                    SystemMessage(content = "You are a search query and keyword generator for adverse media screening."),
                    HumanMessage(content = prompt)
                ]
                
        response = chat(messages)
        result = response.content
        result = result.replace("```json", "").replace("```", "")
        queries = json.loads(result)
        print("Generated search queries successfully.")
        return queries

    except Exception as e:
        print(f"Error generating search queries: {str(e)}")
        return None
    
def find_tag(content, corporate_actions, adverse_media= []):
    query = f"""Find the tag from the following list related to the given company in the provided content. If not found, return an empty list.
    Tags: {corporate_actions}, {adverse_media}
    Content: {content}""" + """
    
    OUTPUT IN JSON format:
    {
        "tags"= [tag1, tag2]
    }
    
    or 
    {   
        "tags"= []
    }
    """
    messages = [
                SystemMessage(content = "You are a tag finder."),
                HumanMessage(content = query)
            ]
                
    response = chat(messages)
    result = response.content
    result = result.replace("```json", "").replace("```", "")
    tags = json.loads(result)
    return tags["tags"]

def news_articles(search_queries, df, company, corporate_actions, adverse_media, fromDate, toDate, max_results):
    def fetch_news_urls(query, fromDate, toDate, num_results=max_results):
        """
        Fetch news article URLs from a given search query using Google News RSS feed
        with a date filter for the last 5 years
        
        Args:
            query (str): The search query
            num_results (int): Maximum number of results to return
            years_back (int): How many years back to search for articles
            
        Returns:
            list: List of article URLs and their publication dates
        """
        end_date = toDate
        start_date = fromDate
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        date_query = f"{query} after:{start_date_str} before:{end_date_str}"
        encoded_query = urllib.parse.quote(date_query)
        
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(rss_url)
            results = []
            for entry in feed.entries[:num_results]:

                if 'link' in entry:
                    match = re.search(r'url=([^&]+)', entry.link)
                    if match:
                        actual_url = urllib.parse.unquote(match.group(1))
                        url = actual_url
                    else:
                        url = entry.link
                                            
                    pub_date = entry.get('published', 'Date unknown')                    
                    results.append({
                        'url': url,
                        'title': entry.get('title', 'No title'),
                        'published': pub_date
                    })
            return results
        
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def get_content(url):
        try:
            decoded_url = new_decoderv1(url)
            if decoded_url.get("status"):
                link=decoded_url["decoded_url"]
                article = Article(link)
                article.download()
                article.parse()
                return([link, article.text])
            else:
                return ""
        except Exception as e:
            print(f"Cannot fetch content for {url}: {e}")
            return None
        
    def get_content_summary(content):
        try:
            summariser = SummarizationService(chat)
            result = summariser.summarize(content)                 
            return result
        
        except Exception as e:
            print(f"Error summarizing content: {e}")
            return None
        
    def get_sentiment(text, company):
        prompt = f"Analyze the sentiment of the given content related to {company} and classify it as 'Positive', 'Negative', or 'Neutral'. Classify it as negative only if the content reflects negatively on the company. Provide only one of these three labels as the response, without any additional text or explanations. Content: {text}"
        messages = [
                        SystemMessage(content = "You are a sentiment analysis AI."),
                        HumanMessage(content = prompt)
                    ]                    
        sentiment = chat(messages)
        return sentiment.content
    
    def check_content(content, company):
        prompt = f"""
        Analyze the content of the given text and check if the text is related to the company '{company}' or not. If it is not related, return empty string, else return the content as such. 
        Content: {content}
        
        OUTPUT FORMAT:
        {content} if it is related to '{company}', 
        "", otherwise
        """
        messages = [
                    SystemMessage(content = "You are a content analysis AI."),
                    HumanMessage(content = prompt)
                ]
        response = chat(messages)
        return response.content
    
    def news(search_query, company, df_news, visited_urls):
        try:
            article_urls = fetch_news_urls(search_query, fromDate, toDate)
        
            print(f"\nFound {len(article_urls)} news articles for '{search_query}':")
            i= 0
            for url in article_urls:
                result=get_content(url["url"])
                link=result[0]
                content=result[1]
                if content == "" or content == None:
                    continue
                if link in visited_urls:
                    continue
                visited_urls.add(link)
                summary=get_content_summary(content)
                summarised_content = check_content(summary, company)
                if summarised_content == '""':
                    continue
                sentiment= get_sentiment(summarised_content, company)
                if sentiment == "Positive":
                    tag = find_tag(content, corporate_actions, [])
                else:
                    tag = find_tag(content, corporate_actions, adverse_media)
                df_news.loc[i]= [link, summary, sentiment, tag]
                i+= 1
            return [visited_urls, df_news]
        except Exception as e:
            print(f"Error fetching news: {e}")
            return None
    
    visited_urls = set()
    for query in search_queries:
        try:
            df_news = pd.DataFrame(columns=["url", "content", "sentiment", "tags"])
            result = news(query, company, df_news, visited_urls)
            visited_urls = result[0]
            df_news = result[1]
            df = pd.concat([df, df_news], ignore_index=True)
        except Exception as e:
            print(f"Error processing news: {e}")
    return df

def articles(company, corporate_actions, adverse_media, max_results, fromDate, toDate):
    entities = [company]
    adverse_keywords = adverse_media
    non_adverse_keywords = ["growth", "recognition", "innovation", "sustainability", "CSR"]
    corporate_actions = corporate_actions
    
    def sentiment_analysis(final_analysis, company):
        if not final_analysis.strip():
            return "No recent mentions found."
    
        prompt = f"""
        Analyze the sentiment of the given content for {company} and classify it as 'Positive', 'Negative', or 'Neutral'. 
        Classify it as negative only if the content reflects negatively on the company.
    
        **Rules:**
        - Classify sentiment as **Positive, Negative, or Neutral**.
        - Provide a brief explanation of why this sentiment was assigned.
    
        **Text:**  
        {final_analysis}
    
        **Output JSON Format:**  
        {{
            "sentiment": "Positive" or "Negative" or "Neutral",
            "explanation": "Brief reason for classification"
        }}
        """
    
        messages = [
            SystemMessage(content="You are a sentiment analysis AI."),
            HumanMessage(content=prompt)
        ]
    
        response = chat(messages)
        return response.content
    
    def search_tavily_adverse(entity, toDate, fromDate):
        try:
            results = []
            if fromDate and toDate:
                date_filter = f" after:{fromDate} before:{toDate}"
            elif fromDate:
                date_filter = f" after:{fromDate}"
            elif toDate:
                date_filter = f" before:{toDate}"
            for keyword in adverse_keywords+non_adverse_keywords+corporate_actions:
                query = f"{entity} {keyword} {date_filter}"
                url = "https://api.tavily.com/search"
                payload = {
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "max_results": max_results
                }
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    results.extend(response.json().get("results", []))
                else:
                    print(f"Error searching Tavily for {query}: {response.status_code}")
            return results
        except Exception as e:
            print(f"Error searching Tavily: {e}")
            return []

    def analyze_with_gpt(text):
        try:
            summariser = SummarizationService(chat)
            result = summariser.summarize(text)                 
            return result
        
        except Exception as e:
            print(f"Error summarizing content: {e}")
            return None

    def adverse_media_screening(entities, company, fromDate, toDate):
        results = []
    
        for entity in entities:
            print(f"Searching for adverse media related to {entity}...")
            search_results = search_tavily_adverse(f"{entity} news", fromDate, toDate)
            visited_urls = set()
            for result in search_results:
                if result.get("url", "") in visited_urls:
                    continue
                visited_urls.add(result.get("url", ""))
                content = result.get("content", "")
                if content:
                    analysis = analyze_with_gpt(content)
                    if analysis == '""':
                        continue
                    sentiment = sentiment_analysis(analysis, company)
                    json_sentiment = json.loads(sentiment)
                    print(json_sentiment)
                    if json_sentiment["sentiment"] == "Positive":
                        tag = find_tag(content, corporate_actions, [])
                    else:
                        tag = find_tag(content, corporate_actions, adverse_media)
                    results.append({
                        "url": result.get("url", ""),
                        "content": analysis,
                        "sentiment": json_sentiment["sentiment"],
                        "tags": tag
                    })
    
        return pd.DataFrame(results)
    
    df = adverse_media_screening(entities, company, fromDate, toDate)
    return df
