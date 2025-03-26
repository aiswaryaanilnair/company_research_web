import asyncio
import base64
import io
import json
import ast
import re
from datetime import date, datetime
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import List

# Import utilities from your existing file
from utilities import (
    generate_search_queries, 
    news_articles, 
    articles, 
    fetch_company_data
)

# Import models and functions from your existing app
from app import (
    generate_evidence,
    final_output_generation,
    reference_generation,
    format_company_data_as_dict,
    get_analysis_results,
    director_check,
    analyze_sentiment_by_tag,
    md_to_html
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Web Research Agent",
    description="A comprehensive corporate research and adverse media screening API"
)

# Initialize LLM
llm = ChatOpenAI(
    openai_api_base="https://api.openai.com/v1",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
)

# Request Model
class CompanyResearchRequest(BaseModel):
    company: str
    country: Optional[str] = None
    state: Optional[str] = None
    from_date: Optional[date] = None
    to_date: Optional[date] = None

# Response Model
class CompanyResearchResponse(BaseModel):
    company_data: dict
    references: dict
    web_research_results: List[dict]
    sentiment_analysis: dict
    markdown_content: str
    download_links: dict

@app.post("/research/company", response_model=CompanyResearchResponse)
async def perform_company_research(request: CompanyResearchRequest):
    try:
        # Validate input
        if not request.company:
            raise HTTPException(status_code=400, detail="Company name is required")
        
        # Set default dates if not provided
        from_date = request.from_date or date.today()
        to_date = request.to_date or date.today()
        
        if from_date > to_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        max_results = 2
        country = request.country or ""
        state = request.state or ""
        
        # Construct query based on input
        if request.country and request.state:
            query = f"Provide the {request.company} details specifically in {request.state},{request.country}, including registration number, primary address, legal form, country, town, registration date, contact info, general details, UBO, directors/shareholders, subsidiaries, parent company, and last reported revenue."
        elif request.country:
            query = f"Provide the {request.company} details specifically in {request.country}, including registration number, primary address, legal form, country, town, registration date, contact info, general details, UBO, directors/shareholders, subsidiaries, parent company, and last reported revenue."
        elif request.state:
            query = f"Provide the {request.company} details specifically in {request.state}, including registration number, primary address, legal form, country, town, registration date, contact info, general details, UBO, directors/shareholders, subsidiaries, parent company, and last reported revenue."
        else:
            query = f"Provide the {request.company} details, including registration number, primary address, legal form, country, town, registration date, contact info, general details, UBO, directors/shareholders, subsidiaries, parent company, and last reported revenue."
        
        # Perform research
        report, references = await generate_evidence(query=query)
        result = final_output_generation(llm, report)
        ref_dict = reference_generation(llm, references)
        
        # Format company data
        data_dict = format_company_data_as_dict(result)
                
        combined_data = {"Company_Data": data_dict, "References": ref_dict}
        with open(f"{request.company}_company_research.json", "w") as file:
            json.dump(combined_data, file, indent=4)
        
        # Generate search queries
        queries = generate_search_queries(request.company, country, data_dict)
        
        # Initialize DataFrame
        df = pd.DataFrame(columns=["url", "content", "sentiment", "tags"])
        
        # Perform web research
        corporate_actions = queries["corporate_actions"]
        adverse_media = queries["adverse_media"]
        df = news_articles(queries["search_queries"], df, request.company, corporate_actions, adverse_media, from_date, to_date, max_results)
        df_articles = articles(request.company, corporate_actions, adverse_media, max_results, from_date, to_date)
        df = pd.concat([df, df_articles], ignore_index=True)
        
        # Sentiment analysis
        sentiment_counts = df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        
        # Analyze content by sentiment
        positive_content = df[df['sentiment'] == 'Positive']['content'].tolist()
        negative_content = df[df['sentiment'] == 'Negative']['content'].tolist()
        neutral_content = df[df['sentiment'] == 'Neutral']['content'].tolist()
        
        positive_content = get_analysis_results(positive_content, request.company)
        negative_content = get_analysis_results(negative_content, request.company)
        neutral_content = get_analysis_results(neutral_content, request.company)
        
        # Director check
        content_list = df["content"].tolist()
        director_content = director_check(content_list, request.company, data_dict)
        
        # Sentiment by tag
        sent_df = analyze_sentiment_by_tag(df)
        
        # Prepare markdown content
        markdown_content = f"# Adverse Media Research Results\n\n"
        markdown_content += sentiment_counts.to_markdown(index=False)
        
        if positive_content != '""':
            markdown_content += "\n\n## Positive Media Keypoints:\n"
            markdown_content += positive_content
        
        if negative_content != '""':
            markdown_content += "\n\n## Negative Media Keypoints:\n"
            markdown_content += negative_content
        
        if neutral_content != '""':
            markdown_content += "\n\n## Neutral Media Keypoints:\n"
            markdown_content += neutral_content
        
        markdown_content += "\n\n## Sentiment Distribution by Category\n"
        markdown_content += sent_df.to_markdown(index=False)
        
        markdown_content += "\n\n## Directors Sanity Check\n"
        markdown_content += director_content
        
        # Prepare files
        output_prefix = request.company.replace(" ", "_")
        
        # Save files
        with open(f"{output_prefix}.md", "w") as f:
            f.write(markdown_content)

        output_file_name_json = f"{output_prefix}_web_research_results.json"
        output_file_name_excel = f"{output_prefix[:4]}_web_research_results.xlsx"

        json_df = df.to_json(output_file_name_json, orient="records", indent=4)
        excel_df = df.to_excel(output_file_name_excel, index=False)

        response = {
            "company_data": data_dict,
            "references": ref_dict,
            "web_research_results": df.to_dict(orient="records"),
            "sentiment_analysis": {
                "sentiment_counts": sentiment_counts.to_dict(orient="records"),
                "sentiment_by_tag": sent_df.to_dict()
            },
            "markdown_content": markdown_content,
            "download_links": {
                "json": output_file_name_json,
                "excel": output_file_name_excel,
                "markdown": f"{output_prefix}.md",
            }
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoints for file downloads
@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        return FileResponse(
            path=filename, 
            media_type='application/octet-stream', 
            filename=filename
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)