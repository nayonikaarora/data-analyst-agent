from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from bs4 import BeautifulSoup
import requests
import json
import re
import numpy as np

app = FastAPI()

@app.post("/api/")
async def analyze_data(file: UploadFile):
    try:
        content = await file.read()
        task = content.decode("utf-8")

        if "wikipedia.org" in task.lower():
            return await handle_wikipedia_task(task)
        elif "indian high court" in task.lower():
            return await handle_indian_judgments_task(task)
        else:
            return JSONResponse(content={"error": "Unrecognized task."}, status_code=400)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Wikipedia handler
async def handle_wikipedia_task(task: str):
    url = re.search(r"https://en\.wikipedia\.org[^\s]+", task).group(0)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="wikitable")
    df = pd.read_html(str(table))[0]

    df.columns = [c.replace("\n", " ").strip() for c in df.columns]
    df["Worldwide gross"] = df["Worldwide gross"].replace('[\$,]', '', regex=True).astype(str)
    df["Worldwide gross"] = df["Worldwide gross"].str.extract(r"(\d+\.\d+|\d+)").astype(float)
    df["Year"] = pd.to_datetime(df["Year"], errors='coerce').dt.year

    q1 = df[(df["Worldwide gross"] >= 2000) & (df["Year"] < 2020)].shape[0]
    q2 = df[df["Worldwide gross"] > 1500].sort_values("Year").iloc[0]["Title"]
    df["Rank"] = pd.to_numeric(df["Rank"], errors='coerce')
    df = df.dropna(subset=["Rank", "Worldwide gross"])
    q3 = round(df["Rank"].corr(df["Worldwide gross"]), 6)

    fig, ax = plt.subplots()
    ax.scatter(df["Rank"], df["Worldwide gross"], alpha=0.6)
    z = np.polyfit(df["Rank"], df["Worldwide gross"], 1)
    p = np.poly1d(z)
    ax.plot(df["Rank"], p(df["Rank"]), "r--")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Gross (in M)")
    ax.set_title("Rank vs Peak Gross")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    img_uri = f"data:image/png;base64,{image_base64}"

    return [q1, q2, q3, img_uri]

# Court dataset handler
async def handle_indian_judgments_task(task: str):
    s3_url = "s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1"
    duckdb.sql("INSTALL httpfs; LOAD httpfs;")
    duckdb.sql("INSTALL parquet; LOAD parquet;")

    df = duckdb.sql(f"""
        SELECT court, year, decision_date, date_of_registration
        FROM read_parquet('{s3_url}')
        WHERE year BETWEEN 2019 AND 2022
    """).df()

    q1 = df["court"].value_counts().idxmax()

    df = df[df["court"] == "33_10"].copy()
    df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")
    df["date_of_registration"] = pd.to_datetime(df["date_of_registration"], errors="coerce")
    df["delay_days"] = (df["decision_date"] - df["date_of_registration"]).dt.days
    df = df.dropna(subset=["delay_days", "year"])
    slope, _ = np.polyfit(df["year"], df["delay_days"], 1)
    slope = round(slope, 3)

    fig, ax = plt.subplots()
    ax.scatter(df["year"], df["delay_days"], alpha=0.5)
    z = np.polyfit(df["year"], df["delay_days"], 1)
    p = np.poly1d(z)
    ax.plot(df["year"], p(df["year"]), "r--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Days of Delay")
    ax.set_title("Judgment Delay by Year")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    img_uri = f"data:image/png;base64,{image_base64}"

    result = {
        "Which high court disposed the most cases from 2019 - 2022?": q1,
        "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": slope,
        "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": img_uri
    }

    return result

@app.get("/")
def root():
    return {"message": "Data Analyst Agent API is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
