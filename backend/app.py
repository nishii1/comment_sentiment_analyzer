from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
import torch
from transformers import BertTokenizer, BertForSequenceClassification


load_dotenv()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "./sentiment-model" 
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

model.eval()


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise ValueError("API Key is missing. Set YOUTUBE_API_KEY in .env file.")

class VideoRequest(BaseModel):
    url: str

def extract_video_id(url: str) -> str:
    """Extracts the YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_comments(video_id: str):
    """Fetches comments from YouTube API."""
    URL = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={API_KEY}&maxResults=100"
    
    try:
        response = requests.get(URL)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to fetch comments: {response.json()}")
        
        data = response.json()
        return [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in data.get("items", [])]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_text(text: str) -> str:
    """Cleans and preprocesses comments for NLP analysis."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words) if words else None




def classify_comment(text: str) -> int:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()  # 0=negative, 1=neutral, 2=positive
    return predicted_class







@app.post("/analyze")
async def analyze_video_comments(request: VideoRequest):
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    comments = get_comments(video_id)
    if not comments:
        raise HTTPException(status_code=404, detail="No comments found")

    sentiment_results = {"positive": 0, "neutral": 0, "negative": 0}
    


    for comment in comments:
        cleaned_comment = preprocess_text(comment)
        if cleaned_comment:
            label = classify_comment(cleaned_comment)
            if label == 0:
                sentiment_results["negative"] += 1
            elif label == 1:
                sentiment_results["neutral"] += 1
            elif label == 2:
                sentiment_results["positive"] += 1

    return sentiment_results
