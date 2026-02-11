# In your FastAPI main.py file
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:8501",  # Allow your Streamlit app's address
    # Add the deployed URL of your Streamlit app here if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/your-endpoint")
def read_data():
    return {"message": "Data from FastAPI"}
