from fastapi import FastAPI
from api.app_router import router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Qwen3-4B Text Generator API",
    version="1.0",
    description="🚀 API for generating text using Qwen3-4B model from Alibaba",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")





