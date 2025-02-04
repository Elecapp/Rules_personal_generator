
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from vessels_router import vessels_router
from covid_router import covid_router

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

metadata_tags = [
    {
        "name": "Vessels",
        "description": "API for vessels data"
    }
]
app = FastAPI(title="LORE API", description="API for vessels data and covid data",
              version="0.1", openapi_tags=metadata_tags)
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )

app.include_router(vessels_router)
app.include_router(covid_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Vessels API"}

