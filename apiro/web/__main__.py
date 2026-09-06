"""Main entry point for `python -m apiro.web`."""
import uvicorn
from apiro.web.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
