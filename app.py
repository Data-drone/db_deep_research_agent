"""Databricks App entry point.

Reads DATABRICKS_APP_PORT and starts uvicorn with the FastAPI app.
"""

import os

import uvicorn

from deep_research.main import app

if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
