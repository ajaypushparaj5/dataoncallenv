"""OpenEnv-compatible server entry point.
Re-exports the FastAPI app from the dataoncallenv package.
"""
from dataoncallenv.api.app import app

__all__ = ["app"]

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()
