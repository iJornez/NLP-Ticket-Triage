from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from src.infer.service import router as infer_router

def create_app() -> FastAPI:
    app = FastAPI(title="NLP Ticket Triage & Sentiment", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": "mock-0.1.0"}

    # redirect root to the Swagger UI
    @app.get("/")
    def root():
        return RedirectResponse(url="/docs")

    # avoid noisy 404s for browser favicon probe
    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    app.include_router(infer_router, prefix="")
    return app

app = create_app()
