"""Remote HTTP boundary for the future Qwen implementation."""

from __future__ import annotations


def create_app():
    """Create the service without importing or loading a model."""

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("install the gpu dependencies to run this service") from exc

    app = FastAPI(title="Active OCR GPU Service", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": "not_loaded"}

    def unavailable(kind: str):
        async def handler() -> None:
            raise HTTPException(status_code=501, detail=f"{kind} is not implemented yet")

        return handler

    for kind in ("train", "predict", "evaluate"):
        app.add_api_route(
            f"/jobs/{kind}",
            unavailable(kind),
            methods=["POST"],
            name=f"submit_{kind}",
        )

    @app.get("/jobs/{job_id}")
    async def job(job_id: str) -> None:
        raise HTTPException(status_code=501, detail="job execution is not implemented yet")

    return app


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("install the gpu dependencies to run this service") from exc
    uvicorn.run(create_app(), host="0.0.0.0", port=9000)


if __name__ == "__main__":
    main()
