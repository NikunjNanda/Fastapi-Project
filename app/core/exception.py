from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_exception_handlers(app: FastAPI):

    async def unhandled_exception_handler(
        request: Request,
        exc: Exception
    ):
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc)
            }
        )

    app.add_exception_handler(
        Exception,
        unhandled_exception_handler
    ) 