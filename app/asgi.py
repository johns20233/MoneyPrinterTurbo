"""Application implementation - ASGI."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse as StarletteJSONResponse

from app.config import config
from app.models.exception import HttpException
from app.router import root_api_router
from app.utils import utils

# Rutas que no requieren API key: documentación Swagger/Redoc y descarga
# publica de videos ya generados (esas URLs se sirven directamente al
# usuario final, no forman parte del contrato backend-a-backend).
PUBLIC_PATH_PREFIXES = ("/docs", "/openapi.json", "/redoc", "/tasks")


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Valida el header X-API-Key contra la variable de entorno MPT_API_KEY.

    Si MPT_API_KEY no esta seteada, el middleware no bloquea nada (modo
    abierto). Esto evita romper despliegues existentes que todavia no
    configuraron la variable, pero se recomienda encarecidamente setearla
    en produccion.
    """

    async def dispatch(self, request: Request, call_next):
        api_key = os.getenv("MPT_API_KEY", "")

        if not api_key:
            return await call_next(request)

        if request.url.path.startswith(PUBLIC_PATH_PREFIXES):
            return await call_next(request)

        provided_key = request.headers.get("X-API-Key", "")
        if provided_key != api_key:
            return StarletteJSONResponse(
                status_code=401,
                content={"status": 401, "message": "invalid or missing API key"},
            )

        return await call_next(request)


@asynccontextmanager
async def application_lifespan(_: FastAPI):
    """集中处理 API 进程启动恢复和关闭日志。"""
    logger.info("startup event")

    # 跨平台发布由当前进程线程池执行，不会在服务重启后恢复。启动时把 Redis
    # 中确认已失去执行进程的活动状态收敛为失败，避免任务永久无法删除。
    from app.services import task as task_service

    task_service.recover_interrupted_cross_posts()
    try:
        yield
    finally:
        logger.info("shutdown event")


def exception_handler(request: Request, e: HttpException):
    return JSONResponse(
        status_code=e.status_code,
        content=utils.get_response(e.status_code, e.data, e.message),
    )


def validation_exception_handler(request: Request, e: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content=utils.get_response(
            status=400, data=e.errors(), message="field required"
        ),
    )


def get_application() -> FastAPI:
    """Initialize FastAPI application.

    Returns:
       FastAPI: Application object instance.

    """
    instance = FastAPI(
        title=config.project_name,
        description=config.project_description,
        version=config.project_version,
        debug=False,
        lifespan=application_lifespan,
    )
    instance.include_router(root_api_router)
    instance.add_exception_handler(HttpException, exception_handler)
    instance.add_exception_handler(RequestValidationError, validation_exception_handler)
    return instance


app = get_application()

# Autenticacion por API key (ver ApiKeyMiddleware arriba).
app.add_middleware(ApiKeyMiddleware)

# Configures the CORS middleware for the FastAPI app
cors_allowed_origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
origins = cors_allowed_origins_str.split(",") if cors_allowed_origins_str else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

task_dir = utils.task_dir()
app.mount(
    "/tasks", StaticFiles(directory=task_dir, html=True, follow_symlink=True), name=""
)

public_dir = utils.public_dir()
app.mount("/", StaticFiles(directory=public_dir, html=True), name="")
