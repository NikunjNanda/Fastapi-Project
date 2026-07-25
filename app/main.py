from fastapi import FastAPI 
from prometheus_fastapi_instrumentator import Instrumentator 
from app.api import routes_auth , routes_predict 
from app.middleware.logging_middleware import LoggingMiddleware 
from app.core.exception import register_exception_handlers 

app = FastAPI(title='Car Price Prediction API') 

#Linking the middleware to this application 
app.add_middleware(LoggingMiddleware)

#Linkign the endpoints
app.include_router(routes_auth.router , tags=['Auth'])
app.include_router(routes_predict.router , tags=['Prediction']) 

#monitoring using prometheus 
Instrumentator().instrument(app).expose(app) 

#add exception handler
register_exception_handlers(app) 

