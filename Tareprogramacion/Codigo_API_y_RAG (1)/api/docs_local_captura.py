"""Solo para la captura: sirve Swagger UI desde archivos locales (sin CDN)."""
import sys
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from main import app

SW = sys.argv[1]
app.mount("/sw", StaticFiles(directory=SW), name="sw")
app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", "") != "/docs"]


@app.get("/docs", include_in_schema=False)
def docs():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Ori API - Swagger UI",
                               swagger_js_url="/sw/swagger-ui-bundle.js",
                               swagger_css_url="/sw/swagger-ui.css")
