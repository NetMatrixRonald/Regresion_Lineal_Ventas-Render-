# Configuración de Uvicorn para Render

# Configuración del servidor
host = "0.0.0.0"
port = 10000
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"

# Configuración de rendimiento
loop = "auto"
http = "auto"
ws = "auto"
lifespan = "auto"

# Configuración de logs
log_level = "info"
access_log = True

# Configuración de timeouts
timeout_keep_alive = 30
timeout_graceful_shutdown = 30

# Configuración de seguridad
limit_concurrency = 1000
limit_max_requests = 1000
backlog = 2048
