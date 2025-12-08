# Gunicorn配置文件
import multiprocessing
import os

# 服务器socket
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker进程
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count()))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 30
keepalive = 2

# 重启
max_requests = 1000
max_requests_jitter = 50
# 注意：对于GPU模型服务，preload_app应设为False
# 避免在fork前加载模型导致CUDA context共享问题
preload_app = False

# 日志
accesslog = "-"
errorlog = "-"
loglevel = os.getenv('LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 进程命名
proc_name = 'cy-model-server'

# 安全
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
