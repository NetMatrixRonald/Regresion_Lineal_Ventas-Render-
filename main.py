#!/usr/bin/env python3
"""
Punto de entrada para Render
"""

from app import app
from asgiref.wsgi import WsgiToAsgi

# Convertir aplicación Flask (WSGI) a ASGI
asgi_app = WsgiToAsgi(app)
