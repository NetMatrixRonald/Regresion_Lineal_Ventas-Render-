#!/usr/bin/env python3
"""
Punto de entrada para Render
"""

from app import app as flask_app
from asgiref.wsgi import WsgiToAsgi

# Convertir aplicación Flask (WSGI) a ASGI y exportar como 'app'
app = WsgiToAsgi(flask_app)
