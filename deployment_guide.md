# 🚀 Guía de Despliegue - Minecraft 3D Pixelator

Esta guía te ayudará a solucionar los problemas comunes de despliegue cuando tu aplicación funciona en local pero no en producción.

## 📋 Archivos Necesarios para Despliegue

Asegúrate de tener todos estos archivos en tu repositorio:

```
minecraft-3d-pixelator/
├── minecraft.py              # Tu aplicación original
├── minecraft_fixed.py        # Versión mejorada para despliegue
├── requirements.txt          # Dependencias Python
├── packages.txt             # Dependencias del sistema (Linux)
├── runtime.txt              # Versión de Python
├── .streamlit/
│   └── config.toml          # Configuración de Streamlit
├── README.md
└── DEPLOYMENT_GUIDE.md
```

## 🔧 Problemas Comunes y Soluciones

### 1. Error de scikit-learn
**Problema**: `ModuleNotFoundError: No module named 'sklearn'`

**Solución**:
- Usa `minecraft_fixed.py` que incluye fallbacks
- Verifica que `requirements.txt` tenga versiones específicas
- Para Streamlit Cloud, asegúrate de que `packages.txt` incluya dependencias del sistema

### 2. Problemas de Memoria
**Problema**: La aplicación se cuelga con imágenes grandes

**Soluciones aplicadas en `minecraft_fixed.py`**:
- Límite máximo de 100x100 bloques
- Procesamiento progresivo de imágenes
- Manejo de errores robusto

### 3. Errores de PIL/Pillow
**Problema**: Errores al procesar imágenes

**Solución**:
- Versión específica de Pillow en requirements.txt
- Validación de formato de imagen
- Conversión automática a RGB

### 4. Problemas de Plataforma
**Problema**: Funciona en Windows/Mac pero no en Linux

**Solución**:
- `packages.txt` con dependencias del sistema Linux
- `runtime.txt` especifica versión exacta de Python

## 🌐 Despliegue en Streamlit Cloud

### Paso 1: Preparar Repositorio
```bash
git add .
git commit -m "Add deployment files and fixes"
git push origin main
```

### Paso 2: Configurar Streamlit Cloud
1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Conecta tu repositorio de GitHub
3. **Importante**: Especifica `minecraft_fixed.py` como archivo principal
4. Espera el despliegue (puede tomar 5-10 minutos)

### Paso 3: Verificar Configuración
- Revisa los logs de despliegue
- Verifica que todas las dependencias se instalen correctamente
- Testa la funcionalidad básica

## 🐳 Despliegue con Docker

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "minecraft_fixed.py", "--server.port=