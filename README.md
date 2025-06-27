# 🧱 Minecraft 3D Pixel Art Generator

Una aplicación web interactiva que convierte cualquier imagen en arte pixelado 3D estilo Minecraft, respetando los colores originales de la imagen.

![Minecraft 3D Art](https://img.shields.io/badge/Minecraft-3D%20Art-green?style=for-the-badge&logo=minecraft)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## ✨ Características

### 🎯 Paleta Adaptiva
- **Extracción inteligente de colores**: Utiliza K-means clustering para identificar los colores dominantes de tu imagen
- **Ajuste automático**: Adapta los colores al estilo visual de Minecraft manteniendo la esencia original
- **Preservación de identidad**: Mantiene la apariencia visual característica de tu imagen

### 🧊 Efecto 3D Realista
- **Bloques con profundidad**: Cada pixel se convierte en un bloque 3D con caras superior, frontal y lateral
- **Iluminación consistente**: Sistema de iluminación que simula la estética de Minecraft
- **Sombras dinámicas**: Efecto de profundidad opcional para mayor realismo

### ⚡ Control Total
- **Modo dual de paletas**: Elige entre paleta adaptiva o paleta clásica de Minecraft
- **Configuración personalizable**: Ajusta el nivel de pixelado y tamaño de bloques
- **Visualización de paleta**: Muestra los colores utilizados en la conversión
- **Descarga de alta calidad**: Exporta el resultado en formato PNG

## 🚀 Instalación

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación rápida

1. **Clona el repositorio**
```bash
git clone https://github.com/tu-usuario/minecraft-3d-pixelator.git
cd minecraft-3d-pixelator
```

2. **Crea un entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instala las dependencias**
```bash
pip install -r requirements.txt
```

4. **Ejecuta la aplicación**
```bash
streamlit run minecraft.py
```

5. **Abre tu navegador** en `http://localhost:8501`

## 📋 Uso

### Paso a paso

1. **Sube tu imagen**: Usa el panel lateral para cargar archivos PNG, JPG o JPEG
2. **Configura los parámetros**:
   - **Tamaño del pixel**: Controla el nivel de pixelado (8-32)
   - **Tamaño del bloque 3D**: Ajusta el tamaño de cada bloque (15-30)
   - **Respetar colores originales**: Activa para usar paleta adaptiva
   - **Efecto de profundidad**: Añade sombras para mayor realismo
3. **Visualiza el resultado**: La imagen se procesa automáticamente
4. **Descarga tu creación**: Usa el botón de descarga para guardar el resultado

### Consejos para mejores resultados

- **Imágenes con contrastes claros** funcionan mejor
- **Resoluciones medianas** (500-1000px) ofrecen el mejor balance calidad/velocidad
- **Prueba diferentes niveles de pixelado** para encontrar el estilo que prefieras
- **El modo de paleta adaptiva** es ideal para retratos y paisajes
- **La paleta clásica** funciona mejor para elementos arquitectónicos

## 🛠️ Tecnologías Utilizadas

- **[Streamlit](https://streamlit.io/)**: Framework para aplicaciones web de datos
- **[PIL/Pillow](https://pillow.readthedocs.io/)**: Procesamiento de imágenes
- **[NumPy](https://numpy.org/)**: Computación numérica
- **[scikit-learn](https://scikit-learn.org/)**: Machine learning para clustering de colores
- **[colorsys](https://docs.python.org/3/library/colorsys.html)**: Conversión entre espacios de color

## 🎨 Cómo Funciona

### Algoritmo de Conversión

1. **Análisis de color**: Extrae los colores dominantes usando K-means clustering
2. **Adaptación de paleta**: Ajusta colores al estilo Minecraft manteniendo la identidad visual
3. **Pixelado**: Reduce la resolución usando interpolación nearest-neighbor
4. **Generación 3D**: Convierte cada pixel en un bloque 3D con:
   - Cara superior (más clara)
   - Cara frontal (color original)
   - Cara lateral (más oscura)
5. **Post-procesado**: Añade sombras y efectos de profundidad

### Paleta de Colores

La aplicación ofrece dos modos:

**Modo Adaptivo** (Recomendado):
- Analiza tu imagen para extraer colores representativos
- Ajusta automáticamente al estilo Minecraft
- Preserva la identidad visual original

**Modo Clásico**:
- Usa la paleta tradicional de Minecraft
- 16 colores básicos característicos del juego
- Ideal para un look más auténtico

## 📊 Estructura del Proyecto

```
minecraft-3d-pixelator/
│
├── minecraft.py              # Aplicación principal
├── requirements.txt          # Dependencias
├── README.md                # Documentación
│
└── examples/                # Imágenes de ejemplo (opcional)
    ├── input/
    └── output/
```

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si tienes ideas para mejorar el proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

### Ideas para contribuir

- [ ] Soporte para más formatos de imagen
- [ ] Nuevas paletas temáticas
- [ ] Optimización de rendimiento
- [ ] Modo batch para múltiples imágenes
- [ ] Efectos adicionales de post-procesado
- [ ] Integración con APIs de Minecraft

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## 🐛 Reporte de Problemas

Si encuentras algún bug o tienes sugerencias:

1. Revisa si ya existe un issue similar
2. Abre un nuevo issue con:
   - Descripción detallada del problema
   - Pasos para reproducirlo
   - Información del sistema (Python, OS)
   - Screenshots si es posible

## 🙏 Agradecimientos

- Inspirado en la estética visual de Minecraft
- Comunidad de Streamlit por la excelente documentación
- Contribuidores de las librerías de código abierto utilizadas

## 📈 Roadmap

- [ ] **v2.0**: Soporte para animaciones GIF
- [ ] **v2.1**: Paletas personalizables por el usuario
- [ ] **v2.2**: Exportación a formatos 3D (.obj, .stl)
- [ ] **v2.3**: Integración con Minecraft mods
- [ ] **v3.0**: Editor de bloques incorporado

---

⭐ **¡Si te gusta este proyecto, no olvides darle una estrella!** ⭐

*Creado con ❤️ para la comunidad de Minecraft y entusiastas del pixel art*