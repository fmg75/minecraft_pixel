import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import io
import colorsys
from collections import Counter
import sys

# Manejo de importación de sklearn con fallback
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn no está disponible. Usando método alternativo para extracción de colores.")

@st.cache_data
def extract_dominant_colors_fallback(_image, num_colors=16):
    """Método alternativo sin sklearn para extraer colores dominantes"""
    # Redimensionar para acelerar
    temp_img = _image.resize((50, 50))
    pixels = list(temp_img.getdata())
    
    # Agrupar colores similares manualmente
    color_groups = {}
    tolerance = 30
    
    for pixel in pixels:
        if len(pixel) == 4:  # RGBA
            r, g, b = pixel[:3]
        else:  # RGB
            r, g, b = pixel
            
        # Buscar grupo existente
        found_group = False
        for group_color in color_groups:
            if (abs(r - group_color[0]) < tolerance and 
                abs(g - group_color[1]) < tolerance and 
                abs(b - group_color[2]) < tolerance):
                color_groups[group_color] += 1
                found_group = True
                break
        
        if not found_group:
            color_groups[(r, g, b)] = 1
    
    # Ordenar por frecuencia y tomar los más comunes
    sorted_colors = sorted(color_groups.keys(), key=lambda x: color_groups[x], reverse=True)
    return sorted_colors[:num_colors]

@st.cache_data
def extract_dominant_colors_sklearn(_image, num_colors=16):
    """Extrae colores dominantes usando sklearn"""
    if not SKLEARN_AVAILABLE:
        return extract_dominant_colors_fallback(_image, num_colors)
    
    try:
        # Redimensionar para acelerar
        temp_img = _image.resize((100, 100))
        pixels = np.array(temp_img).reshape(-1, 3)
        
        # Filtrar píxeles válidos
        pixels = pixels[~np.isnan(pixels).any(axis=1)]
        
        if len(pixels) == 0:
            return [(128, 128, 128)] * num_colors
        
        # K-means con configuración robusta
        kmeans = KMeans(
            n_clusters=min(num_colors, len(pixels)), 
            random_state=42, 
            n_init=10,
            max_iter=100
        )
        kmeans.fit(pixels)
        
        dominant_colors = kmeans.cluster_centers_.astype(int)
        
        # Obtener frecuencias
        labels = kmeans.labels_
        label_counts = Counter(labels)
        
        # Ordenar por frecuencia
        sorted_colors = []
        for i in sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True):
            color = tuple(np.clip(dominant_colors[i], 0, 255))
            sorted_colors.append(color)
        
        return sorted_colors
    
    except Exception as e:
        st.warning(f"Error en sklearn, usando método alternativo: {str(e)}")
        return extract_dominant_colors_fallback(_image, num_colors)

def rotate_image(image, angle):
    """Rota la imagen el ángulo especificado"""
    try:
        if angle == 0:
            return image
        
        # Rotar imagen expandiendo el canvas para evitar recortes
        rotated = image.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        return rotated
    except Exception as e:
        st.warning(f"Error al rotar imagen: {str(e)}")
        return image

def minecraft_color_adjustment(color):
    """Ajusta un color para que se vea más 'Minecraft'"""
    try:
        r, g, b = color
        
        # Asegurar valores válidos
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        
        # Convertir a HSV
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        
        # Ajustes para estilo Minecraft
        s = min(1.0, s * 1.15)  # Más saturado
        
        # Ajustar brillo
        if v < 0.2:
            v = 0.25
        elif v > 0.95:
            v = 0.9
        
        # Convertir de vuelta a RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        return (int(r * 255), int(g * 255), int(b * 255))
    
    except Exception:
        return color  # Devolver original si hay error

@st.cache_data
def create_adaptive_minecraft_palette(image_bytes, preserve_original=True):
    """Crea paleta adaptada usando bytes de imagen para cache"""
    image = Image.open(io.BytesIO(image_bytes))
    
    if preserve_original:
        # Extraer colores dominantes
        if SKLEARN_AVAILABLE:
            dominant_colors = extract_dominant_colors_sklearn(image, num_colors=12)
        else:
            dominant_colors = extract_dominant_colors_fallback(image, num_colors=12)
        
        # Ajustar para estilo Minecraft
        minecraft_adapted = []
        for color in dominant_colors:
            adjusted = minecraft_color_adjustment(color)
            minecraft_adapted.append(adjusted)
        
        # Colores básicos de Minecraft como respaldo
        basic_minecraft = [
            (34, 139, 34),    # Verde hierba
            (139, 69, 19),    # Marrón tierra
            (105, 105, 105),  # Gris piedra
            (255, 255, 255),  # Blanco
            (220, 20, 60),    # Rojo
        ]
        
        # Combinar evitando duplicados
        final_palette = minecraft_adapted.copy()
        for basic_color in basic_minecraft:
            # Verificar si es similar a algún color existente
            is_similar = False
            for existing in final_palette:
                distance = sum((a - b) ** 2 for a, b in zip(basic_color, existing))
                if distance < 2000:  # Umbral de similitud
                    is_similar = True
                    break
            
            if not is_similar and len(final_palette) < 16:
                final_palette.append(basic_color)
        
        return final_palette[:16]
    
    else:
        # Paleta fija clásica
        return [
            (34, 139, 34),    # Verde hierba
            (139, 69, 19),    # Marrón tierra
            (105, 105, 105),  # Gris piedra
            (160, 82, 45),    # Marrón madera
            (255, 215, 0),    # Dorado
            (220, 20, 60),    # Rojo ladrillo
            (25, 25, 112),    # Azul profundo
            (255, 255, 255),  # Blanco
            (0, 0, 0),        # Negro
            (255, 165, 0),    # Naranja
            (128, 0, 128),    # Púrpura
            (255, 192, 203),  # Rosa
            (0, 255, 255),    # Cyan
            (255, 255, 0),    # Amarillo
            (0, 128, 0),      # Verde
            (128, 128, 128),  # Gris
        ]

def rgb_to_minecraft_palette(r, g, b, minecraft_colors):
    """Convierte RGB a paleta Minecraft usando distancia euclidiana"""
    # Asegurar valores válidos
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    
    min_distance = float('inf')
    closest_color = minecraft_colors[0]
    
    for mc_color in minecraft_colors:
        # Distancia euclidiana en espacio RGB
        distance = ((r - mc_color[0]) ** 2 + 
                   (g - mc_color[1]) ** 2 + 
                   (b - mc_color[2]) ** 2)
        
        if distance < min_distance:
            min_distance = distance
            closest_color = mc_color
    
    return closest_color

def create_3d_block(color, size=20):
    """Crea bloque 3D estilo Minecraft"""
    try:
        # Crear imagen base
        img = Image.new('RGB', (size, size), color)
        draw = ImageDraw.Draw(img)
        
        r, g, b = color
        
        # Calcular colores para caras del cubo
        # Cara superior (más clara)
        top_factor = 1.4
        top_color = (
            min(255, int(r * top_factor)),
            min(255, int(g * top_factor)),
            min(255, int(b * top_factor))
        )
        
        # Cara lateral (más oscura)
        side_factor = 0.6
        side_color = (
            max(0, int(r * side_factor)),
            max(0, int(g * side_factor)),
            max(0, int(b * side_factor))
        )
        
        # Calcular depth basado en tamaño
        depth = max(3, size // 5)
        
        # Cara frontal (color original)
        draw.rectangle([0, depth, size - depth, size], fill=color)
        
        # Cara superior
        top_points = [
            (depth, 0),
            (size, 0),
            (size - depth, depth),
            (0, depth)
        ]
        draw.polygon(top_points, fill=top_color)
        
        # Cara lateral derecha
        side_points = [
            (size - depth, depth),
            (size, 0),
            (size, size - depth),
            (size - depth, size)
        ]
        draw.polygon(side_points, fill=side_color)
        
        # Bordes para definición
        draw.rectangle([0, depth, size - depth, size], outline=(0, 0, 0), width=1)
        draw.polygon(top_points, outline=(0, 0, 0), width=1)
        draw.polygon(side_points, outline=(0, 0, 0), width=1)
        
        return img
    
    except Exception as e:
        # Fallback: bloque simple si falla el 3D
        img = Image.new('RGB', (size, size), color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, size-1, size-1], outline=(0, 0, 0), width=1)
        return img

@st.cache_data
def pixelate_image_robust(image_bytes, pixel_size=16, rotation_angle=0):
    """Pixela imagen de forma robusta con rotación"""
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convertir a RGB si es necesario
    if image.mode not in ['RGB', 'RGBA']:
        image = image.convert('RGB')
    
    # Aplicar rotación si es necesaria
    if rotation_angle != 0:
        image = rotate_image(image, rotation_angle)
    
    width, height = image.size
    
    # Calcular nuevas dimensiones
    new_width = max(1, width // pixel_size)
    new_height = max(1, height // pixel_size)
    
    # Redimensionar usando NEAREST para pixelado nítido
    pixelated = image.resize((new_width, new_height), Image.NEAREST)
    
    return pixelated

def create_minecraft_3d_art(image_bytes, block_size=20, pixel_size=16, preserve_original_colors=True, rotation_angle=0):
    """Convierte imagen en arte 3D Minecraft de forma robusta con rotación"""
    try:
        # Crear paleta adaptada (usando imagen original para mejores colores)
        minecraft_colors = create_adaptive_minecraft_palette(image_bytes, preserve_original_colors)
        
        # Pixelar imagen con rotación
        pixelated = pixelate_image_robust(image_bytes, pixel_size, rotation_angle)
        
        # Obtener dimensiones
        pix_width, pix_height = pixelated.size
        
        # Limitar tamaño máximo para evitar problemas de memoria
        max_dimension = 100
        if pix_width > max_dimension or pix_height > max_dimension:
            scale_factor = min(max_dimension / pix_width, max_dimension / pix_height)
            new_width = int(pix_width * scale_factor)
            new_height = int(pix_height * scale_factor)
            pixelated = pixelated.resize((new_width, new_height), Image.NEAREST)
            pix_width, pix_height = new_width, new_height
        
        # Crear canvas resultado
        canvas_width = pix_width * block_size
        canvas_height = pix_height * block_size
        result = Image.new('RGB', (canvas_width, canvas_height), (135, 206, 235))
        
        # Convertir a array numpy de forma segura
        pix_array = np.array(pixelated)
        
        # Asegurar que es 3D
        if len(pix_array.shape) == 2:
            pix_array = np.stack([pix_array] * 3, axis=-1)
        
        # Crear bloques 3D
        for y in range(pix_height):
            for x in range(pix_width):
                try:
                    # Obtener color del pixel
                    pixel = pix_array[y, x]
                    
                    if len(pixel) >= 3:
                        r, g, b = pixel[0], pixel[1], pixel[2]
                    else:
                        # Escala de grises
                        gray = pixel if np.isscalar(pixel) else pixel[0]
                        r = g = b = gray
                    
                    # Convertir a paleta Minecraft
                    mc_color = rgb_to_minecraft_palette(r, g, b, minecraft_colors)
                    
                    # Crear y pegar bloque
                    block = create_3d_block(mc_color, block_size)
                    result.paste(block, (x * block_size, y * block_size))
                
                except Exception as e:
                    # Si falla un bloque, usar color por defecto
                    default_block = create_3d_block((128, 128, 128), block_size)
                    result.paste(default_block, (x * block_size, y * block_size))
        
        return result, minecraft_colors
    
    except Exception as e:
        st.error(f"Error en procesamiento: {str(e)}")
        # Imagen de error
        error_img = Image.new('RGB', (400, 300), (255, 0, 0))
        return error_img, [(255, 0, 0)]

def show_color_palette(colors, title="Paleta de Colores"):
    """Muestra la paleta de colores"""
    try:
        palette_height = 50
        palette_width = len(colors) * 40
        
        palette_img = Image.new('RGB', (palette_width, palette_height), (255, 255, 255))
        draw = ImageDraw.Draw(palette_img)
        
        for i, color in enumerate(colors):
            x = i * 40
            draw.rectangle([x, 0, x + 40, palette_height], fill=color)
            draw.rectangle([x, 0, x + 40, palette_height], outline=(0, 0, 0), width=1)
        
        return palette_img
    except Exception:
        # Fallback simple
        return Image.new('RGB', (400, 50), (200, 200, 200))

# Configuración de página
st.set_page_config(
    page_title="Minecraft 3D Pixelator con Rotación",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 Pixelado Minecraft")
st.markdown("Convierte cualquier imagen en arte pixelado 3D estilo Minecraft - **Con rotación para fotos móviles**")

# Información del sistema
if not SKLEARN_AVAILABLE:
    st.info("ℹ️ Ejecutándose en modo compatible (sin scikit-learn)")

# Sidebar
st.sidebar.header("⚙️ Configuración")

uploaded_file = st.sidebar.file_uploader(
    "Sube tu imagen",
    type=['png', 'jpg', 'jpeg'],
    help="Formatos: PNG, JPG, JPEG"
)

# Control de rotación
st.sidebar.subheader("🔄 Rotación")
rotation_angle = st.sidebar.slider(
    "Rotar imagen (grados)",
    min_value=-180,
    max_value=180,
    value=0,
    step=90,
    help="Útil para corregir fotos tomadas con móvil"
)

# Botones de rotación rápida
col_rot1, col_rot2, col_rot3 = st.sidebar.columns(3)
with col_rot1:
    if st.button("↻ 90°"):
        rotation_angle = 90
with col_rot2:
    if st.button("↻ 180°"):
        rotation_angle = 180
with col_rot3:
    if st.button("↻ 270°"):
        rotation_angle = 270

st.sidebar.subheader("🎨 Pixelado")
pixel_size = st.sidebar.slider(
    "Nivel de pixelado",
    min_value=8,
    max_value=32,
    value=16,
    help="Menor = más detalle"
)

block_size = st.sidebar.slider(
    "Tamaño del bloque 3D",
    min_value=15,
    max_value=30,
    value=20
)

preserve_colors = st.sidebar.checkbox(
    "Adaptar colores originales",
    value=True,
    help="Usa colores de tu imagen adaptados a Minecraft"
)

# Layout principal
col1, col2 = st.columns(2)

if uploaded_file is not None:
    try:
        # Leer bytes del archivo
        image_bytes = uploaded_file.read()
        original_image = Image.open(io.BytesIO(image_bytes))
        
        # Aplicar rotación a la imagen para previsualización
        display_image = rotate_image(original_image, rotation_angle) if rotation_angle != 0 else original_image
        
        with col1:
            st.subheader("📷 Imagen Original")
            if rotation_angle != 0:
                st.info(f"🔄 Rotada {rotation_angle}°")
            st.image(display_image, caption="Imagen subida", use_container_width=True)
            st.info(f"Dimensiones originales: {original_image.size[0]}x{original_image.size[1]} píxeles")
            if rotation_angle != 0:
                st.info(f"Dimensiones rotadas: {display_image.size[0]}x{display_image.size[1]} píxeles")
        
        with col2:
            st.subheader("🧱 Resultado Minecraft 3D")
            
            with st.spinner("Generando arte pixelado..."):
                # Procesar imagen con rotación
                minecraft_art, used_palette = create_minecraft_3d_art(
                    image_bytes, block_size, pixel_size, preserve_colors, rotation_angle
                )
                
                # Mostrar resultado
                st.image(minecraft_art, caption="Arte Minecraft 3D", use_container_width=True)
                
                # Mostrar paleta
                st.subheader("🎨 Paleta Utilizada")
                palette_img = show_color_palette(used_palette)
                st.image(palette_img, caption=f"{len(used_palette)} colores")
                
                # Info del resultado
                st.success(f"Resultado: {minecraft_art.size[0]}x{minecraft_art.size[1]} píxeles")
                
                # Descarga
                buf = io.BytesIO()
                minecraft_art.save(buf, format='PNG', optimize=True)
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Descargar resultado",
                    data=byte_im,
                    file_name=f"minecraft_3d_art_rot{rotation_angle}.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    except Exception as e:
        st.error(f"Error al procesar: {str(e)}")
        st.info("Intenta con una imagen más pequeña o diferente formato")

else:
    st.info("👆 Sube una imagen para comenzar")
    st.markdown("### 💡 Consejos:")
    st.markdown("- Usa el control de rotación si tu foto está girada")
    st.markdown("- Los botones de rotación rápida (90°, 180°, 270°) facilitan la corrección")
    st.markdown("- La rotación es especialmente útil para fotos tomadas con móvil")