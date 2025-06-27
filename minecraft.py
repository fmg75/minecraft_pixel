import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import io
import colorsys
from sklearn.cluster import KMeans
from collections import Counter

def extract_dominant_colors(image, num_colors=16):
    """Extrae los colores dominantes de la imagen"""
    # Convertir imagen a array numpy
    img_array = np.array(image)
    
    # Redimensionar para acelerar el procesamiento
    temp_img = image.resize((100, 100))
    pixels = np.array(temp_img).reshape(-1, 3)
    
    # Usar K-means para encontrar colores dominantes
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    # Obtener la frecuencia de cada color
    labels = kmeans.labels_
    label_counts = Counter(labels)
    
    # Ordenar colores por frecuencia
    sorted_colors = []
    for i in sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True):
        sorted_colors.append(tuple(dominant_colors[i]))
    
    return sorted_colors

def minecraft_color_adjustment(color):
    """Ajusta un color para que se vea más 'Minecraft'"""
    r, g, b = color
    
    # Convertir a HSV para mejor manipulación
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    
    # Aumentar ligeramente la saturación para colores más vibrantes
    s = min(1.0, s * 1.2)
    
    # Ajustar el brillo para evitar colores muy oscuros o muy claros
    if v < 0.2:
        v = 0.3
    elif v > 0.9:
        v = 0.85
    
    # Convertir de vuelta a RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    
    return (int(r * 255), int(g * 255), int(b * 255))

def create_adaptive_minecraft_palette(image, preserve_original=True):
    """Crea una paleta Minecraft adaptada a la imagen original"""
    if preserve_original:
        # Extraer colores dominantes de la imagen
        dominant_colors = extract_dominant_colors(image, num_colors=12)
        
        # Ajustar colores para estilo Minecraft
        minecraft_adapted = [minecraft_color_adjustment(color) for color in dominant_colors]
        
        # Añadir algunos colores básicos de Minecraft para completar
        basic_minecraft = [
            (34, 139, 34),    # Verde hierba
            (139, 69, 19),    # Marrón tierra
            (105, 105, 105),  # Gris piedra
            (255, 255, 255),  # Blanco
        ]
        
        # Combinar paletas evitando duplicados
        final_palette = minecraft_adapted.copy()
        for color in basic_minecraft:
            if not any(sum((a - b) ** 2 for a, b in zip(color, existing)) < 1000 
                      for existing in final_palette):
                final_palette.append(color)
        
        return final_palette[:16]  # Limitar a 16 colores
    
    else:
        # Paleta original fija
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
    """Convierte colores RGB a la paleta Minecraft adaptada"""
    # Encuentra el color más cercano en la paleta
    min_distance = float('inf')
    closest_color = minecraft_colors[0]
    
    for mc_color in minecraft_colors:
        # Usar distancia euclidiana en espacio RGB
        distance = sum((a - b) ** 2 for a, b in zip((r, g, b), mc_color))
        if distance < min_distance:
            min_distance = distance
            closest_color = mc_color
    
    return closest_color

def create_3d_block(color, size=20):
    """Crea un bloque 3D estilo Minecraft"""
    # Crear imagen del bloque
    img = Image.new('RGB', (size, size), color)
    draw = ImageDraw.Draw(img)
    
    r, g, b = color
    
    # Calcular colores para las caras del cubo
    # Cara superior (más clara)
    top_color = tuple(min(255, int(c * 1.3)) for c in color)
    # Cara lateral derecha (más oscura) 
    right_color = tuple(max(0, int(c * 0.7)) for c in color)
    # Cara frontal (color original)
    front_color = color
    
    # Dibujar efecto 3D
    depth = size // 4
    
    # Cara frontal
    draw.rectangle([0, depth, size-depth, size], fill=front_color)
    
    # Cara superior
    points = [
        (depth, 0),
        (size, 0),
        (size-depth, depth),
        (0, depth)
    ]
    draw.polygon(points, fill=top_color)
    
    # Cara lateral derecha
    points = [
        (size-depth, depth),
        (size, 0),
        (size, size-depth),
        (size-depth, size)
    ]
    draw.polygon(points, fill=right_color)
    
    # Agregar bordes para mayor definición
    draw.rectangle([0, depth, size-depth, size], outline=(0, 0, 0), width=1)
    
    return img

def pixelate_image(image, pixel_size=16):
    """Pixela la imagen reduciendo la resolución"""
    # Obtener dimensiones originales
    width, height = image.size
    
    # Calcular nuevas dimensiones para el pixelado
    new_width = width // pixel_size
    new_height = height // pixel_size
    
    # Redimensionar a tamaño pequeño (pixelado)
    small_image = image.resize((new_width, new_height), Image.NEAREST)
    
    return small_image

def create_minecraft_3d_art(image, block_size=20, pixel_size=16, preserve_original_colors=True):
    """Convierte imagen en arte pixelado 3D estilo Minecraft"""
    # Crear paleta adaptada a la imagen
    minecraft_colors = create_adaptive_minecraft_palette(image, preserve_original_colors)
    
    # Pixelar la imagen
    pixelated = pixelate_image(image, pixel_size)
    
    # Obtener dimensiones de la imagen pixelada
    pix_width, pix_height = pixelated.size
    
    # Crear canvas para el resultado final
    canvas_width = pix_width * block_size
    canvas_height = pix_height * block_size
    result = Image.new('RGB', (canvas_width, canvas_height), (135, 206, 235))  # Fondo cielo
    
    # Convertir imagen pixelada a array numpy para procesamiento
    pix_array = np.array(pixelated)
    
    # Crear cada bloque 3D
    for y in range(pix_height):
        for x in range(pix_width):
            # Obtener color del pixel
            if len(pix_array.shape) == 3:
                r, g, b = pix_array[y, x]
            else:
                # Imagen en escala de grises
                gray = pix_array[y, x]
                r, g, b = gray, gray, gray
            
            # Convertir a paleta Minecraft adaptada
            mc_color = rgb_to_minecraft_palette(r, g, b, minecraft_colors)
            
            # Crear bloque 3D
            block = create_3d_block(mc_color, block_size)
            
            # Pegar bloque en el canvas
            result.paste(block, (x * block_size, y * block_size))
    
    return result, minecraft_colors

def add_depth_effect(image, depth_factor=0.1):
    """Añade efecto de profundidad con sombras"""
    # Convertir a numpy array
    img_array = np.array(image)
    
    # Crear máscara de sombra
    shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    
    # Añadir sombra sutil
    width, height = image.size
    for i in range(0, width, 20):
        for j in range(0, height, 20):
            # Sombra en la esquina inferior derecha de cada bloque
            shadow_draw.rectangle([i+15, j+15, i+20, j+20], 
                                fill=(0, 0, 0, 30))
    
    # Aplicar blur a la sombra
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Combinar imagen original con sombra
    result = Image.alpha_composite(image.convert('RGBA'), shadow)
    
    return result.convert('RGB')

def show_color_palette(colors, title="Paleta de Colores"):
    """Muestra la paleta de colores utilizada"""
    palette_height = 50
    palette_width = len(colors) * 40
    
    palette_img = Image.new('RGB', (palette_width, palette_height), (255, 255, 255))
    draw = ImageDraw.Draw(palette_img)
    
    for i, color in enumerate(colors):
        x = i * 40
        draw.rectangle([x, 0, x + 40, palette_height], fill=color)
        draw.rectangle([x, 0, x + 40, palette_height], outline=(0, 0, 0), width=1)
    
    return palette_img

# Configuración de la página
st.set_page_config(
    page_title="Minecraft 3D Pixelator Mejorado",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 Minecraft 3D Pixel Art Generator Mejorado")
st.markdown("Convierte cualquier imagen en arte pixelado 3D estilo Minecraft respetando los colores originales")

# Sidebar con controles
st.sidebar.header("⚙️ Configuración")

# Upload de imagen
uploaded_file = st.sidebar.file_uploader(
    "Sube tu imagen",
    type=['png', 'jpg', 'jpeg'],
    help="Formatos soportados: PNG, JPG, JPEG"
)

# Controles de personalización
pixel_size = st.sidebar.slider(
    "Tamaño del pixel",
    min_value=8,
    max_value=32,
    value=16,
    help="Menor valor = más detalle, mayor valor = más pixelado"
)

block_size = st.sidebar.slider(
    "Tamaño del bloque 3D",
    min_value=15,
    max_value=30,
    value=20,
    help="Tamaño de cada bloque en píxeles"
)

preserve_colors = st.sidebar.checkbox(
    "Respetar colores originales",
    value=True,
    help="Adapta la paleta Minecraft a los colores de tu imagen"
)

add_depth = st.sidebar.checkbox(
    "Añadir efecto de profundidad",
    value=True,
    help="Agrega sombras para mayor efecto 3D"
)

# Columnas para layout
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Cargar imagen
    image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("📷 Imagen Original")
        st.image(image, caption="Imagen subida", use_column_width=True)
        
        # Información de la imagen
        st.info(f"Dimensiones: {image.size[0]}x{image.size[1]} píxeles")
    
    with col2:
        st.subheader("🧱 Resultado Minecraft 3D")
        
        with st.spinner("Generando arte pixelado 3D..."):
            try:
                # Convertir a RGB si es necesario
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Crear arte Minecraft 3D
                minecraft_art, used_palette = create_minecraft_3d_art(
                    image, block_size, pixel_size, preserve_colors
                )
                
                # Añadir efecto de profundidad si está habilitado
                if add_depth:
                    minecraft_art = add_depth_effect(minecraft_art)
                
                # Mostrar resultado
                st.image(minecraft_art, caption="Arte pixelado 3D estilo Minecraft", use_column_width=True)
                
                # Mostrar paleta de colores utilizada
                st.subheader("🎨 Paleta de Colores Utilizada")
                palette_img = show_color_palette(used_palette)
                st.image(palette_img, caption=f"Paleta adaptada ({len(used_palette)} colores)")
                
                # Información del resultado
                st.success(f"Resultado: {minecraft_art.size[0]}x{minecraft_art.size[1]} píxeles")
                
                # Botón de descarga
                buf = io.BytesIO()
                minecraft_art.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Descargar imagen",
                    data=byte_im,
                    file_name="minecraft_3d_art_mejorado.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error al procesar la imagen: {str(e)}")
                st.info("Asegúrate de que scikit-learn esté instalado: pip install scikit-learn")

else:
    # Mostrar imagen de ejemplo o instrucciones
    st.info("👆 Sube una imagen usando el panel lateral para comenzar")
    
    # Mostrar ejemplo
    st.subheader("🎨 Nuevas Características")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Paleta Adaptiva**
        - Extrae colores dominantes de tu imagen
        - Ajusta automáticamente al estilo Minecraft
        - Preserva la esencia visual original
        """)
    
    with col2:
        st.markdown("""
        **🧊 Efecto 3D Mejorado**
        - Bloques con profundidad realista
        - Iluminación consistente
        - Sombras y highlights optimizados
        """)
    
    with col3:
        st.markdown("""
        **⚡ Control Total**
        - Modo respeto de colores originales
        - Visualización de paleta utilizada
        - Descarga en alta calidad
        """)

# Footer
st.markdown("---")
st.markdown(
    "💡 **Nuevo:** Ahora puedes elegir entre respetar los colores originales de tu imagen "
    "o usar la paleta clásica de Minecraft. ¡Experimenta con ambas opciones!"
)