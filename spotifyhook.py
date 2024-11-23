import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from scipy.signal import find_peaks
import logging
import re

class SpotifyCodePlayer:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        self.root = tk.Tk()
        self.root.title("Spotify Code Scanner")
        
        self.is_scanning = False
        self.cap = None
        
        self._init_spotify()
        self._init_constants()
        self.setup_ui()

    def _init_spotify(self):
        try:
            self.auth_manager = SpotifyOAuth(
                client_id='1ab3f5e442cb4cf5977d251fa9dfa1c6',
                client_secret='de8751576422429e863fbcf5a3feb566',
                redirect_uri='http://localhost:8888/callback',
                scope='user-read-playback-state user-modify-playback-state user-read-currently-playing'
            )
            
            if not self.auth_manager.get_cached_token():
                self.auth_manager.get_access_token()
            
            self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
            self.sp.current_user()
            self.logger.info("Conexión con Spotify establecida exitosamente")
            
        except Exception as e:
            self.logger.error(f"Error al conectar con Spotify: {e}")
            messagebox.showerror("Error", 
                "No se pudo conectar con Spotify. Por favor verifica tus credenciales y conexión a internet.")
            raise

    def _init_constants(self):
        self.PROCESSING_CONSTANTS = {
            'BAR_MIN_HEIGHT': 30,
            'BAR_MIN_WIDTH': 2,
            'BAR_MAX_WIDTH': 6,
            'MIN_BARS': 18,
            'MAX_BARS': 32,
            'MIN_ASPECT_RATIO': 4,
            'MIN_AREA': 100
        }
        
        self.CONTENT_TYPES = {
            1: 'track',
            2: 'album',
            3: 'artist',
            4: 'playlist'
        }
        
        self.ID_PATTERNS = {
            'track': r'^(?:spotify:track:|https://open\.spotify\.com/track/)?([a-zA-Z0-9]{22})(?:\?.*)?$',
            'album': r'^(?:spotify:album:|https://open\.spotify\.com/album/)?([a-zA-Z0-9]{22})(?:\?.*)?$',
            'artist': r'^(?:spotify:artist:|https://open\.spotify\.com/artist/)?([a-zA-Z0-9]{22})(?:\?.*)?$',
            'playlist': r'^(?:spotify:playlist:|https://open\.spotify\.com/playlist/)?([a-zA-Z0-9]{22})(?:\?.*)?$'
        }

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.canvas = tk.Canvas(self.main_frame, width=640, height=480)
        self.canvas.grid(row=0, column=0, columnspan=2, pady=5)

        self.start_button = ttk.Button(self.main_frame, text="Iniciar Escaneo", 
                                     command=self.start_scanning)
        self.start_button.grid(row=1, column=0, pady=5, padx=5)

        self.stop_button = ttk.Button(self.main_frame, text="Detener Escaneo", 
                                    command=self.stop_scanning)
        self.stop_button.grid(row=1, column=1, pady=5, padx=5)
        self.stop_button.state(['disabled'])

        self.status_label = ttk.Label(self.main_frame, text="Estado: Detenido")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=5)

        self.track_label = ttk.Label(self.main_frame, text="Canción: -")
        self.track_label.grid(row=3, column=0, columnspan=2, pady=2)

        self.artist_label = ttk.Label(self.main_frame, text="Artista(s): -")
        self.artist_label.grid(row=4, column=0, columnspan=2, pady=2)

        # Añadir indicadores de debug
        self.debug_frame = ttk.LabelFrame(self.main_frame, text="Debug Info", padding="5")
        self.debug_frame.grid(row=5, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        self.bars_label = ttk.Label(self.debug_frame, text="Barras detectadas: 0")
        self.bars_label.grid(row=0, column=0, pady=2)

        self.threshold_label = ttk.Label(self.debug_frame, text="Umbral: 0")
        self.threshold_label.grid(row=1, column=0, pady=2)

    def validate_and_format_spotify_id(self, input_id, content_type):
        try:
            if content_type not in self.ID_PATTERNS:
                return None
                
            input_id = str(input_id).strip()
            
            match = re.match(self.ID_PATTERNS[content_type], input_id)
            if match:
                id_base62 = match.group(1)
                return f"spotify:{content_type}:{id_base62}"
                
            if len(input_id) == 32 and all(c in '0123456789abcdef' for c in input_id.lower()):
                return None
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error al validar ID de Spotify: {e}")
            return None

    def start_scanning(self):
        try:
            if not self.is_scanning:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("No se pudo acceder a la cámara")
                
                self.is_scanning = True
                self.start_button.state(['disabled'])
                self.stop_button.state(['!disabled'])
                self.status_label.configure(text="Estado: Escaneando")
                self.scan_loop()
        except Exception as e:
            self.logger.error(f"Error al iniciar el escaneo: {e}")
            messagebox.showerror("Error", str(e))

    def stop_scanning(self):
        try:
            self.is_scanning = False
            if self.cap:
                self.cap.release()
            self.start_button.state(['!disabled'])
            self.stop_button.state(['disabled'])
            self.status_label.configure(text="Estado: Detenido")
            self.bars_label.configure(text="Barras detectadas: 0")
            self.threshold_label.configure(text="Umbral: 0")
        except Exception as e:
            self.logger.error(f"Error al detener el escaneo: {e}")
            messagebox.showerror("Error", str(e))

    def display_image(self, frame):
        try:
            # Procesar imagen para debug
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY),
                                         255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, 4)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            valid_bars = self._get_valid_bars(contours)
            
            # Actualizar etiquetas de debug
            self.bars_label.configure(text=f"Barras detectadas: {len(valid_bars)}")
            mean_thresh = np.mean(thresh)
            self.threshold_label.configure(text=f"Umbral medio: {mean_thresh:.2f}")
            
            # Dibujar rectángulos en las barras detectadas
            debug_frame = frame.copy()
            for x, y, w, h in valid_bars:
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            image = Image.fromarray(cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB))
            image = image.resize((640, 480), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        except Exception as e:
            self.logger.error(f"Error al mostrar la imagen: {e}")

    def process_spotify_code(self, image):
        try:
            self.logger.debug("Iniciando procesamiento de imagen")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.logger.debug("Imagen convertida a escala de grises")
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            self.logger.debug("Aplicado desenfoque Gaussiano")
            
            thresh = cv2.adaptiveThreshold(blurred, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, 4)
            self.logger.debug("Aplicado umbral adaptativo")
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            self.logger.debug("Aplicadas operaciones morfológicas")
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                      (1, self.PROCESSING_CONSTANTS['BAR_MIN_HEIGHT']))
            detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
            self.logger.debug("Detectadas líneas verticales")
            
            contours, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            self.logger.debug(f"Encontrados {len(contours)} contornos")

            if not contours:
                return None

            valid_bars = self._get_valid_bars(contours)
            self.logger.debug(f"Detectadas {len(valid_bars)} barras válidas")
            
            if len(valid_bars) < self.PROCESSING_CONSTANTS['MIN_BARS']:
                return None

            heights = [h for _, _, _, h in valid_bars]
            heights_norm = np.array(heights) / max(heights) if heights else np.array([])
            binary_code = self._heights_to_binary(heights_norm)
            
            return self._decode_spotify_code(binary_code)

        except Exception as e:
            self.logger.error(f"Error procesando Spotify Code: {e}")
            return None

    def _get_valid_bars(self, contours):
        try:
            valid_bars = []
            avg_height = sum(cv2.boundingRect(c)[3] for c in contours) / len(contours) if contours else 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                area = cv2.contourArea(contour)
                
                if (self.PROCESSING_CONSTANTS['BAR_MIN_WIDTH'] <= w <= 
                    self.PROCESSING_CONSTANTS['BAR_MAX_WIDTH'] and 
                    h >= self.PROCESSING_CONSTANTS['BAR_MIN_HEIGHT'] and
                    aspect_ratio > self.PROCESSING_CONSTANTS['MIN_ASPECT_RATIO'] and
                    area > self.PROCESSING_CONSTANTS['MIN_AREA']):
                    valid_bars.append((x, y, w, h))
                    
            return sorted(valid_bars, key=lambda x: x[0])
        except Exception as e:
            self.logger.error(f"Error al validar barras: {e}")
            return []

    def _heights_to_binary(self, heights):
        try:
            peaks, _ = find_peaks(heights, height=0.5, distance=2)
            binary = np.zeros(len(heights), dtype=int)
            binary[peaks] = 1
            return binary.tolist()
        except Exception as e:
            self.logger.error(f"Error al convertir alturas a binario: {e}")
            return []

    def _decode_spotify_code(self, binary_code):
        try:
            if len(binary_code) < self.PROCESSING_CONSTANTS['MIN_BARS']:
                return None

            code_bytes = [int(''.join(map(str, binary_code[i:i+8])), 2) 
                         for i in range(0, len(binary_code), 8) 
                         if len(binary_code[i:i+8]) == 8]

            if len(code_bytes) < 4:
                return None

            content_type = self.CONTENT_TYPES.get(code_bytes[0])
            if not content_type:
                return None

            content_id = ''.join([format(b, '02x') for b in code_bytes[4:]])
            formatted_uri = self.validate_and_format_spotify_id(content_id, content_type)
            
            if not formatted_uri:
                self.logger.warning(f"ID no válido generado: {content_id}")
                return None

            return formatted_uri

        except Exception as e:
            self.logger.error(f"Error decodificando Spotify Code: {e}")
            return None

    def scan_loop(self):
        try:
            if self.is_scanning and self.cap:
                ret, frame = self.cap.read()
                if ret:
                    spotify_uri = self.process_spotify_code(frame)
                    if spotify_uri:
                        self.logger.info(f"Código Spotify detectado: {spotify_uri}")
                        self.play_spotify_uri(spotify_uri)
                    self.display_image(frame)
                self.root.after(10, self.scan_loop)
        except Exception as e:
            self.logger.error(f"Error en el bucle de escaneo: {e}")
            self.stop_scanning()

    def play_spotify_uri(self, uri):
        try:
            devices = self.sp.devices()['devices']
            if not devices:
                raise Exception("No se encontraron dispositivos disponibles")

            active_device = next((device for device in devices if device['is_active']), devices[0])
            self.logger.info(f"Reproduciendo en dispositivo: {active_device['name']}")

            try:
                self.sp.start_playback(device_id=active_device['id'], uris=[uri])
                track_info = self.sp.track(uri)
                self._update_track_info(track_info)
                self.logger.info(f"Reproduciendo: {track_info['name']}")
            except Exception as playback_error:
                self.logger.error(f"Error al iniciar reproducción: {playback_error}")
                # Intenta reproducir sin especificar dispositivo
                self.sp.start_playback(uris=[uri])
                track_info = self.sp.track(uri)
                self._update_track_info(track_info)

        except Exception as e:
            self.logger.error(f"Error reproduciendo: {e}")
            messagebox.showerror("Error", f"Error al reproducir: {str(e)}")

    def _update_track_info(self, track_info):
        try:
            track_name = track_info['name']
            artists = ', '.join(artist['name'] for artist in track_info['artists'])
            album = track_info['album']['name']

            self.track_label.configure(text=f"Canción: {track_name}")
            self.artist_label.configure(text=f"Artista(s): {artists}")
            self.status_label.configure(text=f"Estado: Reproduciendo desde {album}")
            
            self.logger.info(f"Información actualizada - Canción: {track_name}, Artistas: {artists}")
        except Exception as e:
            self.logger.error(f"Error al actualizar información de la pista: {e}")

    def run(self):
        try:
            # Centrar la ventana
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'{width}x{height}+{x}+{y}')
            
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Error en la ejecución principal: {e}")
            messagebox.showerror("Error", "Error fatal en la aplicación")
        finally:
            if self.cap:
                self.cap.release()

def main():
    try:
        # Configurar logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('spotify_scanner.log'),
                logging.StreamHandler()
            ]
        )
        
        # Crear y ejecutar la aplicación
        app = SpotifyCodePlayer()
        app.run()
        
    except Exception as e:
        logging.error(f"Error al iniciar la aplicación: {e}")
        messagebox.showerror("Error", f"No se pudo iniciar la aplicación: {str(e)}")
        raise

if __name__ == "__main__":
    main()