import numpy as np
import cv2
import sys

class Logger(object):
    def __init__(self, filename="outputs/logfile.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

def select_named_rois(image_2d, names, window_title="Seleccione ROI"):
    """
    Pide al usuario seleccionar ROIs una por una con un nombre específico mostrado en el título.
    """
    img_disp = cv2.normalize(image_2d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rois = []
    total = len(names)
    for i, name in enumerate(names):
        print(f"\n ---> Acción requerida [{i+1}/{total}]: Selecciona la región para '{name}'")
        title = f"{window_title} - Selecciona: {name} (ENTER confirma)"
        roi = cv2.selectROI(title, img_disp, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(title)
        # roi es (x, y, w, h). Si w y h son > 0, se seleccionó correctamente.
        if roi[2] > 0 and roi[3] > 0:
            rois.append(roi)
            print(f"      [OK] ROI seleccionada: {roi}")
        else:
            print("      [!] Selección cancelada o inválida.")
    return rois # Devuelve lista de rectángulos (x, y, w, h)

def get_mean_spectrum(cube, rois):
    """
    Obtiene el espectro promedio para cada ROI seleccionada.
    """
    spectra = []
    for (x, y, w, h) in rois:
        # Extraer ROI espacial
        roi_cube = cube[y:y+h, x:x+w, :]
        # Promediar dimensiones espaciales
        mean_spec = np.mean(roi_cube, axis=(0, 1))
        spectra.append(mean_spec)
    return spectra

def resize_spectral_cube(cube, target_shape):
    """
    Redimensiona el cubo hiperespectral banda por banda para hacer coincidir resoluciones espaciales.
    target_shape debe ser (filas, columnas).
    """
    resized_cube = np.zeros((target_shape[0], target_shape[1], cube.shape[2]), dtype=cube.dtype)
    for i in range(cube.shape[2]):
        # cv2.resize espera (ancho, alto) -> (columnas, filas)
        resized_cube[:, :, i] = cv2.resize(cube[:, :, i], (target_shape[1], target_shape[0]))
    return resized_cube
