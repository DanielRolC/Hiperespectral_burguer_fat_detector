import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2

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

def main():
    print("--- 1. Descargar la información ---")
    # Rutas asumiendo ejecución desde el directorio raíz de la práctica
    path_carne = 'ph_carne/VISNIR/carne.mat'
    path_carne_b = 'ph_carne/VISNIR/carne_b.mat'
    path_cal = 'cal/VISNIR/spect_cal.mat'
    path_cal_b = 'cal/VISNIR/spect_cal_b.mat'
    path_pinkref = 'cal/pinkref.mat'

    print("--- 2. Visualizar la cubeta con las muestras de carne ---")
    carne_mat = sio.loadmat(path_carne)
    carne_cube = carne_mat['data']
    print(f"Dimensiones de carne: {carne_cube.shape} (Y, X, L)")
    
    # Reducir dimensión promediando reflectancia en el eje espectral
    carne_2d = np.mean(carne_cube, axis=2)
    plt.figure()
    plt.title('Imagen 2D - Cubeta de Carne')
    plt.imshow(carne_2d, cmap='gray')
    plt.colorbar()
    plt.show(block=False)

    print("--- 3. Visualizar el material de calibración ---")
    cal_mat = sio.loadmat(path_cal)
    cal_cube = cal_mat['data']
    cal_2d = np.mean(cal_cube, axis=2)
    
    # 3.a Seleccionar ROIs en cada material
    nombres_cal = [
        "Calibración Longitud de Onda (WCS-MC-020)", 
        "Calibración Reflectancia 99% (SRS-99-020)"
    ]
    print(f"\n[!] Vamos a seleccionar {len(nombres_cal)} ROIs para los materiales de calibración.")
    rois_cal = select_named_rois(cal_2d, nombres_cal, "Calibración")
    spectra_cal_raw = get_mean_spectrum(cal_cube, rois_cal)
    
    plt.figure()
    for i, s in enumerate(spectra_cal_raw):
        label_name = nombres_cal[i] if i < len(nombres_cal) else f'ROI {i+1}'
        plt.plot(s, label=label_name)
    plt.title('3.a Espectros Promedio (Sin corregir)')
    plt.xlabel('Banda Espectral (Índice)')
    plt.ylabel('Intensidad Promedio')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    print("--- 3.b Corrección en reflectancia del material de calibracion ---")
    cal_b_mat = sio.loadmat(path_cal_b)
    cal_b_cube = cal_b_mat['data']
    
    # Resize si las dimensiones espaciales son distintas
    if cal_b_cube.shape[:2] != cal_cube.shape[:2]:
        print("Redimensionando referencia blanca para la calibración...")
        cal_b_cube_resized = resize_spectral_cube(cal_b_cube, cal_cube.shape[:2])
    else:
        cal_b_cube_resized = cal_b_cube

    # R = S_mp / W (Asumiendo B=0 según el enunciado)
    # Epsilon pequeño para evitar divisiones por cero
    epsilon = 1e-6
    R_cal = cal_cube / (cal_b_cube_resized + epsilon)
    # Limitar valores atípicos por ruido 
    R_cal = np.clip(R_cal, 0, 1)

    print("--- 3.c Espectros de reflectancia en materiales de calibración ---")
    # Utilizamos las mismas ROIs seleccionadas previamente
    spectra_cal_ref = get_mean_spectrum(R_cal, rois_cal)
    
    plt.figure()
    for i, s in enumerate(spectra_cal_ref):
        label_name = nombres_cal[i] if i < len(nombres_cal) else f'ROI {i+1}'
        plt.plot(s, label=label_name)
    plt.title('3.c Reflectancia Corregida del Material de Calibración')
    plt.xlabel('Banda Espectral (Índice)')
    plt.ylabel('Reflectancia')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    print("--- 4. Representar el espectro WCS-MC-020 ---")
    pinkref_mat = sio.loadmat(path_pinkref)
    pinkref_data = pinkref_mat['referenceDataset'] 
    
    lambdas_ref = pinkref_data[:, 0]
    ref_vals = pinkref_data[:, 1]
    
    plt.figure()
    plt.plot(lambdas_ref, ref_vals, 'm')
    plt.title('4. Espectro Referencia WCS-MC-020 (pinkref)')
    plt.xlabel('Longitud de onda (nm)')
    plt.ylabel('Reflectancia')
    plt.grid(True)
    plt.show(block=False)

    print("--- 5. Calibración del Eje Espectral ---")
    print("Usando los puntos de calibración empíricos proporcionados.")
    puntos_x = [96, 201, 251, 284, 528, 712, 795, 962, 988]
    puntos_y = [423.217, 478.932, 502.81, 537.373, 650.216, 748.964, 797.834, 881.225, 911.206]
        
    coefs = np.polyfit(puntos_x, puntos_y, 1) # Ajuste lineal
    eje_lambdas_calibrado = np.polyval(coefs, np.arange(cal_cube.shape[2]))
    print(f"Ecuación del eje calibrado: lambda = {coefs[0]:.4f} * pixel + {coefs[1]:.4f}")

    print("--- 6. Corrección en reflectancia de la cubeta de carne ---")
    carne_b_mat = sio.loadmat(path_carne_b)
    carne_b_cube = carne_b_mat['data']
    
    if carne_b_cube.shape[:2] != carne_cube.shape[:2]:
        print("Redimensionando referencia blanca para la carne...")
        carne_b_cube_resized = resize_spectral_cube(carne_b_cube, carne_cube.shape[:2])
    else:
        carne_b_cube_resized = carne_b_cube
        
    R_carne = carne_cube / (carne_b_cube_resized + epsilon)
    R_carne = np.clip(R_carne, 0, 1)

    print("--- 7. Evolución de espectros en función de la cantidad de tocino ---")
    R_carne_2d = np.mean(R_carne, axis=2)
    nombres_novilla = ["Novilla - Nivel tocino 1 (Menos)", "Novilla - Nivel tocino 2", "Novilla - Nivel tocino 3 (Más)"]
    print(f"\n[!] Tienes que coger {len(nombres_novilla)} muestras de carne de NOVILLA con diferente tocino.")
    rois_novilla = select_named_rois(R_carne_2d, nombres_novilla, "Novilla")
    spectra_novilla = get_mean_spectrum(R_carne, rois_novilla)

    nombres_cerdo = ["Cerdo - Nivel tocino 1 (Menos)", "Cerdo - Nivel tocino 2", "Cerdo - Nivel tocino 3 (Más)"]
    print(f"\n[!] Tienes que coger {len(nombres_cerdo)} muestras de carne de CERDO con diferente tocino.")
    rois_cerdo = select_named_rois(R_carne_2d, nombres_cerdo, "Cerdo")
    spectra_cerdo = get_mean_spectrum(R_carne, rois_cerdo)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i, s in enumerate(spectra_novilla):
        label_name = nombres_novilla[i] if i < len(nombres_novilla) else f'ROI {i+1}'
        plt.plot(eje_lambdas_calibrado, s, label=label_name)
    plt.title('7. Evolución Tocino (Novilla)')
    plt.xlabel('Longitud de Onda (nm)')
    plt.ylabel('Reflectancia')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for i, s in enumerate(spectra_cerdo):
        label_name = nombres_cerdo[i] if i < len(nombres_cerdo) else f'ROI {i+1}'
        plt.plot(eje_lambdas_calibrado, s, label=label_name)
    plt.title('7. Evolución Tocino (Cerdo)')
    plt.xlabel('Longitud de Onda (nm)')
    plt.ylabel('Reflectancia')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    print("--- 8. Métrica de contraste (presencia de tocino) ---")
    print("\n[!] A continuación, selecciona UNICAMENTE 1 ROI sobre la cubeta A1 (Referencia de espectro de grasa).")
    roi_A1 = select_named_rois(R_carne_2d, ["Cubeta A1 (Referencia pura de Grasa)"], "Grasa")
    if roi_A1:
        ref_grasa = get_mean_spectrum(R_carne, roi_A1)[0]
        
        # 8.1 Euclidean Distance (ED)
        print("Calculando Distancia Euclídea (ED)...")
        ED = np.sqrt(np.sum((R_carne - ref_grasa)**2, axis=2))
        
        # 8.2 Spectral Angle Mapper (SAM)
        print("Calculando SAM...")
        dot_product = np.sum(R_carne * ref_grasa, axis=2)
        norm_x = np.linalg.norm(R_carne, axis=2)
        norm_y = np.linalg.norm(ref_grasa)
        cos_theta = np.clip(dot_product / (norm_x * norm_y + epsilon), -1.0, 1.0)
        SAM = np.arccos(cos_theta)
        
        # 8.3 Bray-Curtis Similarity (BCS)
        print("Calculando Bray-Curtis Similarity (BCS)...")
        diff_sum = np.sum(np.abs(R_carne - ref_grasa), axis=2)
        add_sum = np.sum(R_carne + ref_grasa, axis=2)
        BCS = 100 - (diff_sum / (add_sum + epsilon)) * 100
        
        # Visualizar resultados
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('ED (Distancia Euclídea)')
        plt.imshow(ED, cmap='viridis')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.title('SAM')
        plt.imshow(SAM, cmap='viridis')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.title('BCS (Bray-Curtis)')
        plt.imshow(BCS, cmap='viridis')
        plt.colorbar()
        print("Mostrando las gráficas finales. Cierra todas las ventanas para finalizar el script.")
        plt.show() # Bloquea hasta que se cierren las figuras
    else:
        print("No se seleccionó referencia de grasa, omitiendo métricas.")
        plt.show()
        
    print("Práctica completada exitosamente.")

if __name__ == "__main__":
    main()
