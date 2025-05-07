import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:\\Users\\lucas\\Documents\\UNIFOR 2025.1\\PROCESSAMENTO DIGITAL DE IMAGEM\\TRABALHO_AV2\\projeto_2\\circulos_1.png', cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread('C:\\Users\\lucas\\Documents\\UNIFOR 2025.1\\PROCESSAMENTO DIGITAL DE IMAGEM\\TRABALHO_AV2\\projeto_2\\circulos_1.png')

if img is None:
    print("Erro ao carregar a imagem!")
else:
    img_blur = cv2.GaussianBlur(img, (9, 9), 2)
    circles = cv2.HoughCircles(
        img_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        count = 0
        for i in circles[0, :]:
            cv2.circle(img_color, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.circle(img_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
            count += 1
        print(f"Círculos detectados: {count}")
    else:
        print("Nenhum círculo detectado.")
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title(f'Resultado: {count} círculos detectados')
    plt.show()
    cv2.imwrite('results/resultado_circulos.jpg', img_color)
    print("Resultado salvo como 'results/resultado_circulos.jpg'")