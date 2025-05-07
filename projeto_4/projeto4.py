import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:\\Users\\lucas\\Documents\\UNIFOR 2025.1\\PROCESSAMENTO DIGITAL DE IMAGEM\\TRABALHO_AV2\\projeto_4\\Tumor (79).jpg', cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread('C:\\Users\\lucas\\Documents\\UNIFOR 2025.1\\PROCESSAMENTO DIGITAL DE IMAGEM\\TRABALHO_AV2\\projeto_4\\Tumor (79).jpg')

if img is None:
    print("Erro ao carregar a imagem!")
else:
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary_closed = cv2.morphologyEx(binary_open, cv2.MORPH_CLOSE, kernel, iterations=2)
    tumor = cv2.bitwise_and(img_color, img_color, mask=binary_closed)
    plt.figure(figsize=(15, 10))
    plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Imagem Original (Tons de Cinza)')
    plt.subplot(222), plt.imshow(binary, cmap='gray'), plt.title('Binarização (Otsu)')
    plt.subplot(223), plt.imshow(binary_closed, cmap='gray'), plt.title('Após Operações Morfológicas')
    plt.subplot(224), plt.imshow(cv2.cvtColor(tumor, cv2.COLOR_BGR2RGB)), plt.title('Tumor Isolado')
    plt.show()
    cv2.imwrite('results/resultado_binarizacao.jpg', binary)
    cv2.imwrite('results/resultado_tumor_isolado.jpg', tumor)
    print("Resultados salvos como 'results/resultado_binarizacao.jpg' e 'results/resultado_tumor_isolado.jpg'")