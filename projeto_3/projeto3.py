import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:\\Users\\lucas\\Documents\\UNIFOR 2025.1\\PROCESSAMENTO DIGITAL DE IMAGEM\\TRABALHO_AV2\\projeto_3\\img_folha_1.JPG')

if img is None:
    print("Erro ao carregar a imagem!")
else:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_healthy = np.array([35, 50, 50])
    upper_healthy = np.array([85, 255, 255])
    lower_damaged = np.array([10, 50, 50])
    upper_damaged = np.array([30, 255, 255])
    mask_healthy = cv2.inRange(img_hsv, lower_healthy, upper_healthy)
    mask_damaged = cv2.inRange(img_hsv, lower_damaged, upper_damaged)
    kernel = np.ones((5, 5), np.uint8)
    mask_healthy = cv2.morphologyEx(mask_healthy, cv2.MORPH_OPEN, kernel)
    mask_healthy = cv2.morphologyEx(mask_healthy, cv2.MORPH_CLOSE, kernel)
    mask_damaged = cv2.morphologyEx(mask_damaged, cv2.MORPH_OPEN, kernel)
    mask_damaged = cv2.morphologyEx(mask_damaged, cv2.MORPH_CLOSE, kernel)
    healthy = cv2.bitwise_and(img, img, mask=mask_healthy)
    damaged = cv2.bitwise_and(img, img, mask=mask_damaged)
    plt.figure(figsize=(15, 10))
    plt.subplot(221), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Imagem Original')
    plt.subplot(222), plt.imshow(mask_healthy, cmap='gray'), plt.title('Máscara Saudável')
    plt.subplot(223), plt.imshow(mask_damaged, cmap='gray'), plt.title('Máscara Danificada')
    plt.subplot(224), plt.imshow(cv2.cvtColor(healthy + damaged, cv2.COLOR_BGR2RGB)), plt.title('Regiões Segmentadas')
    plt.show()
    cv2.imwrite('results/resultado_folha_saudavel.jpg', healthy)
    cv2.imwrite('results/resultado_folha_danificada.jpg', damaged)
    print("Resultados salvos como 'results/resultado_folha_saudavel.jpg' e 'results/resultado_folha_danificada.jpg'")