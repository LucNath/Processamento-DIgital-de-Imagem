import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar as imagens
img_fundo_verde = cv2.imread('C:\\Users\\lucas\\Documents\\UNIFOR 2025.1\\PROCESSAMENTO DIGITAL DE IMAGEM\\TRABALHO_AV2\\projeto_1\\img_fundo_verde_1.jpg')
img_background = cv2.imread('C:\\Users\\lucas\\Documents\\UNIFOR 2025.1\\PROCESSAMENTO DIGITAL DE IMAGEM\\TRABALHO_AV2\\projeto_1\\background_2.png')

if img_fundo_verde is None or img_background is None:
    print("Erro ao carregar as imagens!")
else:
    img_background = cv2.resize(img_background, (img_fundo_verde.shape[1], img_fundo_verde.shape[0]))
    img_hsv = cv2.cvtColor(img_fundo_verde, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    person = cv2.bitwise_and(img_fundo_verde, img_fundo_verde, mask=mask_inv)
    background = cv2.bitwise_and(img_background, img_background, mask=mask)
    result = cv2.add(person, background)
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img_fundo_verde, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(132), plt.imshow(mask_inv, cmap='gray'), plt.title('Máscara da Pessoa')
    plt.subplot(133), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Resultado Chroma Key')
    plt.show()
    cv2.imwrite('results/resultado_chroma_key.jpg', result)
    print("Resultado salvo como 'results/resultado_chroma_key.jpg'")
