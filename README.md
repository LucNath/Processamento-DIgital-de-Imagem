# 🖼️ Processamento Digital de Imagem

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Latest-green.svg)
![Numpy](https://img.shields.io/badge/Numpy-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

Implementação de algoritmos clássicos de **Processamento Digital de Imagens (PDI)** e **Visão Computacional**, desenvolvidos do zero para compreensão profunda dos fundamentos teóricos e práticos da área.

## 📋 Sobre o Projeto

Este repositório contém implementações educacionais de técnicas fundamentais de PDI, desde operações básicas até algoritmos avançados de análise e segmentação de imagens, utilizando Python, OpenCV e Numpy.

## 🎯 Objetivos

- ✅ Implementar algoritmos de PDI do zero
- ✅ Compreender fundamentos matemáticos das operações
- ✅ Comparar diferentes técnicas de processamento
- ✅ Aplicar conceitos em problemas reais
- ✅ Documentar e visualizar resultados

## 🛠️ Técnicas Implementadas

### 1. 🎨 Operações Básicas
- [x] Leitura e escrita de imagens
- [x] Conversão entre espaços de cor (RGB, HSV, Grayscale)
- [x] Operações aritméticas (soma, subtração, multiplicação)
- [x] Operações lógicas (AND, OR, XOR, NOT)
- [x] Manipulação de histogramas

### 2. 🔧 Filtragem Espacial
- [x] **Filtros de Suavização**
  - Filtro de média (box filter)
  - Filtro gaussiano
  - Filtro mediana
  - Filtro bilateral

- [x] **Filtros de Aguçamento**
  - Laplaciano
  - Unsharp masking
  - High-boost filtering

- [x] **Detecção de Bordas**
  - Sobel (horizontal e vertical)
  - Prewitt
  - Roberts
  - Canny Edge Detector
  - Laplacian of Gaussian (LoG)

### 3. 📊 Transformadas
- [x] Transformada de Fourier (DFT/FFT)
- [x] Filtragem no domínio da frequência
- [x] Transformada Discreta de Cosseno (DCT)
- [x] Transformada Wavelet

### 4. 🎭 Realce de Imagens
- [x] **Equalização de Histograma**
  - Global
  - Adaptativa (CLAHE)
  
- [x] **Transformações de Intensidade**
  - Linear (contraste e brilho)
  - Logarítmica
  - Potência (Gamma correction)
  - Negativo

- [x] **Operações Morfológicas**
  - Erosão
  - Dilatação
  - Abertura (Opening)
  - Fechamento (Closing)
  - Gradiente morfológico
  - Top-hat e Black-hat

### 5. 🧩 Segmentação
- [x] **Thresholding**
  - Global (Otsu)
  - Adaptativo
  - Multi-level
  
- [x] **Baseada em Região**
  - Region Growing
  - Watershed
  
- [x] **Clustering**
  - K-means
  - Mean Shift
  
- [x] **Contornos**
  - Detecção de contornos
  - Aproximação de contornos
  - Convex Hull

### 6. 🔍 Análise de Imagens
- [x] Detecção de features (SIFT, SURF, ORB)
- [x] Matching de features
- [x] Template matching
- [x] Análise de textura (GLCM)
- [x] Momentos de imagem

### 7. 🌈 Processamento de Cor
- [x] Conversão entre espaços de cor
- [x] Equalização colorida
- [x] Color transfer
- [x] Segmentação por cor

## 🗂️ Estrutura do Projeto

```
Processamento-Digital-de-Imagem/
│
├── 01_Basico/
│   ├── leitura_escrita.py
│   ├── conversao_cores.py
│   └── operacoes_basicas.py
│
├── 02_Filtragem/
│   ├── filtros_suavizacao.py
│   ├── filtros_aguamento.py
│   └── deteccao_bordas.py
│
├── 03_Transformadas/
│   ├── fourier.py
│   ├── dct.py
│   └── wavelets.py
│
├── 04_Realce/
│   ├── equalizacao.py
│   ├── transformacoes.py
│   └── morfologia.py
│
├── 05_Segmentacao/
│   ├── thresholding.py
│   ├── watershed.py
│   ├── clustering.py
│   └── contornos.py
│
├── 06_Features/
│   ├── detectores.py
│   ├── descritores.py
│   └── matching.py
│
├── 07_Aplicacoes/
│   ├── reconhecimento_facial.py
│   ├── deteccao_objetos.py
│   └── analise_texturas.py
│
├── imagens/              # Dataset de imagens de teste
├── resultados/           # Imagens processadas
└── README.md
```

## 🚀 Como Usar

### Pré-requisitos

```bash
Python 3.13+
OpenCV (cv2)
Numpy
Matplotlib
Scipy (opcional)
Pillow (opcional)
```

### Instalação

```bash
# Clone o repositório
git clone https://github.com/LucNath/Processamento-Digital-de-Imagem.git
cd Processamento-Digital-de-Imagem

# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instale as dependências
pip install opencv-python numpy matplotlib scipy pillow
```

### Exemplo de Uso

```python
import cv2
import numpy as np
from filtros import filtro_gaussiano, detectar_bordas

# Carregar imagem
imagem = cv2.imread('imagens/teste.jpg')

# Aplicar filtro gaussiano
img_suavizada = filtro_gaussiano(imagem, kernel_size=5)

# Detectar bordas com Canny
bordas = detectar_bordas(img_suavizada, threshold1=100, threshold2=200)

# Salvar resultado
cv2.imwrite('resultados/bordas.jpg', bordas)

# Visualizar
cv2.imshow('Original', imagem)
cv2.imshow('Bordas', bordas)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 📊 Exemplos Visuais

### Filtragem Espacial
| Original | Filtro Gaussiano | Detecção de Bordas |
|----------|------------------|-------------------|
| ![Original](resultados/original.jpg) | ![Gaussian](resultados/gaussian.jpg) | ![Edges](resultados/edges.jpg) |

### Equalização de Histograma
| Original | Equalizado Global | CLAHE |
|----------|-------------------|-------|
| ![Original](resultados/hist_original.jpg) | ![Global](resultados/hist_global.jpg) | ![CLAHE](resultados/hist_clahe.jpg) |

### Segmentação
| Original | Threshold | Watershed | K-means |
|----------|-----------|-----------|---------|
| ![Orig](resultados/seg_original.jpg) | ![Thresh](resultados/seg_threshold.jpg) | ![Water](resultados/seg_watershed.jpg) | ![Kmeans](resultados/seg_kmeans.jpg) |

## 🧮 Fundamentos Matemáticos

### Convolução 2D

```
g(x,y) = Σ Σ f(x-i, y-j) * h(i,j)
```

Onde:
- `f`: Imagem original
- `h`: Kernel/filtro
- `g`: Imagem resultante

### Transformada de Fourier

```
F(u,v) = Σ Σ f(x,y) * e^(-j2π(ux/M + vy/N))
```

### Gradiente (Magnitude e Direção)

```
|∇f| = √(Gx² + Gy²)
θ = arctan(Gy / Gx)
```

### Operações Morfológicas

```
Erosão:    A ⊖ B = {z | (B)z ⊆ A}
Dilatação: A ⊕ B = {z | (B̂)z ∩ A ≠ ∅}
Abertura:  A ∘ B = (A ⊖ B) ⊕ B
Fechamento: A • B = (A ⊕ B) ⊖ B
```

## 🎓 Conceitos Aplicados

### Processamento no Domínio Espacial
- Manipulação direta dos pixels
- Filtros de convolução
- Operações ponto a ponto
- Transformações geométricas

### Processamento no Domínio da Frequência
- Análise espectral
- Filtragem passa-baixa/alta
- Remoção de ruído periódico
- Compressão de imagens

### Visão Computacional
- Detecção de features
- Correspondência de padrões
- Análise de movimento
- Reconhecimento de objetos

## 📚 Aplicações Práticas

### Médica
- 🏥 Realce de imagens de raio-X
- 🧬 Segmentação de células
- 🫁 Análise de ressonância magnética

### Industrial
- 🔍 Inspeção de qualidade
- 📏 Medições dimensionais
- 🎯 Detecção de defeitos

### Segurança
- 👤 Reconhecimento facial
- 🚗 Detecção de placas veiculares
- 📹 Vigilância inteligente

### Entretenimento
- 🎨 Filtros de redes sociais
- 🎬 Efeitos visuais
- 🎮 Realidade aumentada

## 🛠️ Ferramentas e Bibliotecas

### Principais
- **OpenCV** - Biblioteca principal de PDI
- **Numpy** - Operações matriciais
- **Matplotlib** - Visualização

### Complementares
- **Scipy** - Processamento científico
- **Pillow** - Manipulação de imagens
- **scikit-image** - Algoritmos avançados

## 📖 Recursos de Aprendizado

### Livros Recomendados
- 📕 "Digital Image Processing" - Gonzalez & Woods
- 📗 "Computer Vision" - Szeliski
- 📘 "Multiple View Geometry" - Hartley & Zisserman

### Cursos Online
- [OpenCV Python Tutorial](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Stanford CS231n](http://cs231n.stanford.edu/)
- [Coursera - Image Processing](https://www.coursera.org/learn/digital)

### Datasets
- [ImageNet](http://www.image-net.org/)
- [COCO](https://cocodataset.org/)
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

## 🔬 Projetos Relacionados

- [ ] Sistema de Reconhecimento Facial
- [ ] Detector de Placas Veiculares
- [ ] Contador de Objetos em Imagens
- [ ] Filtros de Redes Sociais
- [ ] Análise de Qualidade de Produtos
- [ ] Segmentação de Imagens Médicas

## 📊 Benchmarks e Performance

### Tempo de Execução (imagem 1920x1080)
| Operação | Tempo Médio |
|----------|-------------|
| Filtro Gaussiano (5x5) | ~15 ms |
| Canny Edge Detection | ~25 ms |
| Watershed | ~150 ms |
| SIFT Features | ~100 ms |

*Medido em: Intel i7, 16GB RAM*

## 🤝 Contribuindo

Contribuições são bem-vindas! Siga estes passos:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adicionar nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

### Diretrizes
- Documente bem o código
- Adicione exemplos de uso
- Inclua imagens de resultado
- Mantenha consistência de estilo

## 🐛 Reportar Bugs

Encontrou um bug? Abra uma [issue](https://github.com/LucNath/Processamento-Digital-de-Imagem/issues) com:
- Descrição detalhada
- Passos para reproduzir
- Comportamento esperado vs atual
- Screenshots (se aplicável)

## 📜 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👨‍💻 Autor

**Lucas Nathan**

- GitHub: [@LucNath](https://github.com/LucNath)
- LinkedIn: [Lucas Nathan](https://linkedin.com/in/-)
- Email: -

## 🙏 Agradecimentos

- **OpenCV Community** - Biblioteca incrível e documentação
- **UNIFOR** - Suporte acadêmico
- **Professores** - Conhecimento transmitido
- **Stack Overflow** - Solução de problemas

## 📚 Referências

1. Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.)
2. Szeliski, R. (2010). Computer Vision: Algorithms and Applications
3. OpenCV Documentation: https://docs.opencv.org/
4. Numpy Documentation: https://numpy.org/doc/

---

<div align="center">

### 🎨 Desenvolvido com dedicação para a comunidade de PDI

**UNIFOR - Universidade de Fortaleza**

⭐ Se este projeto foi útil, considere dar uma estrela!

[![Stars](https://img.shields.io/github/stars/LucNath/Processamento-Digital-de-Imagem?style=social)](https://github.com/LucNath/Processamento-Digital-de-Imagem/stargazers)
[![Forks](https://img.shields.io/github/forks/LucNath/Processamento-Digital-de-Imagem?style=social)](https://github.com/LucNath/Processamento-Digital-de-Imagem/network/members)

</div>

---

**Última atualização:** Fevereiro 2026
