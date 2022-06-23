import cv2
import glob
import numpy as np

#Função para escrever nas Imagens
def escreve(img, texto, cor=(0,255,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (10,20), fonte, 0.5, cor,  0, cv2.LINE_AA)


imgRGB = [cv2.imread(file) for file in sorted(glob.glob("imagens/*.jpg"))] #Guardando as imagens na lista

for i in range(len(imgRGB)): #Utilizando diversas imagens no algoritimo de Detecção

    #Imagem: Redimensionando a imagem de Entrada ficando como padrão 350x350
    img_resize = cv2.resize(imgRGB[i], (350, 350))
    

    #Imagem: Diminuindo o brilho
    img_ajustada = img_resize.copy()
    img_ajustada = np.int16(img_ajustada)
    brightness = -60
    contrast = 30
    img_ajustada = img_ajustada * (contrast/127+1) - contrast + brightness
    img_ajustada = np.clip(img_ajustada, 0, 255)
    img_ajustada = np.uint8(img_ajustada)
    

    #Passo 1: Convertendo a imagem RGB para Tons de Cinza
    img = cv2.cvtColor(img_ajustada, cv2.COLOR_BGR2GRAY)
    h_eq = cv2.equalizeHist(img)

    #Passo 2: Suavizando a Imagem em Tons de Cinza com Filtro Gaussiano
    gauss = cv2.GaussianBlur(img, (3,3),0)

    #Passo 3: Detectando Bordas com Canny
    bordas = cv2.Canny(gauss, 100, 190)

    #Passo 4: Detectando Contornos, e Aplicando os Contornos na Imagem Redimensionada
    (hierarquia, contornos, hierarquia) = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgResult = img_resize.copy()
    cv2.drawContours(imgResult, contornos, -1, (255, 0, 0), 2)

    #Imagem: Escrevendo nas Imagens
    escreve(img_resize, "Imagem Original Redimencionada")
    escreve(img_ajustada, "Redimencionada com Brilho Ajustado")
    escreve(img, "Imagem em Tons de Cinza", 255)
    escreve(gauss, "Imagem com filtro Gaussiano", 255)
    escreve(bordas, "Detector de bordas Canny", 255)

    #Original:
    cv2.imshow("Imagem Original", img_resize)
    cv2.waitKey(0)
    cv2.imshow("Imagem Ajustada", img_ajustada)
    cv2.waitKey(0)

    #Imagem: Juntando as Imagens em uma Janela para Facilitar a Visualização
    imgProc = np.vstack([
        np.hstack([img, gauss, bordas])
        ])
    cv2.imshow("Procedimentos Aplicados", imgProc)
    cv2.waitKey(0)

    #Resultado: Comparando a Imagem Redimencionada Original, com a que está com o Contorno em destaque
    escreve(img_resize, "Imagem Original Redimencionada")
    escreve(imgResult, "Imagem Com Contorno")
    imgComp = np.vstack([
        np.hstack([img_resize, imgResult])
        ])
    cv2.imshow("Resultado", imgComp)
    cv2.waitKey(0)

    #picture1 = img.copy()
    #picture2 = img.copy()
    #picture1_norm = picture1/np.sqrt(np.sum(picture1**2))
    #picture2_norm = picture2/np.sqrt(np.sum(picture2**2))
    #picture_result = np.sum(picture2_norm*picture1_norm)
    #print(picture_result)
    cv2.destroyAllWindows()
