# This is a sample Python script.
import cv2
import numpy as np
import pandas as pd
import zipfile
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# aqui pego o caminho do materiale extraio ele..
path = ("img/material.zip")
zip_object = zipfile.ZipFile(path, 'r')
zip_object.extractall("./")
zip_object.close()

# pego imagem de teste.. , e depois mostro a imagem
img = cv2.imread('Material/testes/teste04.jpg')
cv2.imshow('img', img)

# uso um arquivo de detecacao de face..., busco um modelo ja pronto,passo tipo de detector, expressoes que quero
# detectar
cascate_face = "Material/haarcascade_frontalface_default.xml"
modelo = 'Material/modelo_01_expressoes.h5'
face_detection = cv2.CascadeClassifier(cascate_face)
classificador = load_model(modelo, compile=False)
expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]

# copio a imagem
original = img.copy()

# aqui estou tentando detectar onde esta a face, passando tamanho da imagem, e o tamanho min da face(se for menor ele nao ler)
face = face_detection.detectMultiScale(original, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
print(face)

# podemos passar imagem pra cinza pra ser mais rapido pois a escala  dela vira 1 ... e nao 3, quando temos 3, usamos
# rgb que 'e mais dificil que treinar do que apenas cinza
cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imshow("cinza", cinza)

# x sera o primeiro sera o valor de X, o segundo valor de Y, o 3 'e o fim de x e 4 fim de y
##usamos for para mais de uma face
for (x, y, w, h) in face:
    cinzaShape = cinza.shape
    # faceInitX = face[x][x]
    # faceInitY = face[y][y]
    # faceFinishX = face[x][w]
    # faceFinishY = face[y][h]

    # aqui falo pra pegar inicial a inicial ate o final ...
    # print(faceInitX, ":", faceInitX, '+', faceFinishX, faceInitY, ":", faceInitY, faceFinishY)
    cinzaRoi = cinza[y:y + h, x:x + w]
    cv2.imshow("cinzaRoi", cinzaRoi)

    # proximo passo 'e diminuir o tamanho da imagem para que podemos usala com treinamento mais rapido

    cinzaRoi = cv2.resize(cinzaRoi, (48, 48))
    cv2.imshow("cinzaRoiCut", cinzaRoi)

    ## depois precisamos coverter o roi de inteiro, para float, para facilitar o treinamento
    cinzaRoi = cinzaRoi.astype("float")
    # e agora diminuiremos os valores dividindo por 255 que 'e o valor max
    cinzaRoi = cinzaRoi / 255
    print(cinzaRoi)

    # passamos agora imagem para matriz... vemos que teremos 3 dimensoes...
    # essas dimensoes tem que dar 48 48 e 1... pois decidimos o tam dela de 48z e y .z = que temos 1 cor
    cinzaRoi = img_to_array(cinzaRoi)
    print(cinzaRoi)
    print(cinzaRoi.shape)

    # agora pedimos ao np que expanda as dimens da matriz,e adicionara mais uma dimencao, que sera referente a quantas imagem temos
    cinzaRoi = np.expand_dims(cinzaRoi, axis=0)
    print(cinzaRoi.shape)

    ##aqui ele calcula as probabilidades
    preds = classificador.predict(cinzaRoi)[0]
    print(preds)

    ## retorna maior taxa
    emotion_probability = np.max(preds)
    print(emotion_probability)
    ##maior valor das previsoes
    print(preds.argmax())
    # buscando qual expressao do index retornado
    label = expressoes[preds.argmax()]
    print(label)

    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(original, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("original2", original)
    probabilidades = np.ones((200, 300, 3), dtype='uint8') * 255
    cv2.imshow("original3", original)
    if len(face == 1):
        for (i, (emotion, prob)) in enumerate(zip(expressoes, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(probabilidades, (7, (i * 35) + 5), (w, (i * 35) + 35), (200, 250, 20), - 1)
            cv2.putText(probabilidades, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),1,
                        cv2.LINE_AA)
cv2.imshow("original3", probabilidades)
cv2.waitKey(15000)
