import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json


##########funcs

# normalizando os dados 0 a 1
# para otimizar
# e fazer o treinamento mais rapido
def normalizar(x):
    x = x.astype('float32')
    x = x / 255.0
    return x


#####
# aqui pego o caminho do materiale extraio ele..
path = "img/material.zip"
zip_object = zipfile.ZipFile(path, 'r')
zip_object.extractall("./")
zip_object.close()

base_imgs = 'material/fer2013.zip'
zip_object2 = zipfile.ZipFile(file=base_imgs, mode='r')
zip_object2.extractall('./')
zip_object2.close()

data = pd.read_csv('fer2013/fer2013.csv')
print(data.tail())

plt.figure(figsize=(12, 6))
plt.hist(data['emotion'], bins=30)
plt.title('Imagens x emoções')

# Classes :['Angry','Disgust','Fear'.'Happy','Sad',Surprise','Neutral']

pixels = data['pixels'].tolist()
# print(pixels)
largura, altura = 48, 48

faces = []
amostras = 0
# um for para retirar os espacoes do numeros
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face, dtype=np.uint8).reshape(largura, altura)
    # face = face.astype(np.uint8)

    faces.append(face)

    # aqui so iremos mostrar as primeiras 10 so pra ver se ta ok
    if amostras < 10:
        # strw = str("face")+str(amostras)
        # cv2.imshow(strw, face)
        amostras += 1

print("numero total de aimagens:", str(len(faces)))
faces = np.asarray(faces)
print(faces.shape)
# para por 1 no final do array para indicar que sao cinzas
# entao temos quantidades .. altura... largura... cor

faces = np.expand_dims(faces, -1)
faces = normalizar(faces)

# agora iremos usar o pandas...
# e iremos criar um dummies
# Classes :['Angry','Disgust','Fear'.'Happy','Sad',Surprise','Neutral']
# quando a temos a sequencia :[1,0,0.0,0,0,0] indicara felicidade
# [0,1,0.0,0,0,0] indicara felicidade... e assim em diante
# entao teremos 7 NEURONIOS NA CAMADA DE SAIDA QUE INDICARA A POSSIBILIDADE DE PERTENCER A CADA CLASSES
emocoes = pd.get_dummies(data['emotion']).values

##dividindo para testar, 0.1 seria 10% da base de dados
x_train, x_test, y_train, y_test = train_test_split(faces, emocoes, test_size=0.1, random_state=42)

##entao separamos alguns para treino.. algumas para testes, outra para validacao
_, x_val, _, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=41)
print('img para treinamento', len(x_train))
print('img para teste', len(x_test))
print('img para validacao', len(x_val))

num_features = 64
num_labels = 7  ##numeros de classes
batch_size = 64  ## quantos em quantos registros fazer a atualizacao na base de dados ou seja pega o quanto usa/64 ,
# e analizara em packs se fossem 128 .. seriam 2 packs de 64
epochs = 100  ## serao 100 epocas
width, heigth = 48, 48  ##tamanho das imagens

cv2.waitKey(30000)
model = Sequential()
## add camada de convolucao
# e usando funcao do relu para quando tiver partes pretas serao negativas, entao botamos 0 que 'e cinza ignorando
# as sombras
# para regulariazer o kernel regularize, aumenta a punicao ainda mais.
model.add(
    Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, heigth, 1),
           data_format='channels_last', kernel_regularizer=l2(0.01)))

## segunda camada de convolucao, poderia padding valid ou same, optei pelo filtro do same, pois same mata a ultima parte
# ex:
# 1  2   3                                                                               0
# 4  5   6                                                                               0
# 7  8   9  ele vai matar 3 coluna..., quando usa o same ele adiciona uma colunas zerada.0
# 10 11 12                                                                               0
# entao diminui a perda de informacao
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))

##ele normaliza, igual quando tranformamos pra float  e usamos valores entre 0 e 1
model.add(BatchNormalization())

##Poling  iremos diminuir ainda mais dimensionalidade  e ira sempre pegar o valor maior para representar 2x2
# entao vai dividindo em quadrantes e pegando maiores valores
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# essa camada diminuira o overfitting
# zeramos aneuronios para nao se apegar muito aos dados, padrao seria 0.2
model.add(Dropout(0.5))

# adicionamos mais uma camada convolucao duplicando os filtros
model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# batch para normalizar
model.add(BatchNormalization())
# adicionamos mais uma camada convolucao duplicando os filtros
model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# batch para normalizar
model.add(BatchNormalization())
# depois fechamos com max fooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# zeramos aneuronios para nao se apegar muito aos dados, padrao seria 0.2
model.add(Dropout(0.5))

# adicionamos mais uma camada convolucao quadruplicando  os filtros
model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# batch para normalizar
model.add(BatchNormalization())
# adicionamos mais uma camada convolucao quadruplicando os filtros
model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# batch para normalizar
model.add(BatchNormalization())
# depois fechamos com max fooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# zeramos aneuronios para nao se apegar muito aos dados, padrao seria 0.2
model.add(Dropout(0.5))

# adicionamos mais uma camada convolucao colocando 6x os filtros de filtros
model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# batch para normalizar
model.add(BatchNormalization())
# adicionamos mais uma camada convolucao colocando  6x os filtros  de filtros
model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# batch para normalizar
model.add(BatchNormalization())
# depois fechamos com max fooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# zeramos aneuronios para nao se apegar muito aos dados, padrao seria 0.2
model.add(Dropout(0.5))

##Flatten, primeira camdada
model.add(Flatten())
# camada densa, segunda camada ou primeira camada oculta
model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2 * 2 * num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2 * num_features, activation='relu'))
model.add(Dropout(0.5))

##como teremos 7 classes(emocoes), usaremos softmax
# e gerara uma porcentagem para cada uma emocao/classes
model.add(Dense(num_labels, activation='softmax'))
model.summary()
