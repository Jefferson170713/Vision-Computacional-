import cv2
import numpy as np
from keras.models import load_model

# Carrega o modelo pré-treinado de reconhecimento facial
#model = load_model('C:/Users/55859/Desktop/Estudos/Visão Computacional/Reconhecimento Facial/Rede Neural/haarcascade_fontalface_default.xml')
model = load_model('./haa')
# Captura imagens da câmera
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    # Redimensiona a imagem para 96x96 (o tamanho de entrada do modelo)
    img = cv2.resize(frame, (96, 96))

    # Normaliza a imagem para que os valores de pixel estejam entre 0 e 1
    img = np.array(img, dtype=np.float32) / 255.0

    # Adiciona uma dimensão extra para representar o batch_size
    img = np.expand_dims(img, axis=0)

    # Realiza a inferência do modelo na imagem
    prediction = model.predict(img)

    # Verifica se a imagem contém um rosto humano
    if prediction[0][0] > 0.5:
        cv2.putText(frame, "Rosto detectado!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Rosto nao detectado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exibe a imagem na janela
    cv2.imshow('Webcam', frame)

    # Verifica se a tecla 'Esc' foi pressionada para sair do programa
    if cv2.waitKey(1) == 27:
        break

# Libera a captura da câmera e fecha todas as janelas
video.release()
cv2.destroyAllWindows()
