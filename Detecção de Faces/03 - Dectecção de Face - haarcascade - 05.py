# ********************* [J]efferson **********************
import cv2

haarcascade = './Arquivos_haarcascades/haarcascade_frontalface_default.xml'
img = cv2.imread('./Fotos/foto_10.png.jpg')
img = cv2.resize(img, (800, 600))

i_c = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = cv2.CascadeClassifier(haarcascade)
deteccao = detector.detectMultiScale(img)

for x, y, w, h in deteccao:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
cv2.imshow("Detecção da imagem da Galera", img)
cv2.waitKey(0)
