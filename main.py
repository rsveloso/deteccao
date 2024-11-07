from ultralytics import YOLO
import cv2

#para trabalhar com YOLO erro dll, baixar o libomp140.x86_64.dll e colocar no system32

#Carregar modelos
modelo = YOLO('yolov10n.pt')
# yolov8n.pt - menor

cap = cv2.VideoCapture(0)

frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar quadro da webcam")
        break

    detecoes = modelo(frame, verbose=False)[0]

    cv2.imshow('Detecções', detecoes.plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()