
import cv2
import numpy as np

CAMERA_URL = "http://192.168.101.83:8080/video"  # Coloque seu IP real
AREA_MINIMA = 500

camera = cv2.VideoCapture(CAMERA_URL)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("Erro ao conectar. Confira o endereço da câmera.")
    raise SystemExit

print("Câmera conectada! Pressione Q para sair.")

contador_quadros = 0
quantidade = 0
imagem_processada = None

while True:
    sucesso, imagem = camera.read()

    if not sucesso:
        print("Falha ao receber vídeo.")
        break

    contador_quadros += 1

    # Processa somente um quadro a cada dois.
    if contador_quadros % 2 == 0:
        imagem = cv2.resize(imagem, (640, 360))

        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        borrada = cv2.GaussianBlur(cinza, (5, 5), 0)

        _, mascara = cv2.threshold(
            borrada, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = np.ones((3, 3), np.uint8)
        mascara = cv2.morphologyEx(
            mascara, cv2.MORPH_OPEN, kernel
        )

        contornos, _ = cv2.findContours(
            mascara,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        quantidade = 0

        for contorno in contornos:
            if cv2.contourArea(contorno) < AREA_MINIMA:
                continue

            quantidade += 1
            x, y, largura, altura = cv2.boundingRect(contorno)

            cv2.rectangle(
                imagem, (x, y),
                (x + largura, y + altura),
                (0, 255, 0), 2
            )

        cv2.putText(
            imagem,
            f"Regioes detectadas: {quantidade}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        imagem_processada = imagem

    if imagem_processada is not None:
        cv2.imshow("Contador BM36", imagem_processada)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
