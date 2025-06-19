import cv2
import os
import time
import random
import numpy as np
from tensorflow.keras.models import load_model

# --- Configurações iniciais ---

# Carrega o modelo treinado
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

# Mapeia índices de saída para labels (pastas em dataset/train)
CLASS_NAMES = sorted(os.listdir(os.path.join("dataset", "train")))
# Ex.: ['angry','grimace',...,'neutral']
num_classes = len(CLASS_NAMES)

# Carrega emojis como imagens RGBA
EMOJI_DIR = "emojis"
emoji_imgs = {}
for name in CLASS_NAMES:
    path = os.path.join(EMOJI_DIR, f"{name}.png")
    if os.path.exists(path):
        emoji_imgs[name] = cv2.imread(path, cv2.IMREAD_UNCHANGED)

# Detector de faces Haar Cascade (OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Número de rodadas
ROUNDS = 5
targets = random.sample(CLASS_NAMES, k=ROUNDS)

# Parâmetros de jogo
TIMER_SEC = 10
font = cv2.FONT_HERSHEY_SIMPLEX

# Função para sobrepor PNG com alpha
def overlay_image_alpha(img, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        return
    roi = img[y:y+h, x:x+w]
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        roi[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * roi[:, :, c])
    img[y:y+h, x:x+w] = roi

# Loop do jogo
cap = cv2.VideoCapture(0)
score = 0

print("▶️  Iniciando Main Game")

for idx, target in enumerate(targets, start=1):
    start_time = time.time()
    matched = False
    print(f" Rodada {idx}/{ROUNDS}: imite '{target}'")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Desenha HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (630, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Texto HUD
        remaining = TIMER_SEC - int(time.time() - start_time)
        cv2.putText(frame, f"Rodada {idx}/{ROUNDS}", (20, 40), font, 0.9, (0,255,255), 2)
        cv2.putText(frame, f"Imite: {target.upper()}", (20, 80), font, 0.9, (255,255,255), 2)
        cv2.putText(frame, f"{remaining}s", (frame.shape[1]//2-30, frame.shape[0]//2),
                    font, 2, (0,255,255), 4)

        # Mostra emoji alvo no canto superior direito
        if target in emoji_imgs:
            em = emoji_imgs[target]
            overlay_image_alpha(frame, em, frame.shape[1]-em.shape[1]-10, 10)

        # Para cada face detectada, faz previsão
        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (128,128))
            face_norm = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_norm, axis=0)

            preds = model.predict(face_input, verbose=0)[0]
            pred_class = CLASS_NAMES[np.argmax(preds)]

            # Se acertou, marca matched e mostra feedback
            if pred_class == target and not matched:
                matched = True
                score += 1
                match_time = time.time()

            # Desenha retângulo ao redor da face
            color = (0,255,0) if pred_class == target else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

            # Se já acertou, mostra feedback ao lado
            if matched and time.time() - match_time < 2:
                cv2.putText(frame, "✅", (x+w+5, y+int(h/2)), font, 2, (0,255,0), 3)

        cv2.imshow("Imitador (Modelo Próprio)", frame)

        # Condições de saída da rodada
        if matched and time.time() - match_time >= 2:
            break
        if remaining <= 0:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n🏁 Jogo interrompido. Sua pontuação: {score}/{idx-1}")
            exit()

    # pequeno delay antes da próxima rodada
    time.sleep(0.5)

# Fim do jogo
cap.release()
cv2.destroyAllWindows()
print(f"\n🏁 Jogo encerrado! Sua pontuação final: {score}/{ROUNDS}")
