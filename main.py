import cv2
import time
import random
import threading
from deepface import DeepFace

# Emoções e arquivos
EMOJI_FILES = {
    "happy": "emojis/happy.png",
    "grimace": "emojis/grimace.png",
    "handmouth": "emojis/handmouth.png",
    "sad": "emojis/sad.png",
    "angry": "emojis/angry.png",
    "surprise": "emojis/surprise.png",
    "neutral": "emojis/neutral.png"
}
EMOJI_IMAGES = {k: cv2.imread(v, cv2.IMREAD_UNCHANGED) for k, v in EMOJI_FILES.items()}
ROUNDS = random.sample(list(EMOJI_FILES.keys()), k=5)

emotion_detected = "..."
face_box = None
last_detection_time = 0

# Função overlay de PNG com alpha
def overlay_image_alpha(img, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        return
    roi = img[y:y+h, x:x+w]
    overlay_rgb = overlay[:, :, :3]
    mask = overlay[:, :, 3:] / 255.0
    roi[:] = (1.0 - mask) * roi + mask * overlay_rgb

# Detecção assíncrona
def detect_emotion(frame):
    global emotion_detected, face_box
    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        emotion_detected = result[0]["dominant_emotion"]
        face_box = result[0]["region"]
    except:
        emotion_detected = "..."
        face_box = None

# Início do jogo
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
score = 0

print("🎮 Início do jogo: Imitador de Emojis")

for i, target_emotion in enumerate(ROUNDS, start=1):
    emoji_img = EMOJI_IMAGES[target_emotion]
    matched = False
    round_start = time.time()
    show_feedback = False
    feedback_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        elapsed = int(time.time() - round_start)
        remaining = 10 - elapsed
        small_frame = cv2.resize(frame, (640, 480))

        if time.time() - last_detection_time > 1:
            last_detection_time = time.time()
            threading.Thread(target=detect_emotion, args=(small_frame,)).start()

        # Verifica acerto
        if emotion_detected == target_emotion and not matched:
            matched = True
            score += 1
            show_feedback = True
            feedback_timer = time.time()

        # Barra de cima
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (630, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"Rodada {i}/5", (20, 40), font, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Imite: {target_emotion.upper()}", (20, 80), font, 0.9, (255, 255, 255), 2)

        # Timer grande no centro
        cv2.putText(frame, f"{remaining}s", (frame.shape[1] // 2 - 40, frame.shape[0] // 2),
                    font, 2.5, (0, 255, 255), 4)

        # Mostra emoji ao lado do rosto
        if face_box and emoji_img is not None:
            x, y, w, h = face_box["x"], face_box["y"], face_box["w"], face_box["h"]
            overlay_image_alpha(frame, emoji_img, x + w + 15, y + h // 4)

            # Mensagem de ACERTO ao lado do rosto
            if show_feedback and time.time() - feedback_timer <= 2:
                cv2.putText(frame, "✅ ACERTOU!", (x, y - 10), font, 1, (0, 255, 0), 3)

        # Mostra janela única
        cv2.imshow("😄 Imitador de Emojis", frame)

        # Fim da rodada
        if remaining <= 0 or (matched and time.time() - feedback_timer > 2):
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Final
cap.release()
cv2.destroyAllWindows()
print(f"\n🏁 Jogo encerrado. Você acertou {score}/5 expressões.")
print("Obrigado por jogar! 👋")