import cv2
import random
import threading
import time
from deepface import DeepFace

# Emoções suportadas
EMOJI_EXPRESSIONS = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "fear": "😱",
    "disgust": "🤢",
    "neutral": "😐"
}

# Gera 20 rodadas aleatórias
ROUNDS = random.choices(list(EMOJI_EXPRESSIONS.keys()), k=20)

# Variáveis globais
emotion_detected = "..."
emoji_detected = "😐"
face_box = None
last_detection_time = 0

# Função de detecção assíncrona
def detect_emotion(frame):
    global emotion_detected, emoji_detected, face_box
    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        emotion_detected = result[0]["dominant_emotion"]
        face_box = result[0]["region"]
        emoji_detected = EMOJI_EXPRESSIONS.get(emotion_detected, "😐")
    except:
        emotion_detected = "..."
        emoji_detected = "😐"
        face_box = None

# Inicializa a webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
score = 0

print("🎮 Iniciando o Jogo de Imitação de Emojis!")

for i, target_emotion in enumerate(ROUNDS, start=1):
    matched = False
    emoji_target = EMOJI_EXPRESSIONS[target_emotion]
    round_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        elapsed = int(time.time() - round_start)
        remaining = 10 - elapsed

        # Redimensiona para performance
        small_frame = cv2.resize(frame, (640, 480))

        # Atualiza detecção 1x por segundo
        if time.time() - last_detection_time > 1:
            last_detection_time = time.time()
            threading.Thread(target=detect_emotion, args=(small_frame,)).start()

        # Verifica se acertou
        if emotion_detected == target_emotion:
            matched = True

        # Desenha interface elegante
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (630, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Header
        cv2.putText(frame, f"Rodada {i}/20", (20, 40), font, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Tempo restante: {remaining}s", (400, 40), font, 0.9, (255, 255, 255), 2)

        # Instrução
        cv2.putText(frame, f"Imite: {target_emotion.upper()} {emoji_target}", (20, 80), font, 0.9, (255, 255, 255), 2)

        # Emoji flutuante ao lado do rosto
        if face_box:
            x, y, w, h = face_box["x"], face_box["y"], face_box["w"], face_box["h"]
            emoji_pos = (x + w + 15, y + int(h / 2))
            cv2.putText(frame, emoji_detected, emoji_pos, font, 2, (255, 255, 255), 2)

        cv2.imshow("😄 Imitador de Emojis - Pressione Q para sair", frame)

        # Acertou
        if matched:
            score += 1
            feedback = "✅ ACERTOU!"
            break

        # Errou (tempo acabou)
        if remaining <= 0:
            feedback = "❌ ERROU!"
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Exibe resultado por 2 segundos
    for _ in range(40):
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.rectangle(frame, (100, 150), (550, 300), (0, 0, 0), -1)
        cv2.putText(frame, feedback, (160, 240), font, 2, (0, 255, 0) if matched else (0, 0, 255), 3)
        cv2.imshow("😄 Imitador de Emojis", frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

# Final do jogo
cap.release()
cv2.destroyAllWindows()
print(f"\n🏁 Fim do jogo! Você acertou {score} de 20 expressões.")
