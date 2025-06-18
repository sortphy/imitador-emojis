import cv2
import random
import threading
import time
from deepface import DeepFace

# Emoções com seus emojis
EMOJI_EXPRESSIONS = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "fear": "😱",
    "disgust": "🤢",
    "neutral": "😐"
}

# Rodadas aleatórias
ROUNDS = random.choices(list(EMOJI_EXPRESSIONS.keys()), k=20)

# Inicialização
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
score = 0
emotion_detected = "..."
emoji_detected = "😐"
last_detection_time = 0
face_box = None

# Controle para análise assíncrona
lock = threading.Lock()

def detect_emotion_async(frame):
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

print("🎮 Iniciando Jogo: Imitador de Emojis!")

for i, target_emotion in enumerate(ROUNDS, start=1):
    target_emoji = EMOJI_EXPRESSIONS[target_emotion]
    matched = False
    round_start = time.time()

    while time.time() - round_start < 7:  # 7 segundos por rodada
        ret, frame = cap.read()
        if not ret:
            continue

        # Redimensiona para melhor desempenho
        small_frame = cv2.resize(frame, (640, 480))

        # Faz nova detecção a cada 1 segundo
        if time.time() - last_detection_time > 1:
            last_detection_time = time.time()
            threading.Thread(target=detect_emotion_async, args=(small_frame,)).start()

        # Texto UI
        feedback = "✅ ACERTOU!" if emotion_detected == target_emotion else f"Imite: {target_emotion.upper()} {target_emoji}"
        color = (0, 255, 0) if emotion_detected == target_emotion else (50, 50, 255)

        # Mostra informações na tela
        cv2.putText(frame, f"Rodada {i}/20", (20, 40), font, 0.9, (200, 255, 255), 2)
        cv2.putText(frame, feedback, (20, 80), font, 0.9, color, 2)

        # Mostra emoji ao lado do rosto detectado
        if face_box:
            x, y, w, h = face_box["x"], face_box["y"], face_box["w"], face_box["h"]
            emoji_pos = (x + w + 10, y + int(h / 2))
            cv2.putText(frame, emoji_detected, emoji_pos, font, 2, (255, 255, 255), 2)

        cv2.imshow("😄 Imitador de Emojis - Pressione Q para sair", frame)

        if emotion_detected == target_emotion:
            score += 1
            time.sleep(1)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Encerramento
cap.release()
cv2.destroyAllWindows()
print(f"\n🏁 Fim do jogo! Pontuação final: {score}/20")
