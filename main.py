import cv2
import random
import time
from deepface import DeepFace

# Mapeamento de emoções com emojis
EMOJI_EXPRESSIONS = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "fear": "😱",
    "disgust": "🤢",
    "neutral": "😐"
}

# Lista expandida com 20 rodadas aleatórias
ROUNDS = random.choices(list(EMOJI_EXPRESSIONS.keys()), k=20)

# Inicializa a webcam
cap = cv2.VideoCapture(0)
score = 0
font = cv2.FONT_HERSHEY_SIMPLEX

print("Iniciando o jogo. Prepare-se para imitar emojis!")

for i, target_emotion in enumerate(ROUNDS, start=1):
    emoji = EMOJI_EXPRESSIONS[target_emotion]
    start_time = time.time()
    matched = False

    while time.time() - start_time < 7:  # 7 segundos para imitar
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            # Analisa a emoção detectada
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            detected_emotion = result[0]['dominant_emotion']

            # Checa se corresponde
            matched = detected_emotion == target_emotion
            feedback = "✅ ACERTOU!" if matched else f"Tente imitar: {emoji}"
        except:
            feedback = "😐 Detecção falhou..."

        # Mostra o emoji-alvo e feedback
        cv2.putText(frame, f"Rodada {i}/20", (20, 40), font, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Imite: {target_emotion.upper()} {emoji}", (20, 90), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, feedback, (20, 140), font, 1, (0, 255, 0) if matched else (0, 0, 255), 2)
        cv2.imshow("Jogo de Emoji - Pressione Q para sair", frame)

        if matched:
            score += 1
            time.sleep(1)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final do jogo
cap.release()
cv2.destroyAllWindows()
print(f"\nPontuação final: {score} de 20")
