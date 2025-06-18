import cv2
from deepface import DeepFace

# Emoções mapeadas para emojis
emoji_map = {
    'happy': '😄',
    'sad': '😢',
    'angry': '😠',
    'surprise': '😲',
    'fear': '😱',
    'disgust': '🤢',
    'neutral': '😐'
}

# Inicia a webcam
cap = cv2.VideoCapture(0)
print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Detecta a emoção
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        emoji = emoji_map.get(emotion, '😐')  # padrão: neutro

        # Exibe o texto e emoji na imagem
        cv2.putText(frame, f'{emotion.upper()} {emoji}', (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    except Exception as e:
        print("Erro:", e)

    # Mostra o vídeo com a emoção
    cv2.imshow('Expressão Facial + Emoji', frame)

    # Tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finaliza
cap.release()
cv2.destroyAllWindows()