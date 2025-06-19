import cv2
import os
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 1) Captura dos nomes de emoji
EMOJI_DIR = "emojis"
emoji_files = sorted([f for f in os.listdir(EMOJI_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
emoji_names = [os.path.splitext(f)[0] for f in emoji_files]

# 2) Cria pastas de destino
BASE_DATASET = os.path.join("dataset", "train")
for name in emoji_names:
    os.makedirs(os.path.join(BASE_DATASET, name), exist_ok=True)

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

# 3) Loop de coleta para cada emoji (permite várias fotos)
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

print("🎓 MODO DE TREINO PERSONALIZADO")
print("→ Espaço: capturar foto   |   N: próximo emoji   |   Q: sair")

for name, file in zip(emoji_names, emoji_files):
    path_img = os.path.join(EMOJI_DIR, file)
    emoji_img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)
    count = len(os.listdir(os.path.join(BASE_DATASET, name)))
    print(f"\n📸 Coletando para '{name}' (já existem {count} imagens).")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        overlay_image_alpha(frame, emoji_img, 10, 10)
        cv2.putText(frame, f"Emoji: {name}", (10, emoji_img.shape[0] + 35), font, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Fotos: {count}", (10, emoji_img.shape[0] + 70), font, 0.8, (255,255,255), 2)
        cv2.putText(frame, "[Espaço] Capturar   [N] Próximo   [Q] Sair",
                    (10, frame.shape[0] - 20), font, 0.6, (200,200,200), 1)

        cv2.imshow("Treino Personalizado", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            filename = f"{int(time.time()*1000)}.jpg"
            dst = os.path.join(BASE_DATASET, name, filename)
            img = cv2.resize(frame, (128,128))
            cv2.imwrite(dst, img)
            count += 1
            print(f"  → {dst}")
            time.sleep(0.3)
        elif key == ord('n'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("\nSaindo da coleta de dados.")
            exit()

cap.release()
cv2.destroyAllWindows()

# 4) Verifica se cada classe tem ao menos 5 imagens
min_imgs = 5
for name in emoji_names:
    c = len(os.listdir(os.path.join(BASE_DATASET, name)))
    if c < min_imgs:
        raise RuntimeError(f"Faltam imagens em '{name}' (tem {c}, precisa de ≥{min_imgs}).")

print("\n✅ Todas as classes têm ≥5 imagens. Iniciando treino...")

# 5) Data augmentation com ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    BASE_DATASET, target_size=(128,128), batch_size=16,
    subset="training", class_mode="categorical"
)
val_gen = datagen.flow_from_directory(
    BASE_DATASET, target_size=(128,128), batch_size=16,
    subset="validation", class_mode="categorical"
)

# 6) Modelo CNN com BatchNorm e Dropout
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    BatchNormalization(), MaxPool2D(),
    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(), MaxPool2D(),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(train_gen.class_indices), activation="softmax")
])
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

# 7) Callbacks para treino eficiente
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# 8) Treino
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks
)

# 9) Salva o modelo final
model.save("model.h5")
print("✅ Treino concluído. Modelo salvo em 'model.h5'.")
