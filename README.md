# Projeto Imitador de Emojis

Este repositório contém um jogo de imitação de expressões faciais usando sua própria webcam e um classificador personalizado treinado sobre suas próprias imagens.

## 🚀 Recursos

* **Modo de Treino Personalizado** (`treino_personalizado.py`):

  * Captura múltiplas imagens de cada emoji escolhido.
  * Salva imagens em `dataset/train/<emoji>/`.
  * Realiza o treino automático de um modelo CNN com data augmentation e callbacks.

* **Modo Jogo Principal** (`main.py`):

  * Carrega o modelo treinado (`model.h5`).
  * Detecta faces usando Haar Cascade do OpenCV.
  * Exibe emoji-alvo e cronômetro.
  * Classifica expressões com seu modelo próprio.
  * Contabiliza pontuação em 5 rodadas.

## 📁 Estrutura de Pastas

```
projeto_imitando_emojis/
├── emojis/                  # PNGs transparentes de cada emoji
│   ├── angry.png
│   ├── happy.png
│   └── ...
├── dataset/
│   └── train/              # Coleta de imagens por emoji
│       ├── angry/
│       ├── happy/
│       └── ...
├── treino_personalizado.py # Script de coleta e treino
├── main_game.py            # Script principal do jogo
├── model.h5                # Modelo treinado (após execução)
├── best_model.h5           # Melhor checkpoint
└── README.md               # Este arquivo
```

## 🛠️ Pré-requisitos

* Python 3.8+ ou 3.9+
* Pacotes Python:

  ```bash
  pip install opencv-python tensorflow numpy
  ```

## ⚙️ Instalação

1. Clone este repositório:

   ```bash
   git clone https://github.com/seuusuario/projeto_imitador_emojis.git
   cd projeto_imitando_emojis
   ```
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

> **Observação:** Caso não haja `requirements.txt`, instale manualmente:
>
> ```bash
> pip install opencv-python tensorflow numpy
> ```

## 🎓 Modo de Treino Personalizado

1. Certifique-se de ter as imagens de emoji em `emojis/`.
2. Execute:

   ```bash
   python treino_personalizado.py
   ```
3. Para cada emoji listado:

   * Pressione **Espaço** para capturar uma foto e salvar.
   * Pressione **N** para avançar ao próximo emoji.
   * Pressione **Q** para sair a qualquer momento.
4. Após capturar pelo menos 5 imagens por classe, o script inicia o treino:

   * Data augmentation aplicado.
   * Callbacks: EarlyStopping, ReduceLROnPlateau e ModelCheckpoint.
   * Modelo salvo em `model.h5` e `best_model.h5`.

## ▶️ Modo Jogo Principal

1. Certifique-se de ter o modelo `model.h5` na raiz do projeto.
2. Execute:

   ```bash
   python main_game.py
   ```
3. O jogo terá 5 rodadas:

   * Um emoji-alvo é exibido no canto.
   * Você tem 10 segundos para imitar a expressão.
   * Face detectada com Haar Cascade.
   * Classificação via o seu modelo.
   * Pontuação final exibida ao terminar.

## 💡 Dicas de Treino e Melhoria

* Colete **mínimo de 100 a 200 imagens** por classe para robustez.
* Utilize **data augmentation** para aumentar diversidade.
* Experimente **transfer learning** com redes pré-treinadas (MobileNet, EfficientNet).
* Ajuste **callbacks** e hiperparâmetros no script de treino.

## 🤝 Contribuição

Contribuições são bem-vindas! Siga estes passos:

1. Fork este repositório.
2. Crie uma branch: `git checkout -b feature/nome_da_feature`.
3. Commit suas mudanças: `git commit -m 'Adiciona nova feature'`.
4. Push para a branch: `git push origin feature/nome_da_feature`.
5. Abra um Pull Request.

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.
