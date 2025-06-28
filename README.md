# Classificador de Dígitos Manuscritos (MNIST)

Este projeto usa Deep Learning (Keras/TensorFlow) para treinar uma rede neural capaz de reconhecer dígitos manuscritos usando o dataset público MNIST.

## Como rodar

1. Crie um ambiente venv com a versão do python anterior a 3.11:
    ```
    py -3.11 -m venv venv
    .\venv\Scripts\activate
    ```

2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

3. Execute o treinamento:
   ```
   python src/train_model.py
   ```

4. Avalie o modelo:
   ```
   python src/evaluate_model.py
   ```

O projeto é didático, ideal para quem está começando em IA e Deep Learning.

## Estrutura

- `src/train_model.py`: Treinamento do modelo
- `src/evaluate_model.py`: Avaliação do modelo treinado
- `src/utils.py`: Funções auxiliares