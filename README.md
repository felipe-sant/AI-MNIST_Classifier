<div align="center">

# Classificador de Dígitos Manuscritos (MNIST)

![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![tensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

</div>

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

   Caso queira testar:
   ```
   python src/evaluate_model.py
   ```

4. Execute o comando
    ```
    uvicorn index:app
    ```

## Estrutura

- `src/train_model.py`: Treinamento do modelo
- `src/evaluate_model.py`: Avaliação do modelo treinado
- `src/utils.py`: Funções auxiliares

<hr>

<div align="center">
    developed by <a href="https://github.com/felipe-sant?tab=followers">@felipe-sant</a>
</div>
