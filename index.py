import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suprime avisos e info do TensorFlow
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")          # Suprime warnings do Python (incluindo absl)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from src.functions.indetifyImage import indenfityImage

while True:
    print()
    print("Selecione uma imagem")
    print("--------------------")
    print("1 - cinco.jpg")
    print("2 - sete.webp")
    print("0 - Sair")
    print("--------------------")
    
    text = int(input("Selecione: "))
    
    if text == 0:
        break
    
    if text == 1:
        indenfityImage("data/cinco.jpg")
        
    if text == 2: 
        indenfityImage("data/sete.webp")