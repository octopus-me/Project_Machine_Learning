from typing import Any
import gzip
import numpy as np
import matplotlib.pyplot as plt
from Modelo import Modelo

ARQ_IMG_TREINO = './MNIST/train-images-idx3-ubyte.gz'
ARQ_ROT_TREINO = './MNIST/train-labels-idx1-ubyte.gz'
ARQ_IMG_TESTE = './MNIST/t10k-images-idx3-ubyte.gz'
ARQ_ROT_TESTE  = './MNIST/t10k-labels-idx1-ubyte.gz'

def ler_dataset(arquivo_imagens: str, arquivo_rotulos: str) -> tuple[np.ndarray, list[int]]:
    return ler_imagens_mnist(arquivo_imagens), ler_rotulos_mnist(arquivo_rotulos)

def ler_imagens_mnist(nome_arquivo: str) -> np.ndarray:
    with gzip.open(nome_arquivo, 'rb') as f:
        int.from_bytes(f.read(4), byteorder='big')
        num_imagens = int.from_bytes(f.read(4), byteorder='big')
        num_linhas = int.from_bytes(f.read(4), byteorder='big')
        num_colunas = int.from_bytes(f.read(4), byteorder='big')
        buffer = f.read(num_linhas * num_colunas * num_imagens)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_imagens, num_linhas, num_colunas)
        return data

def ler_rotulos_mnist(nome_arquivo: str):
    with gzip.open(nome_arquivo, 'rb') as f:
        int.from_bytes(f.read(4), byteorder='big')
        num_itens: int = int.from_bytes(f.read(4), byteorder='big')
        buffer: bytes = f.read(num_itens)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels.tolist()

def compararResultados(rotulos_corretos: list[int], rotulos_gerados: list[int]) -> float:
    assert len(rotulos_corretos) == len(rotulos_gerados)
    num_rotulos: int = len(rotulos_corretos)
    num_acertos: int = 0
    for i in range(num_rotulos):
        if rotulos_corretos[i] == rotulos_gerados[i]: num_acertos += 1
    return 100*num_acertos/num_rotulos

def exibir_imagem(imagem: np.ndarray, titulo: Any = 'Imagem') -> None:
    plt.imshow(imagem, cmap='gray') # type: ignore
    plt.title(titulo) # type: ignore
    plt.show() # type: ignore

def main() -> None:
    imagens_treino, rotulos_treino = ler_dataset(ARQ_IMG_TREINO, ARQ_ROT_TREINO)
    imagens_teste, rotulos_teste = ler_dataset(ARQ_IMG_TESTE, ARQ_ROT_TESTE)

    # Exibindo algumas imagens de treinamento
    for i in range(5, 10):
        exibir_imagem(imagens_treino[i], rotulos_treino[i])

    # Achatando as imagens para que tenham forma (60000, 784)
    imagens_treino_achatadas = imagens_treino.reshape(imagens_treino.shape[0], -1)
    imagens_teste_achatadas = imagens_teste.reshape(imagens_teste.shape[0], -1)

    modelo = Modelo(list(range(10)), [784, 128, 128, 10])
    modelo.treinar(imagens_treino_achatadas, rotulos_treino) 

    rotulos_gerados_teste = modelo.classificar(imagens_teste_achatadas) 

    porcentagem_acerto = compararResultados(rotulos_teste, rotulos_gerados_teste) 
    print(f"Porcentagem de acerto: {porcentagem_acerto:.2f}%")

if __name__ == "__main__": 
    main()
