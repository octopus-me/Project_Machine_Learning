import numpy as np
from typing import List

class Neuronio:
    def __init__(self, num_pesos: int, vies: float) -> None:
        self.pesos = np.random.randn(num_pesos) * 0.01  # Pesos iniciais pequenos e aleatórios
        self.vies = vies

class Camada:
    def __init__(self, num_entradas: int, num_neuronios: int):
        self.neuronios = [Neuronio(num_entradas, 0.0) for _ in range(num_neuronios)]

class Modelo:
    def __init__(self, classes: List[int], tamanhos_camadas: List[int]):
        assert len(classes) > 1
        self.classes = classes
        self.tamanhos_camadas = tamanhos_camadas

        # Criação das camadas
        self.camadas = []
        for i in range(len(tamanhos_camadas) - 1):
            self.camadas.append(Camada(tamanhos_camadas[i], tamanhos_camadas[i+1]))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_derivada(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _forward(self, x: np.ndarray) -> List[np.ndarray]:
        ativacoes = [x]
        for camada in self.camadas[:-1]:
            x = self._relu(np.dot(x, np.array([neuronio.pesos for neuronio in camada.neuronios]).T) + \
                           np.array([neuronio.vies for neuronio in camada.neuronios]))
            ativacoes.append(x)
        
        # Última camada com softmax
        camada_saida = self.camadas[-1]
        x = self._softmax(np.dot(x, np.array([neuronio.pesos for neuronio in camada_saida.neuronios]).T) + \
                          np.array([neuronio.vies for neuronio in camada_saida.neuronios]))
        ativacoes.append(x)
        return ativacoes

    def _calcular_perda(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        perda = np.sum(log_likelihood) / m
        return perda

    def _backward(self, ativacoes: List[np.ndarray], y_true: np.ndarray) -> List[np.ndarray]:
        gradientes = []
        m = y_true.shape[0]

        # Inicializa o gradiente da camada de saída
        camada_saida = self.camadas[-1]
        delta = ativacoes[-1]
        delta[range(m), y_true] -= 1
        delta /= m

        for i in reversed(range(len(self.camadas))):
            camada = self.camadas[i]
            a = ativacoes[i]
            gradiente_w = np.dot(a.T, delta)
            gradiente_b = np.sum(delta, axis=0, keepdims=True)
            gradientes.append((gradiente_w, gradiente_b))

            if i > 0:
                delta = np.dot(delta, np.array([neuronio.pesos for neuronio in camada.neuronios])) * self._relu_derivada(a)
        
        gradientes.reverse()
        return gradientes

    def treinar(self, imagens_treino: List[np.ndarray], rotulos_treino: List[int], epocas: int = 100, taxa_aprendizagem: float = 0.01) -> None:
        imagens_treino = np.array(imagens_treino)
        rotulos_treino = np.array(rotulos_treino)
        
        for epoca in range(epocas):
            # Passagem para frente
            ativacoes = self._forward(imagens_treino)
            
            # Cálculo da perda
            perda = self._calcular_perda(ativacoes[-1], rotulos_treino)
            print(f"Epoca {epoca+1}/{epocas}, Perda: {perda}")
            
            # Backpropagation
            gradientes = self._backward(ativacoes, rotulos_treino)
            
            # Atualização dos pesos e vieses
            for i, camada in enumerate(self.camadas):
                for j, neuronio in enumerate(camada.neuronios):
                    neuronio.pesos -= taxa_aprendizagem * gradientes[i][0][:, j]
                    neuronio.vies -= taxa_aprendizagem * gradientes[i][1][0, j]

    def classificar(self, imagens_teste: List[np.ndarray]) -> List[int]:
        rotulos_gerados = []
        for imagem in imagens_teste:
            imagem_achata = imagem.reshape(1, -1)
            predicao = self._forward(imagem_achata)[-1]
            rotulo = np.argmax(predicao)
            rotulos_gerados.append(rotulo)
        return rotulos_gerados