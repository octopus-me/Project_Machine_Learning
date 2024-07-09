# Projeto: Redes Neurais para o Reconhecimento de Dígitos Manuscritos

## Introdução
Este projeto faz parte da disciplina **MAC0460 - Introdução ao Aprendizado de Máquina** oferecida pelo Instituto de Matemática e Estatística da Universidade de São Paulo (USP). Ele aborda o problema de reconhecimento de dígitos manuscritos utilizando redes neurais artificiais, especificamente treinadas e testadas com a base de dados MNIST.

## Autores
- Guilherme Wallace
- Thiago Duvanel
- Mikhail Futorny
- Cássio Cancio

## Descrição
O reconhecimento de dígitos manuscritos é um problema fundamental em aprendizado de máquina e visão computacional. Este projeto implementa uma rede neural para classificar dígitos manuscritos utilizando o algoritmo de gradiente descendente estocástico. 

### Resumo
O projeto discute os blocos básicos das redes neurais, incluindo perceptrons e neurônios sigmoidais, e apresenta a arquitetura das redes neurais utilizada. A eficácia do modelo é demonstrada através de experimentos com a base de dados MNIST.

## Metodologia
A metodologia aplicada no projeto inclui a pré-processamento dos dados, definição da arquitetura da rede neural e o algoritmo de treinamento.

### Conjunto de Dados
Utilizamos a base de dados MNIST, que contém 60.000 imagens de treinamento e 10.000 imagens de teste, cada uma com resolução de 28x28 pixels em escala de cinza.

### Pré-processamento dos Dados
Os dados foram normalizados para valores entre 0 e 1, dividindo cada pixel por 255. Os rótulos foram codificados utilizando a técnica de *one-hot encoding*. Adicionalmente, aplicamos "ruído sal e pimenta" para criar variações nos dados e avaliamos a eficácia do modelo na presença de ruído.

### Arquitetura da Rede Neural
A rede neural desenvolvida possui a seguinte arquitetura:
- **Camada de entrada**: 784 neurônios (28x28 pixels)
- **Primeira camada oculta**: 128 neurônios com função de ativação ReLU
- **Segunda camada oculta**: 64 neurônios com função de ativação ReLU
- **Camada de saída**: 10 neurônios (correspondentes aos dígitos 0-9) com função de ativação softmax

### Algoritmo de Treinamento
Utilizamos o algoritmo de gradiente descendente estocástico (SGD) para treinar a rede neural. Os hiperparâmetros principais são:
- **Épocas**: 15
- **Taxa de aprendizado**: 0.01

## Como Executar
Para executar este projeto, siga os passos abaixo:

### Pré-requisitos
- Python 3.x
- Bibliotecas: numpy


### Instruções
1. Clone o repositório:
```
sh
   git clone https://github.com/seu_usuario/ProjetoRedesNeurais.git
   cd ProjetoRedesNeurais
```
2. Instale as dependências necessárias:
```
   pip install numpy
```
   
3.  Execute o script projeto_1.py:
```
  python projeto_1.py
```

### **Estrutura do Projeto**


   ```
ProjetoRedesNeurais/
├── Modelo.py
├── projeto_1.py
├── README.md
└── dados/
    ├── mnist_train.csv
    └── mnist_test.csv
  ```

## Resultados
Os resultados experimentais demonstram a eficácia da rede neural desenvolvida, alcançando uma alta precisão na classificação de dígitos na base de dados MNIST.

## Conclusão
Este projeto destaca o potencial das redes neurais para resolver problemas complexos de reconhecimento de padrões e fornece insights sobre o design e a otimização desses modelos. A implementação de uma rede neural para a classificação de dígitos manuscritos mostrou-se eficiente, mesmo na presença de ruído nos dados.

## Referências
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
- Base de dados MNIST

## Contato
Para mais informações, entre em contato com os autores através dos emails institucionais fornecidos pela Universidade de São Paulo (USP).
