# Rede Neural do Zero em Python

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=yellow)
![NumPy](https://img.shields.io/badge/Numpy-1.24.3-blue?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-orange?logo=matplotlib)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange?logo=scikit-learn)

Este projeto é uma implementação de uma rede neural artificial com uma camada oculta, construída do zero utilizando apenas a biblioteca NumPy. O objetivo é demonstrar um entendimento fundamental dos mecanismos internos de uma rede neural, como a propagação direta (forward pass) e a retropropagação (backpropagation) para o ajuste dos pesos.

O código foi escrito de forma didática, com variáveis e comentários em português, para facilitar o entendimento do fluxo de dados e dos cálculos envolvidos no treinamento do modelo.

## Demonstração

O modelo foi treinado para resolver um problema de classificação não linear, utilizando o dataset "make_moons" do Scikit-learn.

#### Dados Originais
Os dados consistem em duas classes distribuídas em formato de "luas", tornando o problema ideal para um classificador não linear.

![Dados Originais](caminho/para/sua/imagem_dados.png)

#### Fronteira de Decisão Aprendida
Após o treinamento, a rede neural aprende uma fronteira de decisão capaz de separar as duas classes com alta acurácia.

![Fronteira de Decisão](caminho/para/sua/imagem_fronteira.png)

## Principais Funcionalidades

-   **Classe `RedeNeural` construída do zero:** Toda a lógica está encapsulada em uma classe Python reutilizável.
-   **Propagação Direta (Forward Pass):** Calcula a saída da rede para uma dada entrada.
-   **Funções de Ativação:** Utiliza `tanh` na camada oculta e `Softmax` na camada de saída para classificação.
-   **Cálculo da Perda:** Implementa a função de perda de Entropia Cruzada (Cross-Entropy Loss).
-   **Retropropagação (Backpropagation):** Calcula os gradientes e atualiza os pesos e biases do modelo através da descida de gradiente.
-   **Salvamento e Carregamento de Modelo:** Métodos para salvar os pesos treinados em um arquivo `.npz` e carregá-los posteriormente, evitando a necessidade de retreinamento.
-   **Código Didático:** Variáveis e métodos nomeados em português para facilitar a compreensão.

## Tecnologias Utilizadas

-   **Python:** Linguagem principal do projeto.
-   **NumPy:** Utilizada para todas as operações matemáticas e manipulação de arrays.
-   **Matplotlib:** Para a visualização dos dados e da fronteira de decisão.
-   **Scikit-learn:** Apenas para a geração do dataset de exemplo (`make_moons`).
-   **Jupyter Notebook:** Para a demonstração interativa do treinamento e teste do modelo.

## Como Executar o Projeto

Para executar este projeto em sua máquina local, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git](https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git)
    cd SEU-REPOSITORIO
    ```

2.  **Crie um ambiente virtual e instale as dependências:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
    pip install numpy matplotlib scikit-learn jupyter
    ```

3.  **Inicie o Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

4.  **Abra e execute o arquivo `main.ipynb`:** O notebook contém todo o fluxo, desde a geração dos dados, treinamento do modelo, salvamento, carregamento e visualização dos resultados.

## Estrutura do Projeto
