import numpy as np

class RedeNeural:
    # Uma implementação de uma rede neural simples com uma camada oculta, com nomes de variáveis em português para fins didáticos.
    def __init__(self, dados_entrada: np.ndarray, rotulos:np.ndarray, neuronios_ocultos: int = 10, neuronios_saida: int = 2):
        self.dados_entrada = dados_entrada
        self.rotulos = rotulos
        self.neuronios_camada_oculta = neuronios_ocultos
        self.neuronios_camada_saida = neuronios_saida
        self.neuronios_camada_entrada = self.dados_entrada.shape[1]

        # --- Inicializa os pesos e bias ---
        # Usando a Inicialização de Xavier/Glorot
        self.Pesos1 = np.random.randn(self.neuronios_camada_entrada, self.neuronios_camada_oculta) / np.sqrt(self.neuronios_camada_entrada)
        self.Bias1 = np.zeros((1, self.neuronios_camada_oculta))
        self.Pesos2 = np.random.randn(self.neuronios_camada_oculta, self.neuronios_camada_saida) / np.sqrt(self.neuronios_camada_oculta)
        self.Bias2 = np.zeros((1, self.neuronios_camada_saida))
        
        # Dicionário para fácil acesso aos parâmetros do modelo
        self.dicionario_modelo = {'Pesos1': self.Pesos1, 'Bias1': self.Bias1, 'Pesos2': self.Pesos2, 'Bias2': self.Bias2}
        
        # Atributos para armazenar valores intermediários (inicializados para clareza)
        self.Z1: np.ndarray = np.array([])
        self.A1: np.ndarray = np.array([]) # A de "Ativação"
        self.Z2: np.ndarray = np.array([])

    def propagacao_direta(self, dados_teste=None):
        # Executa o forward pass (propagação direta) na rede.
        dados = self.dados_entrada if dados_teste is None else dados_teste

        # Equação da reta (camada 1)
        self.Z1 = dados.dot(self.Pesos1) + self.Bias1

        # Função de ativação (camada 1)
        self.A1 = np.tanh(self.Z1)

        # Equação da reta (camada 2)
        self.Z2 = self.A1.dot(self.Pesos2) + self.Bias2

        # Função de ativação Softmax na saída para obter probabilidades
        valores_exponenciais = np.exp(self.Z2 - np.max(self.Z2, axis=1, keepdims=True))
        saida_softmax = valores_exponenciais / np.sum(valores_exponenciais, axis=1, keepdims=True)

        return saida_softmax

    def calcular_perda(self, saida_softmax):
        # Calcula a perda usando a entropia cruzada (Cross-Entropy).
        numero_amostras = self.rotulos.shape[0]
        
        # Seleciona as probabilidades previstas para as classes corretas
        probabilidades_classes_corretas = saida_softmax[range(numero_amostras), self.rotulos]

        # Evita o cálculo de log(0)
        probabilidades_classes_corretas = np.clip(probabilidades_classes_corretas, 1e-15, 1 - 1e-15)
        
        # Calcula a perda logarítmica
        log_probabilidades = -np.log(probabilidades_classes_corretas)

        # Calcula o custo médio sobre todas as amostras
        custo = np.sum(log_probabilidades) / numero_amostras
        return custo

    def retropropagacao(self, saida_softmax: np.ndarray, taxa_aprendizado: float = 0.01) -> None:
        # Executa o backpropagation para calcular e aplicar os gradientes.
        numero_amostras = self.dados_entrada.shape[0]
        
        # Gradiente da camada de saída (erro_saida = softmax - y_one_hot)
        erro_camada_saida = np.copy(saida_softmax)
        erro_camada_saida[range(numero_amostras), self.rotulos] -= 1
        erro_camada_saida /= numero_amostras # Normaliza o gradiente

        # Gradientes para os pesos e bias da segunda camada
        gradiente_Pesos2 = (self.A1.T).dot(erro_camada_saida)
        gradiente_Bias2 = np.sum(erro_camada_saida, axis=0, keepdims=True)

        # Propagação do erro para a camada oculta
        erro_camada_oculta = erro_camada_saida.dot(self.Pesos2.T) * (1 - np.power(self.A1, 2))

        # Gradientes para os pesos e bias da primeira camada
        gradiente_Pesos1 = (self.dados_entrada.T).dot(erro_camada_oculta)
        gradiente_Bias1 = np.sum(erro_camada_oculta, axis=0, keepdims=True)

        # Atualização dos pesos e bias (descida de gradiente)
        self.Pesos1 -= taxa_aprendizado * gradiente_Pesos1
        self.Bias1 -= taxa_aprendizado * gradiente_Bias1
        self.Pesos2 -= taxa_aprendizado * gradiente_Pesos2
        self.Bias2 -= taxa_aprendizado * gradiente_Bias2

    def treinar(self, epocas: int, taxa_aprendizado: float = 0.01):
        # Loop principal de treinamento do modelo.
        for epoca_atual in range(epocas):
            # Propagação direta
            saidas_rede = self.propagacao_direta()
            
            # Cálculo da perda
            perda = self.calcular_perda(saidas_rede)
            
            # Retropropagação
            self.retropropagacao(saidas_rede, taxa_aprendizado)

            # Imprime o status do treino periodicamente
            if (epoca_atual + 1) % max(1, epocas // 10) == 0 or epoca_atual == 0:
                previsao = np.argmax(saidas_rede, axis=1)
                acertos = (previsao == self.rotulos).sum()
                acuracia = acertos / self.rotulos.shape[0]
                print(f'Época: [{epoca_atual + 1}/{epocas}] | Perda: {perda:.4f} | Acurácia: {acuracia:.4f}')
        
        # Retorna a predição final após o término do treino
        saidas_finais = self.propagacao_direta()
        previsao_final = np.argmax(saidas_finais, axis=1)
        return previsao_final
    
    def prever(self, dados_para_prever):
        # Faz a predição para um novo conjunto de dados.
        saidas_rede = self.propagacao_direta(dados_teste=dados_para_prever)
        return np.argmax(saidas_rede, axis=1)
    
    def salvar_modelo(self, caminho_arquivo: str) -> None:
        # Salva os pesos e biases atuais do modelo em um arquivo .npz.
        print(f"Salvando o modelo em '{caminho_arquivo}'...")
        # np.savez permite salvar múltiplos arrays em um único arquivo, usando palavras-chave para recuperá-los depois.
        np.savez(caminho_arquivo, 
                 Pesos1=self.Pesos1, 
                 Bias1=self.Bias1, 
                 Pesos2=self.Pesos2, 
                 Bias2=self.Bias2)
        print("Modelo salvo com sucesso!")

    def carregar_modelo(self, caminho_arquivo: str) -> None:
        # Carrega os pesos e biases de um arquivo .npz para o modelo atual.
        
        print(f"Carregando modelo de '{caminho_arquivo}'...")
        # np.load carrega o arquivo .npz
        dados_carregados = np.load(caminho_arquivo)
        
        # Atribui os pesos e biases salvos aos atributos da classe
        self.Pesos1 = dados_carregados['Pesos1']
        self.Bias1 = dados_carregados['Bias1']
        self.Pesos2 = dados_carregados['Pesos2']
        self.Bias2 = dados_carregados['Bias2']
        print("Modelo carregado com sucesso!")