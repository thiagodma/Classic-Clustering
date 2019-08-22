#Importando os pacotes que serão utilizados
from stop_words import get_stop_words
from docx import Document
import os, os.path, glob, re, unicodedata, time, nltk
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
#nltk.download('rslp')
#nltk.download('stopwords')

class ClassicClustering():
    """Essa classe contém funcionalidades básicas que são úteis para a clusterização
    de textos utilizando uma abordagem clássica."""

    def __init__(self):
        '''Este é o construtor da classe.
           Aqui você deve adicionar/remover atributos para se ajustar ao seu problema

           Variáveis de entrada:
           None

           Variáveis de saída:
           None
        '''
        self.textos = [] #atributo que contém os textos sem processamento algum
        self.textos_tratados = [] #atributo que contém os textos já processados
        self.textos_stem = [] #atributo que contém os textos tratados após passar por um stemmer
        self.textos_id = [] #atributo que contém um identificador único para cada texto
        self.stop_words = [] #atributo que contém todas as stop words que serão levadas em consideração no seu problema

    def define_stop_words(self):
        '''
        Inicializa o atributo "stop_words" da classe pegando stopwords de
        diferentes bibliotecas e as tratando para ficarem no formato correto.

        Você pode (deve) adicionar/remover stop words para se ajustar ao seu problema.

        Variáveis de entrada:
        None

        Variáveis de saída:
        None
        '''

        self.stop_words = get_stop_words('portuguese')
        self.stop_words = self.stop_words + nltk.corpus.stopwords.words('portuguese')
        self.stop_words = self.stop_words + ['art','dou','secao','pag','pagina', 'in', 'inc', 'obs', 'sob', 'ltda','ia']
        self.stop_words = self.stop_words + ['ndash', 'mdash', 'lsquo','rsquo','ldquo','rdquo','bull','hellip','prime','lsaquo','rsaquo','frasl', 'ordm']
        self.stop_words = self.stop_words + ['prezado', 'prezados', 'prezada', 'prezadas', 'gereg', 'ggali','usuario', 'usuaria', 'deseja','gostaria', 'boa tarde', 'bom dia', 'boa noite']
        self.stop_words = self.stop_words + ['rdc','resolucao','portaria','lei','janeiro','fevereiro','marco','abril','maio','junho','julho','agosto','setembro','outubro','novembro','dezembro']
        self.stop_words = self.stop_words + ['decreto','anvisa','anvs','diretoria','colegiada','capitulo','item','regulamento','tecnico','nr','instrucao','normativa','anexo']
        self.stop_words = self.stop_words + ['paragrafo', 'unico','devem','caso','boas','vigilancia','sanitaria','cada']
        self.stop_words = list(dict.fromkeys(self.stop_words))
        self.stop_words = ' '.join(self.stop_words)

        #As stop_words vem com acentos/cedilhas. Aqui eu tiro os caracteres indesejados
        self.stop_words = self.limpa_utf8(self.stop_words)

    def limpa_utf8(self, texto):
        '''
        Recodifica em utf-8. Remove cedilhas, acentos e coisas de latin.

        Variáveis de entrada:
        texto: é uma variável do tipo string que contém um texto. Não pode ser uma palavra única.

        Variáveis de saída:
        texto_tratado: é uma variável do tipo string que contém um texto recodificado em utf-8.
        '''

        texto = texto.split()
        texto_tratado = []
        for palavra in texto:
            # Unicode normalize transforma um caracter em seu equivalente em latin.
            nfkd = unicodedata.normalize('NFKD', palavra)
            palavra_sem_acento = u"".join([c for c in nfkd if not unicodedata.combining(c)])
            texto_tratado.append(palavra_sem_acento)

        return ' '.join(texto_tratado)

    def trata_textos(self, texto):
        '''
        Trata os textos. Remove stopwords, sites, pontuacao, caracteres especiais etc.
        Você pode (deve) alterar esse método para se ajustar da melhor forma possível ao seu problema

        Variáveis de entrada:
        texto: é uma string que contém o texto a ser tratado.

        Variáveis de saída:
        texto_limpo: é uma string que contém o texto já tratado.
        '''

        #converte todos caracteres para letra minúscula
        texto_lower = texto.lower()
        texto_lower = re.sub(r'\xa0',' ',texto_lower)

        #tira sites
        texto_sem_sites =  re.sub('(http|www)[^ ]+','',texto_lower)

        #Remove acentos e pontuação
        texto_sem_acento_pontuacao = self.limpa_utf8(texto_sem_sites)

        #Remove hifens e barras
        texto_sem_hifens_e_barras = re.sub('[-\/]', ' ', texto_sem_acento_pontuacao)

        #Troca qualquer tipo de espacamento por espaço
        texto_sem_espacamentos = re.sub(r'\s', ' ', texto_sem_hifens_e_barras)

        #Remove pontuacao e digitos
        texto_limpo = re.sub('[^A-Za-z]', ' ' , texto_sem_espacamentos)

        #Retira numeros romanos e stopwords
        texto_limpo = texto_limpo.split()
        texto_sem_stopwords = [self.tira_stopwords_e_romanos(palavra) for palavra in texto_limpo]
        texto_sem_stopwords = ' '.join(texto_sem_stopwords)

        #Remove pontuacao e digitos
        texto_limpo = re.sub('[^A-Za-z]', ' ' , texto_sem_stopwords)

        #Remove espaços extras
        texto_limpo = re.sub(' +', ' ', texto_limpo)

        return texto_limpo

    def tira_stopwords_e_romanos(self, palavra, values={'m': 1000, 'd': 500, 'c': 100, 'l': 50,
                                    'x': 10, 'v': 5, 'i': 1}):
        '''
        Retira stop words e números romanos.

        Variáveis de entrada:
        palavra: é uma string que contém uma palavra. Não funciona se colocar um texto.

        Variáveis de saída:
        out: é uma string vazia se a entrada era um número romano ou stop word. Se a entrada
        não era romano ou stop word, retorna a própria entrada.
        '''

        #recodifica em utf-8
        palavra = self.limpa_utf8(palavra)

        #remove stopwords
        if palavra in self.stop_words:
            return ''

        #se a palavra tem tamanho menor que 2 provavelmente não é importante
        if(len(palavra) < 2 ):
            return ''

        if (palavra == ''): return ''

        #se a palavra não é um número romano eu retorno a própia palavra.
        out = re.sub('[^mdclxvi]', '', palavra)
        if (len(out) != len(palavra)):
            return palavra

        #se chegou aqui é porque é um número romano
        return ''

    def importa_textos(self):
        '''Aqui você deve definir sua função de importar textos.
           Nesse método você deve inicializar os atributos 'textos' e 'textos_tratados'.

           Você pode/deve adicionar variáveis de entrada para esse método para se ajustar
           ao seu problema.
        '''

    def stem(self):
        '''
        Faz o stemming nas palavras utilizando o pacote nltk com o RSLP Portuguese stemmer.
        O resultado do stemming fica salvo no atributo textos_stem.

        Variáveis de entrada:
        None

        Variáveis de saída:
        None
        '''

        print('Comecou a fazer o stemming.')
        t = time.time()

        #inicializando o objeto stemmer
        stemmer = nltk.stem.RSLPStemmer()

        for texto in self.textos_tratados:
            #Faz o stemming para cada palavra na resolucao
            palavras_stemmed_texto = [stemmer.stem(word) for word in texto.split()]
            #Faz o append da resolucao que passou pelo stemming
            self.textos_stem.append(" ".join(palavras_stemmed_texto))

        print('Tempo para fazer o stemming: ' + str(time.time() - t) + '\n')


    def analisa_clusters(self, base_tfidf, id_clusters):
        '''
        Tenta visualizar as cluster definidas. Além disso retorna um dataframe
        que contém a informação do número de textos por cluster

        Variáveis de entrada:
        base_tfidf: é uma matriz obtida pela aplicação do tfidf no bag of words obtido a partir
        dos textos que passaram pelo stemmer. Não é a matriz com dimensionalidade reduzida.
        id_cluster: é um array que contém os identificadores de cada cluster.

        Variáveis de saída:
        cluster_n_textos: é um dataframe em que a primeira coluna está o identificador da cluster
        e na segunda coluna está o número de textos na cluster.
        '''

        clusters = np.unique(id_clusters)

        #inicializa o output da funcao
        n_textos = np.zeros(len(clusters)) #numero de textos pertencentes a uma cluster

        #reduz a dimensionalidade para 2 dimensoes
        base_tfidf_reduced = self.SVD(2,base_tfidf)
        X = base_tfidf_reduced[:,0]
        Y = base_tfidf_reduced[:,1]

        colors = cm.rainbow(np.linspace(0, 1, len(n_textos)))

        for cluster, color in zip(clusters, colors):
            idxs = np.where(id_clusters == cluster) #a primeira cluster não é a 0 e sim a 1
            n_textos[cluster-1] = len(idxs[0])
            x = X[idxs[0]]
            y = Y[idxs[0]]
            plt.scatter(x, y, color=color)

        n_textos = pd.DataFrame(n_textos, columns=['numero de textos'])
        cluster_n_textos = pd.DataFrame(clusters,columns=['cluster_id']).join(n_textos)

        return cluster_n_textos

    def SVD(self,dim,base_tfidf):
        '''
        Reduz a dimensionalidade dos dados de entrada.

        Variáveis de entrada:
        dim: é um inteiro que corresponde ao número de dimensões desejada na saída.
        base_tfidf: base de dados a ter sua dimensionalidade reduzida.

        Variáveis de saída:
        base_tfidf_reduced: base de dados com dimensionalidade reduzida.
        '''

        print('Começou a redução de dimensionalidade.')
        t = time.time()
        svd = TruncatedSVD(n_components = dim, random_state = 42)
        base_tfidf_reduced = svd.fit_transform(base_tfidf)
        print('Número de dimensões de entrada: ' + str(base_tfidf.shape[1]))
        print(str(dim) + ' dimensões explicam ' + str(svd.explained_variance_ratio_.sum()) + ' da variância.')
        elpsd = time.time() - t
        print('Tempo para fazer a redução de dimensionalidade: ' + str(elpsd) + '\n')
        return base_tfidf_reduced

    def generate_wordcloud(self, cluster_id, filename):
        '''
        Gera uma nuvem de palavras de uma cluster com identificador 'cluster_id'.

        Variáveis de entrada:
        cluster_id: é um inteiro que contém o identificador da cluster que se deseja fazer uma word cloud.
        filename: é o nome do arquivo csv que contém o identificador da cluster e o texto.

        Variáveis de saída:
        None
        '''

        df = pd.read_csv(filename,sep='|')
        a = df[df['cluster_id'] == cluster]

        L=[]
        for i in range(a.shape[0]):
            L.append(self.textos_tratados[a.iloc[i,2]])

        text = '\n'.join(L)

        wordcloud = WordCloud(stopwords=self.stop_words.split()).generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def generate_csvs(self):
        '''
        Crie aqui o método que cria csvs que serão utilizados para a análise dos resultados.
        Você pode (deve) adicionar variáveis de entrada para se ajustar ao seu problema.

        Sugere-se que sejam criados dois csvs:
        1. info_cluster.csv: a primeira coluna contém o identificador da cluster ('cluster_id') e a segunda coluna contém
        o número de textos na cluster ('numero de textos').
        2. textos_por_cluster.csv: a primeira coluna contém o texto (ou uma referência para o texto) e a segunda coluna contém
        o identificador de qual cluster esse texto pertence.
        '''
