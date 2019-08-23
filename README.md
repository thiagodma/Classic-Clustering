# Classic Clustering

### I. Descrição
Esse pacote contém diversas funcionalidades úteis para fazer uma clusterização de texto utilizando uma abordagem clássica. O ideal é que você herde a classe e modifique apenas os métodos necessários.

### II. Detalhamento dos métodos

#### 0. __init__():
Obs: Esse é o construtor da classe. Nele, são inicializados diversos atributos, a saber:

  i. textos: atributo que contém os textos sem processamento algum
  ii. textos_tratados: atributo que contém os textos já processados
  iii. textos_stem: atributo que contém os textos tratados após passar por um stemmer
  iv. textos_id: atributo que contém um identificador único para cada texto
  v. stop_words: atributo que contém todas as stop words que serão levadas em consideração no seu problema

#### 1. define_stop_words(): 

Obs: Em geral não é necessário modificar esse método.

Inicializa o atributo "stop_words" da classe pegando stopwords de diferentes bibliotecas e as tratando para ficarem no formato correto. Você pode adicionar stop words para se ajustar ao seu problema.

Variáveis de entrada:
user_defined_stopwords: é uma variável opcional. É uma lista de strings que você deseja adicionar como stopwords.

Variáveis de saída:
None

#### 2. importa_textos():

Você deve definir esse método. Nele, você DEVE inicializar os atributos 'textos' , 'textos_tratados' e 'textos_id'.

#### 3. limpa_utf8(texto:str):

Obs: Em geral não é necessário modificar esse método.

Recodifica em utf-8. Remove cedilhas, acentos e coisas de latin.

Variáveis de entrada:
texto: é uma variável do tipo string que contém um texto. Não pode ser uma palavra única.

Variáveis de saída:
texto_tratado: é uma variável do tipo string que contém um texto recodificado em utf-8.

#### 4. trata_textos(texto:str):

Obs: Em geral é interessante modificar esse método. Para casos simples não é necessário

Trata os textos. Remove stopwords, sites, pontuacao, caracteres especiais etc. Você pode (deve) alterar esse método para se ajustar da melhor forma possível ao seu problema.

Variáveis de entrada:
texto: é uma string que contém o texto a ser tratado.

Variáveis de saída:
texto_limpo: é uma string que contém o texto já tratado.

#### 5. tira_stopwords_e_romanos(palavra:str):

Obs: Em geral não é necessário modificar esse método.

Retira stop words e números romanos.

Variáveis de entrada:
palavra: é uma string que contém uma palavra. Não funciona se colocar um texto.

Variáveis de saída:
out: é uma string vazia se a entrada era um número romano ou stop word. Se a entrada não era romano ou stop word, retorna a própria entrada.

#### 6. stem():

Obs: Em geral não é necessário modificar esse método.

Faz o stemming nas palavras utilizando o pacote nltk com o RSLP Portuguese stemmer. O resultado do stemming fica salvo no atributo textos_stem.

Variáveis de entrada:
None

Variáveis de saída:
None

#### 7. vec_tfidf(stem:bool=True):

Obs: Em geral não é necessário modificar esse método.

Vetoriza e aplica o tfidf nos textos. Por padrão utiliza stemming. Se não quiser stemming é só mudar o parâmetro 'stem' para False.

Variáveis de entrada:
stem: é uma variável opcional. É um bool que diz se vai usar os textos com stemming ou não.

Variáveis de saída:
base_tfidf: matriz esparsa que contém a vetorização e aplicação do tfidf
nos textos

#### 8. SVD(base_tfidf, dim:int = 500):

Obs: Em geral não é necessário modificar esse método.

Reduz a dimensionalidade dos dados de entrada.

Variáveis de entrada:
base_tfidf: base de dados a ter sua dimensionalidade reduzida.
dim: é uma variável opcional. É um inteiro que corresponde ao número de dimensões desejada na saída.

Variáveis de saída:
base_tfidf_reduced: base de dados com dimensionalidade reduzida.

#### 9. analisa_clusters(base_tfidf, id_clusters):

Obs: Em geral não é necessário modificar esse método.

Tenta visualizar as cluster definidas. Além disso retorna um dataframe que contém a informação do número de textos por cluster.

Variáveis de entrada:
base_tfidf: é uma matriz obtida pela aplicação do tfidf no bag of words obtido a partir dos textos que passaram pelo stemmer. Não é a matriz com dimensionalidade reduzida.
id_cluster: é um array que contém os identificadores de cada cluster.

Variáveis de saída:
cluster_n_textos: é um dataframe em que a primeira coluna está o identificador da cluster e na segunda coluna está o número de textos na cluster.

#### 9. generate_csvs():
Você deve criar método que cria csvs que serão utilizados para a análise dos resultados.

É necessário que você crie um csv com as seguintes características:
  1. Uma coluna que contém o identificador do texto (o atributo textos_id);
  2. Uma coluna que contém o códigos das clusters. Essa coluna DEVE se chamar 'cluster_id';
  3. Uma coluna que contém os textos tratados (o atributo textos_tratados)

#### 10. generate_wordcloud(cluster_id, filename:str):

Obs: Se você não modificou o formato dos csvs não é necessário modificar esse método.

Gera uma nuvem de palavras de uma cluster com identificador 'cluster_id'.

Variáveis de entrada:
cluster_id: é um inteiro que contém o identificador da cluster que se deseja fazer uma word cloud.
filename: é o nome do arquivo csv que contém o identificador da cluster e o texto.

Variáveis de saída:
None
