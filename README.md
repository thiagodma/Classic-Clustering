# Classic Clustering

### I. Descrição
Esse pacote contém diversas funcionalidades úteis para fazer uma clusterização de texto utilizando uma abordagem clássica.

### II. Detalhamento dos métodos

#### 1. define_stop_words(): 
Inicializa o atributo "stop_words" da classe pegando stopwords de diferentes bibliotecas e as tratando para ficarem no formato correto. Você pode (deve) adicionar/remover stop words para se ajustar ao seu problema.

Variáveis de entrada:
None

Variáveis de saída:
None

#### 2. limpa_utf8(texto):
Recodifica em utf-8. Remove cedilhas, acentos e coisas de latin.

Variáveis de entrada:
texto: é uma variável do tipo string que contém um texto. Não pode ser uma palavra única.

Variáveis de saída:
texto_tratado: é uma variável do tipo string que contém um texto recodificado em utf-8.

#### 3. trata_textos(texto):
Trata os textos. Remove stopwords, sites, pontuacao, caracteres especiais etc. Você pode (deve) alterar esse método para se ajustar da melhor forma possível ao seu problema.

Variáveis de entrada:
texto: é uma string que contém o texto a ser tratado.

Variáveis de saída:
texto_limpo: é uma string que contém o texto já tratado.

#### 4. tira_stopwords_e_romanos(palavra):
Retira stop words e números romanos.

Variáveis de entrada:
palavra: é uma string que contém uma palavra. Não funciona se colocar um texto.

Variáveis de saída:
out: é uma string vazia se a entrada era um número romano ou stop word. Se a entrada não era romano ou stop word, retorna a própria entrada.

#### 5. importa_textos():
Você deve definir seu método de importar textos. Nesse método você deve inicializar os atributos 'textos' e 'textos_tratados'. Você pode (deve) adicionar variáveis de entrada para esse método para se ajustar ao seu problema.

#### 6. stem():
Faz o stemming nas palavras utilizando o pacote nltk com o RSLP Portuguese stemmer. O resultado do stemming fica salvo no atriuto textos_stem.

Variáveis de entrada:
None

Variáveis de saída:
None

#### 7. analisa_clusters(base_tfidf, id_clusters):
Tenta visualizar as cluster definidas. Além disso retorna um dataframe que contém a informação do número de textos por cluster.

Variáveis de entrada:
base_tfidf: é uma matriz obtida pela aplicação do tfidf no bag of words obtido a partir dos textos que passaram pelo stemmer. Não é a matriz com dimensionalidade reduzida.
id_cluster: é um array que contém os identificadores de cada cluster.

Variáveis de saída:
cluster_n_textos: é um dataframe em que a primeira coluna está o identificador da cluster e na segunda coluna está o número de textos na cluster.

#### 8. SVD(dim, base_tfidf):
Reduz a dimensionalidade dos dados de entrada.

Variáveis de entrada:
dim: é um inteiro que corresponde ao número de dimensões desejada na saída.
base_tfidf: base de dados a ter sua dimensionalidade reduzida.

Variáveis de saída:
base_tfidf_reduced: base de dados com dimensionalidade reduzida.

#### 9. generate_wordcloud(cluster_id, filename):
Gera uma nuvem de palavras de uma cluster com identificador 'cluster_id'.

Variáveis de entrada:
cluster_id: é um inteiro que contém o identificador da cluster que se deseja fazer uma word cloud.
filename: é o nome do arquivo csv que contém o identificador da cluster e o texto.

Variáveis de saída:
None

#### 10. generate_csvs():
Você deve criar método que cria csvs que serão utilizados para a análise dos resultados. Você pode (deve) adicionar variáveis de entrada para se ajustar ao seu problema.

Sugere-se que sejam criados dois csvs:
1. info_cluster.csv: a primeira coluna contém o identificador da cluster ('cluster_id') e a segunda coluna contém o número de textos na cluster ('numero de textos').
2. textos_por_cluster.csv: a primeira coluna contém o texto (ou uma referência para o texto) e a segunda coluna contém o identificador de qual cluster esse texto pertence.
