from classic_clustering import ClassicClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.cluster import hierarchy
import pandas as pd
import time
import matplotlib.pyplot as plt

#Aqui eu crio uma instância da classe ClassicClustering
cc = ClassicClustering()

#Aqui eu defino as stop words gerais e especificas para esse problema. O atributo stop_words foi definido.
cc.define_stop_words()

#Aqui eu carrego os atributos textos e textos_tratados.
cc.importa_textos()

#Faz o stemming e guarda o resultado no atributo textos_stem
cc.stem()

#Vetorizando e aplicando o tfidf
vec = CountVectorizer()
bag_palavras = vec.fit_transform(cc.textos_stem)
feature_names = vec.get_feature_names()
base_tfidf = TfidfTransformer().fit_transform(bag_palavras)
base_tfidf = base_tfidf.todense()

#Reduzindo a dimensionalidade. Você deve definir o número de dimensões da saída.
#Procure um valor que garanta pelo menos 85% da variância explicada
n_dims = 600
base_tfidf_reduced = cc.SVD(n_dims, base_tfidf)


#Clustering
print('Começou a clusterização.')
t = time.time()
clusters_por_cosseno = hierarchy.linkage(base_tfidf_reduced,"average", metric="cosine") #pode testar metric="euclidean" também
plt.figure()
dn = hierarchy.dendrogram(clusters_por_cosseno)

#Você deve alterar o limite de dissimilaridade da forma que melhor se ajuste ao seu dendograma
limite_dissimilaridade = 0.92

id_clusters = hierarchy.fcluster(clusters_por_cosseno, limite_dissimilaridade, criterion="distance")
elpsd = time.time() - t
print('Tempo para fazer a clusterização: ' + str(elpsd) + '\n')

cc.generate_csvs()
