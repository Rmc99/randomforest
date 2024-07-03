# Importar as bibliotecas necessárias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree

# Carregar o conjunto de dados Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Visualizar as primeiras linhas do DataFrame
print("Primeiras 5 linhas do conjunto de dados Iris:")
print(df.head())

# Obter uma descrição estatística dos dados
print("\nDescrição estatística do conjunto de dados Iris:")
print(df.describe())

# Verificar as características estatísticas de cada espécie
print("\nDescrição estatística das características por espécie:")
print(df.groupby('species').describe())

# Tamanhos médios das sépalas e pétalas por espécie
avg_sizes = df.groupby('species').mean()
print("\nTamanhos médios das sépalas e pétalas por espécie:")
print(avg_sizes)

# Dividir os dados em conjuntos de treino e teste
X = df.drop(columns=['target', 'species'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar o classificador de Random Forest com o critério de GINI
rf_classifier = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)

# Treinar o modelo
rf_classifier.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = rf_classifier.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
classification_report_output = classification_report(y_test, y_pred)

print(f'\nAcurácia: {accuracy}')
print('\nRelatório de Classificação:')
print(classification_report_output)

# Visualização dos dados usando pairplot com tamanho reduzido
sns.pairplot(df, hue='species', markers=['o', 's', 'D'], height=2.5, aspect=1)
plt.suptitle("Visualização do conjunto de dados Iris", y=1.02)
plt.show()

# Visualização de uma árvore dentro da floresta
plt.figure(figsize=(20, 10))
plot_tree(rf_classifier.estimators_[0], feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Visualização de uma árvore na floresta")
plt.show()
