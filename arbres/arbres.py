#%% importation des librairies nécessaires
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from sklearn import tree, datasets, model_selection
from sklearn.model_selection import learning_curve, cross_val_score, train_test_split
from tp_arbres_source import (rand_checkers,plot_2d, frontiere)

# %% question 1
from sklearn import tree

#%% question 2
# génération des échantillons
random.seed(2609)
n = 456
n1 = n//4
n2 = n//4
n3 = n//4
n4 = n//4
sigma = 0.1
data = rand_checkers(n1, n2, n3, n4, sigma)
plot_2d(data)

# construction des classifieurs
dt_entropy = tree.DecisionTreeClassifier(criterion="entropy")
dt_gini = tree.DecisionTreeClassifier(criterion="gini")
X = data[:,:2]
Y = np.array(data[:,2], dtype=int)
dt_entropy.fit(X,Y)
dt_gini.fit(X,Y)
print("Critère d'entropie :", dt_entropy.score(X,Y))
print("Paramètres du Classifieur Associé :", dt_entropy.get_params())
print("Critère de Gini :", dt_gini.score(X,Y))
print("Paramètres du Classifieur Associé :", dt_gini.get_params())

# construction des courbes d'erreurs commises en fonction de la profondeur maximale de l'arbre
d_max = 12
scores_entropy = np.zeros(d_max)
scores_gini = np.zeros(d_max)
for i in range(d_max):
    # critère d'entropie
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i+1)
    dt_entropy.fit(X,Y)
    scores_entropy[i] = dt_entropy.score(X, Y)
    # critère de gini
    dt_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=i+1)
    dt_gini.fit(X,Y)
    scores_gini[i] = dt_gini.score(X,Y)
# affichage du graphique
plt.figure(figsize=(15,5))
plt.plot(range(1, d_max + 1), 1-scores_entropy, label="Critère d'Entropie", color="green")
plt.plot(range(1, d_max + 1), 1-scores_gini, label="Critère de Gini")
plt.legend()
plt.xlabel("Profondeur Maximale")
plt.ylabel("Proportion d'Erreurs")
plt.draw()
plt.title("Graphique de l'Erreur")

# %% question 3
random.seed(2609)

# représentation graphique de la classification obtenue
dt_entropy.max_depth = np.argmin(1-scores_entropy)+1
plt.figure(figsize=(15,5))
frontiere(lambda x: dt_entropy.predict(x.reshape((1, -1))), X, Y, step=100)
plt.title("Meilleures Frontières avec le Critère d'Entropie")
plt.draw()
print("Meilleurs Scores avec le Critère d'Entropie :", dt_entropy.score(X, Y))

# représentation graphique des données simulées
plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(141)
plt.title('Données Simulées')
plot_2d(data[:, :2], data[:, 2], w=None)

#%% question 4
import graphviz
from sklearn.tree import export_graphviz
tree.plot_tree(dt_entropy, filled=True)
data = tree.export_graphviz(dt_entropy, filled=True)
graph = graphviz.Source(data)
graph.render("./graphviz/entropy_tree", format='pdf')

#%% question 5
# génération de l'échantillon test
n = 160
n1 = n//4
n2 = n//4
n3 = n//4
n4 = n//4
sigma = 0.1
data_test = rand_checkers(n1, n2, n3, n4, sigma)
plot_2d(data_test)

X_test = data_test[:,:2]
Y_test = np.asarray(data_test[:,2], dtype=int)

# construction des courbes d'erreurs commises en fonction de la profondeur maximale de l'arbre
d_max = 12
scores_entropy = np.zeros(d_max)
scores_gini = np.zeros(d_max)
for i in range(d_max):
    # critère d'entropie
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i+1)
    dt_entropy.fit(X,Y)
    scores_entropy[i] = dt_entropy.score(X_test, Y_test)
    # critère de gini
    dt_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=i+1)
    dt_gini.fit(X,Y)
    scores_gini[i] = dt_gini.score(X_test,Y_test)

# affichage du graphique
plt.figure(figsize=(15,5))
plt.plot(range(1, d_max + 1), 1-scores_entropy, label="Critère d'Entropie", color="green")
plt.plot(range(1, d_max + 1), 1-scores_gini, label="Critère de Gini")
plt.legend()
plt.xlabel("Profondeur Maximale")
plt.ylabel("Proportion d'Erreurs")
plt.draw()
plt.title("Graphique de l'Erreur de Test")

# calcul de la proportion d'erreurs
error_rate_entropy = 1 - np.average(scores_entropy)
error_rate_gini = 1 - np.average(scores_gini)
print("Proportion Moyenne d'Erreurs avec le Critère d'Entropie :{:.2f}%".format(error_rate_entropy * 100))
print("Proportion Moyenne d'Erreurs avec le Critère de Gini :{:.2f}%".format(error_rate_gini * 100))

#%% question 6
# partitionnement des données
digits = datasets.load_digits()
n_samples = len(digits.data)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(digits.data,
                                                                    digits.target, 
                                                                    test_size=0.2,
                                                                    random_state=50)

# construction des courbes d'erreurs commises en fonction de la profondeur maximale de l'arbre sur les données d'apprentissage
d_max = 12
scores_entropy = np.zeros(d_max)
scores_gini = np.zeros(d_max)
for i in range(d_max):
    # critère d'entropie
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i+1)
    dt_entropy.fit(X_train,Y_train)
    scores_entropy[i] = dt_entropy.score(X_train, Y_train)
    # critère de gini
    dt_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=i+1)
    dt_gini.fit(X_train,Y_train)
    scores_gini[i] = dt_gini.score(X_train,Y_train)
# affichage du graphique
plt.figure(figsize=(15,5))
plt.plot(range(1, d_max + 1), 1-scores_entropy, label="Critère d'Entropie", color="green")
plt.plot(range(1, d_max + 1), 1-scores_gini, label="Critère de Gini")
plt.legend()
plt.xlabel("Profondeur Maximale")
plt.ylabel("Proportion d'Erreurs")
plt.draw()

# construction des courbes d'erreurs commises en fonction de la profondeur maximale de l'arbre sur les données test
d_max = 12
scores_entropy = np.zeros(d_max)
scores_gini = np.zeros(d_max)
for i in range(d_max):
    # critère d'entropie
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i+1)
    dt_entropy.fit(X_train,Y_train)
    scores_entropy[i] = dt_entropy.score(X_test, Y_test)
    # critère de gini
    dt_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=i+1)
    dt_gini.fit(X_train,Y_train)
    scores_gini[i] = dt_gini.score(X_test,Y_test)
# affichage du graphique
plt.figure(figsize=(15,5))
plt.plot(range(1, d_max + 1), 1-scores_entropy, label="Critère d'Entropie", color="green")
plt.plot(range(1, d_max + 1), 1-scores_gini, label="Critère de Gini")
plt.legend()
plt.xlabel("Profondeur Maximale")
plt.ylabel("Proportion d'Erreurs")
plt.draw()

#%% question 7
np.random.seed(2609)
# recherche de la profondeur optimale
d_max = 40
X = digits.data
Y = digits.target
error = np.zeros(d_max)
for i in range(d_max):
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i+1)
    error[i] = np.mean(1 - cross_val_score(dt_entropy, X, Y, cv=5))
# affichage de la courbe d'erreurs
plt.plot(error)
plt.xlabel("Profondeur Maximale")
plt.ylabel("Proportion d'Erreurs")
# affichage de la profondeur optimale
d_optimal = 1 + np.argmin(error)
print("Profondeur Maximale Optimale: ", d_optimal)

#%% question 8
# chargement de l'ensemble de données digits
digits = datasets.load_digits()
X = digits.data
Y = digits.target
# création d'un classificateur d'arbre de décision avec la profondeur maximale optimale
dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=28)
# calcul de la courbe d'apprentissage
n_samples, train_curve, test_curve = learning_curve(dt, X, Y,train_sizes=np.linspace(0.1, 1, 8))
# calculs des moyennes et écarts types des scores
train_scores_mean = np.mean(train_curve, axis=1)
test_scores_mean = np.mean(test_curve, axis=1)
train_scores_std = np.std(train_curve, axis=1)
test_scores_std = np.std(test_curve, axis=1)
# traçage de la courbe d'apprentissage
plt.figure()
plt.grid()
plt.fill_between(n_samples, train_scores_mean -1.96*train_scores_std,
                  train_scores_mean + 1.96*train_scores_std)
plt.fill_between(n_samples, test_scores_mean -1.96*test_scores_std,
                  test_scores_mean + 1.96*test_scores_std, alpha=0.1)
plt.plot(n_samples, train_scores_mean,"o-", label="échantillon d'apprentissage", color="green")
plt.plot(n_samples, test_scores_mean, "o-", label="échantillon test")
plt.legend(loc="lower right")
plt.xlabel("Taille d'Échantillon d'Apprentissage")
plt.ylabel("Précision")
plt.title("Courbes d'Apprentissage pour le Meilleur Arbre de Décision")

# %%
