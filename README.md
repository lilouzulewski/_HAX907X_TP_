# Apprentissage Statistique - TP

- **auteur:** Lilou Zulewski
- **adresse mail:** lilou.zulewski01@etu.umontpellier.fr

Ce dépôt git vise à contenir l'ensemble des travaux pratiques effectués en apprentissage statistique.

### Compilation des fichiers

L'ensemble des travaux pratiques présentés a été effectué sur le logiciel VSCode.

##### Les fichiers `.qmd` :

Pour générer un fichier `.pdf` à partir de fichiers Quarto (`.qmd`), assurez-vous d'abord d'avoir correctement installé Quarto dans votre environnement. Ensuite, exécutez la commande suivante dans le répertoire approprié :

```python
quarto render arbres.qmd --to pdf
```

##### Les fichiers `.py` :

Afin de pouvoir compiler les fichiers `.py`, il est impératif d'avoir installé les packages répertoriés dans le fichier `requirements.txt` comme suit :

```python
$ pip install -r requirements.txt
```


