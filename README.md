# Projet 5 - Processus gaussiens pour l'optimisation bayésienne

Ce dépôt contient le script Python utilisé pour l'expérience du Projet 5 : optimisation bayésienne avec processus gaussiens sur la fonction de Rosenbrock en dimension 10.

## 1. Prérequis

Avant de lancer le script, il faut avoir :

- Python installé sur la machine ;
- `pip` pour installer les bibliothèques Python ;
- un terminal : PowerShell sous Windows, Terminal sous macOS/Linux ;
- les bibliothèques Python suivantes :
  - `numpy` ;
  - `matplotlib`.

Le script n'utilise pas `scikit-learn`, `PyTorch`, `GPyTorch` ou `SciPy`.

## 2. Structure du projet

```text
projet5_gp/
└── gaussian_process_bayesian_optimization.py
README.md
```

Le script principal est :

```text
projet5_gp/gaussian_process_bayesian_optimization.py
```

## 3. Créer un environnement virtuel

### Sous Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### Sous macOS/Linux

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
```

## 4. Installer les dépendances

### Sous Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m pip install numpy matplotlib
```

### Sous macOS/Linux

```bash
./.venv/bin/python -m pip install numpy matplotlib
```

## 5. Vérifier l'installation

### Sous Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -c "import numpy; import matplotlib; print('Installation OK')"
```

### Sous macOS/Linux

```bash
./.venv/bin/python -c "import numpy; import matplotlib; print('Installation OK')"
```

## 6. Lancer un test rapide

Cette commande permet de vérifier rapidement que le script fonctionne.

### Sous Windows PowerShell

```powershell
.\.venv\Scripts\python.exe .\projet5_gp\gaussian_process_bayesian_optimization.py --n_iter 50 --runs 2 --candidates 50 --output_dir results_test
```

### Sous macOS/Linux

```bash
./.venv/bin/python ./projet5_gp/gaussian_process_bayesian_optimization.py --n_iter 50 --runs 2 --candidates 50 --output_dir results_test
```

## 7. Lancer l'expérience finale

Cette commande lance l'expérience principale avec 200 évaluations.

### Sous Windows PowerShell

```powershell
.\.venv\Scripts\python.exe .\projet5_gp\gaussian_process_bayesian_optimization.py --n_iter 200 --runs 3 --candidates 50 --output_dir results
```

### Sous macOS/Linux

```bash
./.venv/bin/python ./projet5_gp/gaussian_process_bayesian_optimization.py --n_iter 200 --runs 3 --candidates 50 --output_dir results
```

## 8. Reproduire les résultats du rapport

Pour reproduire les résultats du rapport avec des hyperparamètres fixes, utiliser :

### Sous Windows PowerShell

```powershell
.\.venv\Scripts\python.exe .\projet5_gp\gaussian_process_bayesian_optimization.py --n_iter 200 --runs 3 --candidates 50 --output_dir results_report --no_tune
```

### Sous macOS/Linux

```bash
./.venv/bin/python ./projet5_gp/gaussian_process_bayesian_optimization.py --n_iter 200 --runs 3 --candidates 50 --output_dir results_report --no_tune
```

## 9. Fichiers générés

Après exécution, le dossier indiqué par `--output_dir` contient :

```text
results/
├── convergence_comparison.png
├── posterior_mean_variance_slice.png
├── final_values.csv
└── summary.txt
```

- `convergence_comparison.png` : comparaison GP-EI vs Random Search ;
- `posterior_mean_variance_slice.png` : moyenne et incertitude a posteriori ;
- `final_values.csv` : résultats numériques par répétition ;
- `summary.txt` : résumé statistique de l'expérience.

## 10. Signification des paramètres

| Paramètre | Description |
|---|---|
| `--n_iter` | Nombre total d'évaluations de la fonction boîte noire. |
| `--runs` | Nombre de répétitions indépendantes. |
| `--candidates` | Nombre de points candidats testés à chaque itération. |
| `--output_dir` | Dossier de sauvegarde des résultats. |
| `--no_tune` | Désactive l'optimisation L-BFGS des hyperparamètres. |

## 11. Problèmes fréquents

### Erreur NumPy ou Matplotlib

Si une erreur apparaît lors de l'import de NumPy ou Matplotlib, réinstaller les dépendances dans l'environnement virtuel :

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall numpy matplotlib
```

Sous macOS/Linux :

```bash
./.venv/bin/python -m pip install --upgrade --force-reinstall numpy matplotlib
```

### Python non reconnu

Vérifier que Python est bien installé :

```bash
python --version
```

ou :

```bash
python3 --version
```
## 12. Déclaration d'assistance IA
Une assistance par IA a été utilisée  pour la relecture, l’aide au débogage du code et l’amélioration de la clarté des explications.
