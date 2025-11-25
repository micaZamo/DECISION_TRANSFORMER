# Aprendizaje por Refuerzo – 📌 Decision Transformer – Recomendación Secuencial (Netflix8)
### Diplomatura en Ciencia de Datos – FAMAF 2025

Este proyecto implementa un sistema de recomendación secuencial basado en Decision Transformer, junto con múltiples baselines, evaluación y experimentos de return-conditioning.

## 📁 Estructura del repositorio 
```  markdown
├── src/
│   ├── models/
│   │   ├── decision_transformer.py
│   │   ├── baselines.py
│   │   └── __init__.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── data_preprocessing.py
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── __init__.py
│   └── evaluation/
│       ├── evaluate.py
│       ├── metrics.py
│       └── __init__.py
│
├── data/
│   ├── processed/
│   │   └── trajectories_train.pkl
│   ├── test_users/
│   │   └── netflix8_test.json
│   ├── train/
│   │   └── netflix8_train.df
│   └── groups/
│       └── mu_netflix8.csv
│
├── notebooks/
│   ├── exploracion_preparacion.ipynb
│   ├── training.ipynb
│   ├── evaluacion.ipynb
│   └── return_conditioning.ipynb
│
├── REPORTE.pdf
├── requirements.txt
└── README.md
```  

## 🚀 Cómo correr el proyecto

1️⃣ Instalar dependencias

```python
pip install -r requirements.txt
```

2️⃣ Generar dataset para Decision Transformer

Se ejecuta en el notebook exploracion_preparacion.ipynb:
```python
from data_preprocessing import create_dt_dataset

trajectories = create_dt_dataset(df_train)
```

Esto guarda:

data/processed/trajectories_train.pkl

3️⃣ Entrenar el modelo

En el notebook training.ipynb:

```python
from src.models.decision_transformer import DecisionTransformer
from src.training.trainer import train_decision_transformer
```

Produce el checkpoint:

results/checkpoints/dt_netflix.pth

4️⃣ Evaluación

En evaluacion.ipynb se comparan:

* Decision Transformer

* Behavior Cloning

* Popularity

* Random

Con métricas:

* HR@K (5, 10, 20)

* NDCG@K

* MRR

5️⃣ Experimentos de Return Conditioning

En return_conditioning.ipynb:

* performance vs target R̂

* análisis por grupos (cold-start)


## 📊 Resultados principales

El Decision Transformer aprende patrones secuenciales pero su performance absoluta es baja debido a la dimensión de 752 clases.

Behavior Cloning obtiene métricas similares.

Popularity y Random sorprendentemente no están tan lejos, lo que sugiere que el dataset es difícil y ruidoso.

El conditioning por return-to-go tiene efecto, pero limitado.

El desempeño varía entre grupos de usuarios.


##  👩‍💻 Autora
Micaela Zamorano

Diplomatura en Ciencia de Datos – FAMAF

2025
