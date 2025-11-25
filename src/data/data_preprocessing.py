import numpy as np
import pandas as pd

def create_dt_dataset(df_train):
    """
    Convierte el dataset raw del TP al formato requerido por el Decision Transformer.

    Args:
        df_train (pd.DataFrame):
            DataFrame con columnas:
                - 'items': array con IDs de películas/libros vistos por un usuario
                - 'ratings': array con ratings correspondientes
                - 'user_group': cluster del usuario (0–7)

    Returns:
        trajectories (list):
            Lista de trayectorias, una por usuario.
            Cada trayectoria es un diccionario con:
                - 'items': numpy array de item IDs
                - 'ratings': numpy array de ratings
                - 'returns_to_go': suma de rewards futuros desde cada punto
                - 'timesteps': array [0, 1, ..., T-1]
                - 'user_group': entero (cluster del usuario)
                
    Ejemplo:
        [
            {
                'items': array([...]),
                'ratings': array([...]),
                'returns_to_go': array([...]),
                'timesteps': array([0,1,2,...]),
                'user_group': 3
            },
            ...
        ]
    """

    trajectories = []

    # Iterar sobre cada usuario
    for _, row in df_train.iterrows():
        items = row["items"]          # numpy array
        ratings = row["ratings"]      # numpy array
        group = row["user_group"]     # int

        # === Calcular returns-to-go ===
        # R_t = sum_{i=t}^{T} ratings[i]
        returns = np.zeros(len(ratings))

        # último timestep
        returns[-1] = ratings[-1]

        # acumular hacia atrás
        for t in range(len(ratings) - 2, -1, -1):
            returns[t] = ratings[t] + returns[t + 1]

        # Construir diccionario de trayectoria
        trajectory = {
            "items": items,
            "ratings": ratings,
            "returns_to_go": returns,
            "timesteps": np.arange(len(items)),
            "user_group": group,
        }

        trajectories.append(trajectory)

    return trajectories


# Ejemplo de uso (comentado para evitar ejecución automática)
# if __name__ == "__main__":
#     df_train = pd.read_pickle("../data/train/netflix8_train.df")
#     trajs = create_dt_dataset(df_train)
#     print("Total trayectorias:", len(trajs))
#     print("Keys trayectoria 0:", trajs[0].keys())

