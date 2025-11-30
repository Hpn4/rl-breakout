import pandas as pd

# Charger le CSV
df = pd.read_csv("model/D3QN4/logs/train.csv")

# Statistiques utiles
print(df["Ep steps"].describe())

print(df["Ep steps"].max())

import matplotlib.pyplot as plt

plt.plot(df["Avg R"])
plt.xlabel("Épisode")
plt.ylabel("Avg reward")
plt.title("Évolution du Ep reward")
plt.show()
