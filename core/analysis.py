import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("logs/mission_log.csv")

plt.figure()
df["reward"].rolling(20).mean().plot()
plt.title("Reward moving average")
plt.grid()
plt.show()

plt.figure()
df["success"].rolling(20).mean().plot()
plt.title("Success rate")
plt.grid()
plt.show()

plt.figure()
df["energy_per_cabbage"].rolling(20).mean().plot()
plt.title("Energy per cabbage")
plt.grid()
plt.show()

plt.figure()
df["overlap_rate"].rolling(20).mean().plot()
plt.title("Coverage overlap rate")
plt.grid()
plt.show()

plt.figure()
df["total_turns"].rolling(20).mean().plot()
plt.title("Turns per mission")
plt.grid()
plt.show()