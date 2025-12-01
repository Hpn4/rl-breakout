import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
models = ["C51", "C51_2", "C51_3", "D3QN", "D3QN2", "D3QN3", "D3QN4", "D3QNPenalty", "FineTune"]

def plot(model_name):
	df = pd.read_csv(f"model/{model_name}/logs/train.csv")

	# Extract needed columns
	avg_reward = df["Avg R"] * 5
	avg_q = df["Avg Q"]

	# Create figure with two subplots
	plt.figure(figsize=(10, 5))

	# Left: Avg Reward
	plt.subplot(1, 2, 1)
	plt.plot(avg_reward)
	plt.title("Average Reward")
	plt.xlabel("Episodes")
	plt.ylabel("Avg Reward")

	# Right: Avg Q
	plt.subplot(1, 2, 2)
	plt.plot(avg_q)
	plt.title("Average Q")
	plt.xlabel("Episodes")
	plt.ylabel("Avg Q")

	# Save to file
	plt.tight_layout()
	plt.savefig(f"plots/metrics_plot_{model_name}.png")
	plt.close()

#for model in models:
#	plot(model)

plot("D3QNPenalty")