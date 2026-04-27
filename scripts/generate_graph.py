import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure Docs directory exists
os.makedirs('Docs', exist_ok=True)

# Data generation to match the visual curves
x = np.linspace(0, 100, 100)

# Creating curves that plateau to match the provided image
# Gradient Boost: Starts ~60, ends ~91
y_gb = 91 - 31 * np.exp(-x / 35)

# Random Forest: Starts ~62, ends ~91.5
y_rf = 91.5 - 29.5 * np.exp(-x / 35)

# Agrifusion-X: Starts ~63, ends ~93
y_agri = 93 - 30 * np.exp(-x / 35)

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(x, y_gb, label='Gradient Boost', color='#1f77b4', linewidth=2.5)
plt.plot(x, y_rf, label='Random Forest', color='#ff7f0e', linewidth=2.5)
plt.plot(x, y_agri, label='Agrifusion-X (Hybrid Model)', color='#2ca02c', linewidth=2.5)

plt.title('Performance Comparison of Gradient Boost, Random Forest and Agrifusion-X', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Training Progress (%)', fontsize=12, fontweight='bold')
plt.ylabel('Model Accuracy (%)', fontsize=12, fontweight='bold')

plt.xlim(-5, 105)
plt.ylim(58, 95)

plt.grid(True, linestyle='-', alpha=0.7)
plt.legend(loc='lower right', fontsize=11, framealpha=1, edgecolor='black')

# Save the plot
plt.tight_layout()
plt.savefig('Docs/performance_graph.png')
print("Graph successfully generated and saved to Docs/performance_graph.png")
