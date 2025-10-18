import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_files = {
    'FiLoRA': 'result/eval_score_FiLoRA.csv',
    'LoRA_XS': 'result/eval_score_LoRA_XS.csv',
    'Random': 'result/eval_score_random.csv'
}

data = {}

for label, filename in csv_files.items():
    try:
        df = pd.read_csv(filename)
        data[label] = df
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Please make sure the CSV files are in the same directory as the script.")
        exit()
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        exit()

epochs_to_plot = np.arange(1, 26, 2)

plot_values = {}
for label, df in data.items():
    plot_values[label] = [df.loc[df['epoch'] == e, 'eval_score'].iloc[0] * 100 for e in epochs_to_plot]

fig, ax = plt.subplots(figsize=(10, 5))

bar_width = 0.25
index = np.arange(len(epochs_to_plot))

bars1 = ax.bar(index - bar_width, plot_values['Random'], bar_width, label='Random', color='#528FAD')
bars2 = ax.bar(index, plot_values['FiLoRA'], bar_width, label='FiLoRA', color='#F7AA58')
bars3 = ax.bar(index + bar_width, plot_values['LoRA_XS'], bar_width, label='LoRA_XS', color='#E76254')

# ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('CoLA Mcc (%)', fontsize=12)
# ax.set_title('GSM8K Accuracy over Epochs for Different Methods', fontsize=14)

ax.set_xticks(index)
ax.set_xticklabels(epochs_to_plot)

ax.set_ylim(50, 80)

ax.legend(fontsize=15)

plt.tight_layout()

plt.savefig('CoLA_MCC.svg', format='svg')

plt.show()