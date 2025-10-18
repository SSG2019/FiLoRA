import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_files = {
    'FiLoRA': 'result/train_loss_FiLoRA.csv',
    'LoRA_XS': 'result/train_loss_LoRA_XS.csv',
    'Random': 'result/train_loss_random.csv'
}

data = {}

for label, filename in csv_files.items():
    try:
        df = pd.read_csv(filename)
        data[label] = df
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Please make sure the CSV files are in the correct directory.")
        exit()
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        exit()

fig, ax = plt.subplots(figsize=(10, 5))
axins = ax.inset_axes([0.5, 0.55, 0.45, 0.3])  # x, y, width, height

colors = {
    'FiLoRA': '#F7AA58',
    'LoRA_XS': '#E76254',
    'Random': '#528FAD'
}

average_interval_main = 150
average_interval_inset = 2


def get_averaged_data(df_original, interval):
    if df_original.empty:
        return pd.DataFrame({'global_step': [], 'train_loss': []})

    df = df_original.copy()

    max_step = df['global_step'].max()
    min_step = df['global_step'].min()

    bins_start = min(0, min_step)
    bins = np.arange(bins_start, max_step + interval + 1, interval)

    df['bin'] = pd.cut(df['global_step'], bins=bins, right=True, labels=bins[1:], include_lowest=(min_step <= 0))

    df_filtered = df.dropna(subset=['bin'])

    if df_filtered.empty:
        return pd.DataFrame({'global_step': [], 'train_loss': []})

    averaged_df = df_filtered.groupby('bin', observed=True).agg(
        train_loss=('train_loss', 'mean')
    ).reset_index()

    averaged_df = averaged_df.rename(columns={'bin': 'global_step'})

    averaged_df['global_step'] = averaged_df['global_step'].astype(int)

    if 0 in df_original['global_step'].values:
        if not averaged_df['global_step'].isin([0]).any():
            first_point_loss = df_original[df_original['global_step'] == 0]['train_loss'].iloc[0]
            averaged_df = pd.concat(
                [pd.DataFrame({'global_step': [0], 'train_loss': [first_point_loss]}), averaged_df]).sort_values(
                'global_step').reset_index(drop=True)

    return averaged_df


for label, df in data.items():
    averaged_df = get_averaged_data(df, average_interval_main)
    ax.plot(averaged_df['global_step'],
            averaged_df['train_loss'],
            label=label,  # 主图保留 label
            color=colors[label],
            linewidth=2.5,
            alpha=0.9
            )

for label, df in data.items():
    df_subset = df[df['global_step'] <= 100]
    averaged_df_inset = get_averaged_data(df_subset, average_interval_inset)

    axins.plot(averaged_df_inset['global_step'],
               averaged_df_inset['train_loss'],
               # label=label,
               color=colors[label],
               linewidth=1.5,
               alpha=0.9
               )

# ax.set_xlabel('Global Step', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
# ax.set_title('Training Loss over Global Steps for Different Methods', fontsize=14)
ax.set_ylim(0.10, 0.60)


all_global_steps = pd.concat([df['global_step'] for df in data.values()])
ax.set_xlim(0, all_global_steps.max())

formatter_x = plt.FuncFormatter(lambda x, pos: f'{int(x / 1000)}k' if x >= 1000 else f'{int(x)}')
ax.xaxis.set_major_formatter(formatter_x)

ax.legend(fontsize=15)
ax.grid(True, linestyle='--', alpha=0.6)

axins.set_xlabel('The First 100 Steps', fontsize=10)
axins.set_ylabel('Training Loss', fontsize=10)
axins.set_xlim(-5, 105)
axins.set_ylim(0.55, 0.8)
axins.set_xticks([0, 25, 50, 75, 100])
axins.tick_params(axis='x', labelsize=10)
axins.tick_params(axis='y', labelsize=10)
# axins.legend(fontsize=8) # 小窗去除图例
axins.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('CoLA_loss.svg', format='svg')
plt.show()