import matplotlib.pyplot as plt


def plot_C_results(df, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

   
    color1, color2 = 'tab:red', 'tab:blue'
    ax1.set_xlabel('C Value')
    ax1.set_ylabel('ELO', color=color1)
  
    line1 = ax1.plot(df['C_Value'], df['ELO'], marker='o',
                     color=color1, label='ELO')

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('Avg Turns', color=color2)
 
    line2 = ax1_twin.plot(df['C_Value'], df['Avg_Turns'], marker='s',
                          color=color2, linestyle='--', label='Avg Turns')
    ax1.set_title(f"{title}: Performance")

  
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

   
    ax2.bar(df['C_Value'].astype(str), df['Avg_Iters'],
            color='tab:green', alpha=0.7)
    ax2.set_xlabel('C Value')
    ax2.set_ylabel('Avg Iterations per Move')
    ax2.set_title(f"{title}: Computational Efficiency")

 
    for i, v in enumerate(df['Avg_Iters']):
        ax2.text(i, v, f"{v:.0f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
