import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

csvs = sorted(glob.glob('experiments/results/results_*.csv'))
df = pd.read_csv(csvs[-1])

agg = df.groupby(['model', 'governance_level']).agg({
    'css': 'mean',
    'critical_miss': 'mean',
    'hallucination_impactful': 'mean',
    'unsafe_reassurance': 'mean',
    'total_tokens': 'mean',
    'total_latency': 'mean'
}).reset_index()

# Plot CSS vs CCS (Pareto frontier)
g0_tokens = agg[agg['governance_level'] == 'G0']['total_tokens'].mean()
g0_latency = agg[agg['governance_level'] == 'G0']['total_latency'].mean()

agg['ccs'] = 0.6 * (agg['total_tokens'] / g0_tokens) + 0.4 * (agg['total_latency'] / g0_latency)

plt.figure(figsize=(10, 6))
mdf = agg.sort_values('ccs')
plt.plot(mdf['ccs'], mdf['css'], 'o-', linewidth=2, markersize=8)
for _, row in mdf.iterrows():
    plt.annotate(row['governance_level'], (row['ccs'], row['css']),
                textcoords='offset points', xytext=(5, 5))

plt.xlabel('Composite Cost Score (CCS)')
plt.ylabel('Clinical Safety Score (CSS)')
plt.title('GovBench-Med: Governance-Cost Pareto Frontier')
plt.grid(True, alpha=0.3)
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/pareto_frontier.png', dpi=300)
print('Saved to paper/figures/pareto_frontier.png')