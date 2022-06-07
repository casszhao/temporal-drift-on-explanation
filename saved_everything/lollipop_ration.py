import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

df = pd.read_csv("all_tasks_all_posthoc.csv")
print(df)
df = df[df['thresholder'].str.contains('topk')] # topk

df['gradients'] = df['gradients']/df['random']
df['deeplift'] = df['deeplift']/df['random']
df['attention'] = df['attention']/df['random']
df['lime'] = df['lime']/df['random']
df['gradientshap'] = df['gradientshap']/df['random']
df['deepliftshap'] = df['deepliftshap']/df['random']
df['scaled attention'] = df['scaled attention']/df['random']
df['ig'] = df['ig']/df['random']

df['random'] = df['random']/df['random']
df['mean-f1'] = df['mean-f1']*100

print(df)

data = 'AmazPantry'
df = df[df['Task'].str.contains(str(data))]
print(df)
suff = df[df['Rationales_metrics'].str.contains('AOPC_sufficiency')]
comp = df[df['Rationales_metrics'].str.contains('AOPC_comprehensiveness')]
my_range=suff['Domain']
print(df)

ALPHA = 0.5
SIZE = 150
xlabel_size = 13
ylabel_size = 15
domainlabel_size = 20
legend_font_size = 11

if data == 'Xfact':
    bert_min = 37
    bert_max = 38.5
    suff_min = 0.8
    suff_max = 1.3
    comp_min = 0.8
    comp_max = 1.8
elif data == 'Factcheck':
    bert_min = 70
    bert_max = 75
    suff_min = 0.8
    suff_max = 2.9
    comp_min = 0.2
    comp_max = 4.9
elif data == 'Agnews':
    bert_min = 84
    bert_max = 91
    suff_min = 0.9
    suff_max = 1.6
    comp_min = 0.8
    comp_max = 3.4
elif data == 'Amazdigimu':
    bert_min = 55
    bert_max = 75
    suff_min = 0.3
    suff_max = 3.5
    comp_min = 0.5
    comp_max = 3
    task_name = 'AmazDigiMu'
elif data == 'AmazPantry':
    bert_min = 67
    bert_max = 72
    suff_min = 0.5
    suff_max = 2.5
    comp_min = 0.6
    comp_max = 2.3
elif data == 'Yelp':
    bert_min = 59
    bert_max = 62
    suff_min = 0.7
    suff_max = 2.7
    comp_min = 0.6
    comp_max = 2.8
else:
    bert_min = 0.37
    bert_max = 0.39
    suff_min = 0.3
    suff_max = 0.5
    comp_min = 0.1
    comp_max = 0.35


fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [0.9, 4, 4]}, sharey='all', figsize=(13,2.3))
temporal_order = ["Full size", "SynD", "AsyD1", "AsyD2"]


# Set number of ticks for x-axis
ax[0].set_yticks(range(len(my_range)))
# Set ticks labels for x-axis
ax[0].set_yticklabels(my_range)

ax[0].hlines(y=my_range, xmin=bert_min, xmax=bert_max, color='grey', alpha=0.35)
ax[0].scatter(df['mean-f1'], df['Domain'], color='dimgray', alpha=1, label='F1', marker=">", s=166) # marker='|'
for x,y in zip(df['mean-f1'],df['Domain']):
    label = format(x, '.1f')
    ax[0].annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',
                 fontsize= 13,
                 ) 
ax[0].set_xlabel('BERT avg macro-F1',fontsize=xlabel_size)
if data == 'Amazdigimu':
    ax[0].set_ylabel('AmazDigiMu', fontsize=xlabel_size)
else:
    ax[0].set_ylabel(str(data), fontsize=xlabel_size)
#ax[0].set_yticklabels(my_range, fontsize=13)

plt.plot()


############################################################   SUFF  
ax[1].hlines(y=my_range, xmin=suff_min, xmax=suff_max, color='grey', alpha=0.35)
ax[1].vlines(x = 1, ymin=0, ymax=3.5, color='black', alpha=1)
ax[1].scatter(suff['random'], my_range, color='black', alpha=1, label='Random', marker='|', s=230)

ax[1].scatter(suff['gradients'], my_range, color='red', alpha=ALPHA , label='Gradients', s=SIZE)
ax[1].scatter(suff['deeplift'], my_range, color='green', alpha=ALPHA , label='Deeplift', s=SIZE)
ax[1].scatter(suff['ig'], my_range, color='purple', alpha=ALPHA , label='IG', s=SIZE)
ax[1].scatter(suff['gradientshap'], my_range, color='saddlebrown', alpha=ALPHA , label='Gradientshap', s=SIZE)
ax[1].scatter(suff['scaled attention'], my_range, color='blue', alpha=ALPHA , label='Scaled Attention', s=SIZE)
ax[1].scatter(suff['deepliftshap'], my_range, color='orange', alpha=ALPHA+0.2, label='Deepliftshap', s=SIZE)
ax[1].scatter(suff['attention'], my_range, color='gold', alpha=ALPHA+0.4, label='Attention', s=SIZE)
ax[1].scatter(suff['lime'], my_range, color='plum', alpha=ALPHA+0.4, label='Lime', s=SIZE)
ax[1].set_xlabel('AOPC Sufficiency',fontsize=xlabel_size)
plt.plot()


########################################   COM  #################################
ax[2].hlines(y=my_range, xmin=comp_min, xmax=comp_max, color='grey', alpha=0.35)
ax[2].vlines(x = 1, ymin=0, ymax=3.5, color='black', alpha=1)
ax[2].scatter(comp['random'], my_range, color='black', alpha=1, label='Random', marker='|', s=230)

ax[2].scatter(comp['gradients'], my_range, color='red', alpha=ALPHA , label='Gradients', s=SIZE)
ax[2].scatter(comp['deeplift'], my_range, color='green', alpha=ALPHA , label='Deeplift', s=SIZE)
ax[2].scatter(comp['ig'], my_range, color='purple', alpha=ALPHA , label='IG', s=SIZE)
ax[2].scatter(comp['gradientshap'], my_range, color='saddlebrown', alpha=ALPHA , label='Gradientshap', s=SIZE)
ax[2].scatter(comp['scaled attention'], my_range, color='blue', alpha=ALPHA , label='Scaled Attention', s=SIZE)
ax[2].scatter(comp['deepliftshap'], my_range, color='orange', alpha=ALPHA+0.2, label='Deepliftshap', s=SIZE)
ax[2].scatter(comp['attention'], my_range, color='gold', alpha= ALPHA+0.4, label='Attention', s=SIZE)
ax[2].scatter(comp['lime'], my_range, color='plum', alpha=ALPHA+0.4, label='Lime', s=SIZE)
ax[2].set_xlabel('AOPC Comprehensiveness',fontsize=xlabel_size)
plt.plot()


plt.subplots_adjust(
    left=0.074,
    bottom=0.229, 
    right=0.843, 
    top=0.902, 
    wspace=0.05, 
    hspace=0.2,
    )



plt.legend(bbox_to_anchor=(1.02, 1.1), loc='upper left', borderaxespad=0, fontsize=legend_font_size)
#fig.update_layout(yaxis5=dict(type='category',categoryorder='category ascending'))
fig1 = plt.gcf()

plt.show()
plt.draw()
fig1.savefig('./plot/'+str(data)+'.png', dpi=250)
#plt.savefig('./plot/'+str(data), format='png') # bbox_inches = 'tight', dpi=350,