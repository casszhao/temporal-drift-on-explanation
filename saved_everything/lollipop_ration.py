import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

df = pd.read_csv("all_tasks_all_posthoc.csv")
print(df)
df = df[df['thresholder'].str.contains('contigi')] # topk

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



# for fname in os.listdir('../posthoc_results/factcheck/'):
#     print(fname)
#     if 'ood1' in str(fname):
#         ood1 = pd.read_json('../posthoc_results/factcheck/'+fname)
#         print(ood1)
#     elif 'ood2' in str(fname):
#         ood2 = pd.read_json('../posthoc_results/factcheck/'+fname)
#     else:
#         indomain = pd.read_json('../posthoc_results/factcheck/'+fname)

# exit()



data = 'AmazPantry'


df = df[df['Task'].str.contains(str(data))]

suff = df[df['Rationales_metrics'].str.contains('AOPC_sufficiency')]
comp = df[df['Rationales_metrics'].str.contains('AOPC_comprehensiveness')]
my_range=suff['Domain']
print(my_range)

ALPHA = 0.5
SIZE = 111
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
elif data == 'AmazDigiMu':
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


fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [0.9, 4, 4]}, sharey='all', figsize=(12,2.1))


# Set number of ticks for x-axis
ax[0].set_yticks(range(len(my_range)))
# Set ticks labels for x-axis
ax[0].set_yticklabels(my_range)

ax[0].hlines(y=my_range, xmin=bert_min, xmax=bert_max, color='grey', alpha=0.35)
ax[0].scatter(df['mean-f1'], df['Domain'], color='dimgray', alpha=0.66, label='F1', marker="o", s=89) # marker='|'
for x,y in zip(df['mean-f1'],df['Domain']):
    label = format(x, '.1f')
    ax[0].annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',
                 fontsize= 12,
                 ) 
ax[0].set_xlabel('BERT avg macro-F1',fontsize=xlabel_size)
if data == 'Amazdigimu':
    ax[0].set_ylabel('AmazDigiMu', fontsize=xlabel_size)
else:
    ax[0].set_ylabel(str(data), fontsize=xlabel_size)
ax[0].invert_yaxis()
#ax[0].set_yticklabels(my_range, fontsize=13)

plt.plot()

YMIN = -1
YMAX = 4
############################################################   SUFF  
ax[1].hlines(y=my_range, xmin=suff_min, xmax=suff_max, color='grey', alpha=0.35)
ax[1].vlines(x = 1, ymin=YMIN, ymax=YMAX, color='black', alpha=1)
ax[1].scatter(suff['random'], my_range, color='black', alpha=1, label='Random', marker='|', s=230)
ax[1].scatter(suff['deeplift'], my_range, color='green', alpha=ALPHA , label='DL', s=SIZE, marker='p')
ax[1].scatter(suff['deepliftshap'], my_range, color='orange', alpha=ALPHA+0.2, label='Dsp', s=SIZE, marker='<')
ax[1].scatter(suff['lime'], my_range, color='plum', alpha=ALPHA+0.4, label='LIME', s=SIZE)
ax[1].scatter(suff['attention'], my_range, color='gold', alpha= ALPHA+0.45, label=r'$\alpha$', s=SIZE-21, marker="D")
ax[1].scatter(suff['scaled attention'], my_range, color='blue', alpha=ALPHA , label=r'$\alpha\nabla\alpha$', s=SIZE, marker="s")
ax[1].scatter(suff['gradients'], my_range, color='red', alpha=1, label=r'$x\nabla x $', s=SIZE+33, marker="1")
ax[1].scatter(suff['ig'], my_range, color='purple', alpha=1, label='IG', s=SIZE+33, marker="2")
ax[1].scatter(suff['gradientshap'], my_range, color='saddlebrown', alpha=1, label='Gsp', s=SIZE+33, marker="3")

ax[1].set_xlabel('AOPC Sufficiency',fontsize=xlabel_size)
ax[1].invert_yaxis()
plt.plot()


########################################   COM  #################################
ax[2].hlines(y=my_range, xmin=comp_min, xmax=comp_max, color='grey', alpha=0.35)
ax[2].vlines(x = 1, ymin=YMIN, ymax=YMAX, color='black', alpha=1)


ax[2].scatter(comp['random'], my_range, color='black', alpha=1, label='Random', marker='|', s=230)
ax[2].scatter(comp['deeplift'], my_range, color='green', alpha=ALPHA , label='DL', s=SIZE, marker='p')
ax[2].scatter(comp['deepliftshap'], my_range, color='orange', alpha=ALPHA+0.2, label='Dsp', s=SIZE, marker='<')
ax[2].scatter(comp['lime'], my_range, color='plum', alpha=ALPHA+0.4, label='LIME', s=SIZE)
ax[2].scatter(comp['attention'], my_range, color='gold', alpha= ALPHA+0.45, label=r'$\alpha$', s=SIZE-21, marker="D")
ax[2].scatter(comp['scaled attention'], my_range, color='blue', alpha=ALPHA , label=r'$\alpha\nabla\alpha$', s=SIZE, marker="s")
ax[2].scatter(comp['gradients'], my_range, color='red', alpha=1, label=r'$x\nabla x $', s=SIZE+33, marker="1")
ax[2].scatter(comp['ig'], my_range, color='purple', alpha=1, label='IG', s=SIZE+33, marker="2")
ax[2].scatter(comp['gradientshap'], my_range, color='saddlebrown', alpha=1, label='Gsp', s=SIZE+33, marker="3")

ax[2].set_xlabel('AOPC Comprehensiveness',fontsize=xlabel_size)
ax[2].invert_yaxis()
plt.plot()


plt.subplots_adjust(
    left=0.07,
    bottom=0.221, 
    right=0.893, 
    top=0.914, 
    wspace=0.076, 
    hspace=0.2,
    )



plt.legend(bbox_to_anchor=(1.05, 1.1), loc='upper left', borderaxespad=0, fontsize=legend_font_size-1.5)
#plt.legend(bbox_to_anchor=(-0.3, -0.3), loc='upper center', borderaxespad=0, fontsize=11,fancybox=True,ncol=9)
#fig.update_layout(yaxis5=dict(type='category',categoryorder='category ascending'))
fig1 = plt.gcf()

plt.show()
plt.draw()
fig1.savefig('./plot/'+str(data)+'.png', dpi=660)
#plt.savefig('./plot/'+str(data), format='png') # bbox_inches = 'tight', dpi=350,