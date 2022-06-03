import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes




df = pd.read_csv("all_tasks_all_posthoc.csv")
#df = df.pop(df.columns.values[0])
df = df[df['thresholder'].str.contains('topk')]

ALPHA = 0.4
SIZE = 140
xlabel_size = 13
ylabel_size = 15
domainlabel_size = 20
legend_font_size = 11


data = 'factcheck'
df = df[df['Task'].str.contains(str(data))]
suff = df[df['Rationales_metrics'].str.contains('AOPC_sufficiency')]
comp = df[df['Rationales_metrics'].str.contains('AOPC_comprehensiveness')]
print(comp)
my_range=suff['Domain']




fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1.5, 4, 4]}, sharey='all', figsize=(11.5,4.5))




ax[0].hlines(y=my_range, xmin=0.84, xmax=0.95, color='grey', alpha=0.35)
ax[0].scatter(df['mean-f1'], df['Domain'], color='dimgray', alpha=1, label='F1', marker='$F1$', s=166) # marker='|'
ax[0].set_xlabel('BERT avg macro-F1',fontsize=xlabel_size)
ax[0].set_ylabel(str(data), fontsize=xlabel_size)
ax[0].set_yticklabels(my_range, fontsize=13)
plt.plot()


############################################################   SUFF  
ax[1].hlines(y=my_range, xmin=0.25, xmax=0.5, color='grey', alpha=0.35)
ax[1].scatter(suff['random'], my_range, color='dimgray', alpha=1, label='Random', marker='|', s=344)
ax[1].scatter(suff['gradients'], my_range, color='red', alpha=ALPHA , label='Gradients', s=SIZE)
ax[1].scatter(suff['deeplift'], my_range, color='green', alpha=ALPHA , label='Deeplift', s=SIZE)
ax[1].scatter(suff['ig'], my_range, color='purple', alpha=ALPHA , label='IG', s=SIZE)
ax[1].scatter(suff['gradientshap'], my_range, color='saddlebrown', alpha=ALPHA , label='Gradientshap', s=SIZE)
ax[1].scatter(suff['scaled attention'], my_range, color='blue', alpha=ALPHA , label='Scaled Attention', s=SIZE)
ax[1].scatter(suff['deepliftshap'], my_range, color='orange', alpha=ALPHA , label='Deepliftshap', s=SIZE)
ax[1].scatter(suff['attention'], my_range, color='gold', alpha=ALPHA+0.3, label='Attention', s=SIZE)
ax[1].scatter(suff['lime'], my_range, color='plum', alpha=ALPHA+0.2 , label='Lime', s=SIZE)
ax[1].set_xlabel('AOPC Sufficiency',fontsize=xlabel_size)
plt.plot()


########################################   COM  #################################
# plt.subplot(133)

# The horizontal plot is made using the hline function
ax[2].hlines(y=my_range, xmin=0.05, xmax=0.4, color='grey', alpha=0.35)
ax[2].scatter(comp['random'], my_range, color='dimgray', alpha=1, label='Random', marker='d', s=222)
ax[2].scatter(comp['gradients'], my_range, color='red', alpha=ALPHA , label='Gradients', s=SIZE)
ax[2].scatter(comp['deeplift'], my_range, color='green', alpha=ALPHA , label='Deeplift', s=SIZE)
ax[2].scatter(comp['ig'], my_range, color='purple', alpha=ALPHA , label='IG', s=SIZE)
ax[2].scatter(comp['gradientshap'], my_range, color='saddlebrown', alpha=ALPHA , label='Gradientshap', s=SIZE)
ax[2].scatter(comp['scaled attention'], my_range, color='blue', alpha=ALPHA , label='Scaled Attention', s=SIZE)
ax[2].scatter(comp['deepliftshap'], my_range, color='orange', alpha=ALPHA , label='Deepliftshap', s=SIZE)
ax[2].scatter(comp['attention'], my_range, color='gold', alpha= ALPHA+0.3, label='Attention', s=SIZE)
ax[2].scatter(comp['lime'], my_range, color='plum', alpha=ALPHA+0.2 , label='Lime', s=SIZE)
    # # Annotation
    # plt.axhline(0.5, color='green')
    # plt.legend()
# Add title and axis names
# plt.yticks(my_range, df['group'])
#plt.title("AOPC Sufficiency Comparison of Feature Attributes", loc='left')
#ax[0,2].xlabel('AOPC Comprehensiveness', fontsize=xlabel_size)
#plt.ylabel('Domain')
# hide y axis for sharing one together
# ax[2] = plt.gca()
# ax[2].axes.yaxis.set_ticklabels([])
ax[2].set_xlabel('AOPC Comprehensiveness',fontsize=xlabel_size)
#ax[2].legend()
plt.plot()



plt.subplots_adjust(
    left=0.1,
    #bottom=0.1, 
    right=0.8, 
    top=0.98, 
    wspace=0.05, 
    # hspace=0.4,
    )

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=legend_font_size)

# Show the graph

plt.show()
plt.savefig('./plot/'+str(args.dataset)+'_vio.png', bbox_inches = 'tight', dpi=250, format='png')