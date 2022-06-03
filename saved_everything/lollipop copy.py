import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from palettable import cartocolors






# np.random.seed(19680801)

# dt = 0.01
# t = np.arange(0, 30, dt)
# nse1 = np.random.randn(len(t))                 # white noise 1
# nse2 = np.random.randn(len(t))                 # white noise 2

# # Two signals with a coherent part at 10Hz and a random part
# s1 = np.sin(2 * np.pi * 10 * t) + nse1
# s2 = np.sin(2 * np.pi * 10 * t) + nse2

# fig, axs = plt.subplots(2, 1)
# axs[0].plot(t, s1, t, s2)
# axs[0].set_xlim(0, 2)
# axs[0].set_xlabel('time')
# axs[0].set_ylabel('s1 and s2')
# axs[0].grid(True)

# cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
# axs[1].set_ylabel('coherence')

# fig.tight_layout()
# plt.show()
















df = pd.read_csv("all_tasks_all_posthoc.csv")
#df = df.pop(df.columns.values[0])
df = df[df['thresholder'].str.contains('topk')]

ALPHA = 0.4
SIZE = 140
xlabel_size = 15
ylabel_size = 13 
domainlabel_size = 25


data = 'Agnews'
df = df[df['Task'].str.contains(str(data))]
df = df[df['Rationales_metrics'].str.contains('AOPC_sufficiency')]
#df = df[df['Rationales_metrics'].str.contains('AOPC_comprehensiveness')]
my_range=df['Domain']
#print(df)



fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1.5, 4, 4]}, sharey='all')




ax[0].hlines(y=my_range, xmin=0.84, xmax=0.95, color='grey', alpha=0.35)
#ax[0].set_xlim(0, 2)
ax[0].scatter(df['mean-f1'], df['Domain'], color='dimgray', alpha=1, label='F1', marker='$F1$', s=166) # marker='|'
#ax[0].xlabel('BERT F1', fontsize=xlabel_size)
#ax[0].ylabel(str(data), fontsize=ylabel_size)
ax[0].set_xlabel('BERT avg macro-F1',fontsize=xlabel_size)

ax[0].set_ylabel(str(data), fontsize=xlabel_size)
#ax[0].tick_params(axis='both', which='major', labelsize=domainlabel_size)
# plt.setp(ax.get_xticklabels(), fontsize=33)
plt.plot()


############################################################ 2 
#plt.subplot(132)
# The horizontal plot is made using the hline function
ax[1].hlines(y=my_range, xmin=0.2, xmax=0.5, color='grey', alpha=0.35)
ax[1].scatter(df['random'], my_range, color='dimgray', alpha=1, label='Random', marker='|', s=344)
ax[1].scatter(df['gradients'], my_range, color='red', alpha=ALPHA , label='Gradients', s=SIZE)
ax[1].scatter(df['deeplift'], my_range, color='green', alpha=ALPHA , label='Deeplift', s=SIZE)
ax[1].scatter(df['ig'], my_range, color='purple', alpha=ALPHA , label='IG', s=SIZE)
ax[1].scatter(df['gradientshap'], my_range, color='saddlebrown', alpha=ALPHA , label='Gradientshap', s=SIZE)
ax[1].scatter(df['scaled attention'], my_range, color='blue', alpha=ALPHA , label='Scaled Attention', s=SIZE)
ax[1].scatter(df['deepliftshap'], my_range, color='orange', alpha=ALPHA , label='Deepliftshap', s=SIZE)
ax[1].scatter(df['attention'], my_range, color='gold', alpha=ALPHA , label='Attention', s=SIZE)
ax[1].scatter(df['lime'], my_range, color='plum', alpha=ALPHA , label='Lime', s=SIZE)
    # # Annotation
    # plt.axhline(0.5, color='green')
    # plt.legend()
# Add title and axis names
# plt.yticks(my_range, df['group'])
#plt.title("AOPC Sufficiency Comparison of Feature Attributes", loc='left')
#ax[1].xlabel('AOPC Sufficiency', fontsize=xlabel_size)

# ax[1] = plt.gca()
# ax[1].axes.yaxis.set_ticklabels([])
ax[1].set_xlabel('AOPC Sufficiency',fontsize=xlabel_size)
plt.plot()


########################################   3 #################################
# plt.subplot(133)

# The horizontal plot is made using the hline function
ax[2].hlines(y=my_range, xmin=0.2, xmax=0.6, color='grey', alpha=0.35)
ax[2].scatter(df['random'], my_range, color='dimgray', alpha=1, label='Random', marker='d', s=222)
ax[2].scatter(df['gradients'], my_range, color='red', alpha=ALPHA , label='Gradients', s=SIZE)
ax[2].scatter(df['deeplift'], my_range, color='green', alpha=ALPHA , label='Deeplift', s=SIZE)
ax[2].scatter(df['ig'], my_range, color='purple', alpha=ALPHA , label='IG', s=SIZE)
ax[2].scatter(df['gradientshap'], my_range, color='saddlebrown', alpha=ALPHA , label='Gradientshap', s=SIZE)
ax[2].scatter(df['scaled attention'], my_range, color='blue', alpha=ALPHA , label='Scaled Attention', s=SIZE)
ax[2].scatter(df['deepliftshap'], my_range, color='orange', alpha=ALPHA , label='Deepliftshap', s=SIZE)
ax[2].scatter(df['attention'], my_range, color='gold', alpha=ALPHA , label='Attention', s=SIZE)
ax[2].scatter(df['lime'], my_range, color='plum', alpha=ALPHA , label='Lime', s=SIZE)
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
ax[2].set_xlabel('AOPC Sufficiency',fontsize=xlabel_size)
#ax[2].legend()
plt.plot()



plt.subplots_adjust(
    #left=0.4,
    # bottom=0.1, 
    right=0.7, 
    # top=0.2, 
    wspace=0.05, 
    # hspace=0.4,
    )

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
# Show the graph
plt.show()