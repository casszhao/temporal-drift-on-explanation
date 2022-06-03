import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from palettable import cartocolors



df = pd.read_csv("all_tasks_all_posthoc.csv")
#df = df.pop(df.columns.values[0])
df = df[df['thresholder'].str.contains('topk')]

ALPHA = 0.4
SIZE = 111
xlabel_size = 11
ylabel_size = 11


data = 'Agnews'
df = df[df['Task'].str.contains(str(data))]
df = df[df['Rationales_metrics'].str.contains('AOPC_sufficiency')]
#df = df[df['Rationales_metrics'].str.contains('AOPC_comprehensiveness')]
my_range=df['Domain']
print(df)


plt.subplot(131, gridspec_kw={'width_ratios': [2, 1]})
plt.hlines(y=my_range, xmin=0.84, xmax=0.95, color='grey', alpha=0.35)
plt.scatter(df['mean-f1'], df['Domain'], color='dimgray', alpha=1, label='F1', marker='$F1$', s=166) # marker='|'
plt.xlabel('BERT F1', fontsize=xlabel_size)
plt.ylabel(str(data), fontsize=ylabel_size)
plt.plot()


plt.subplot(132)
# The horizontal plot is made using the hline function
plt.hlines(y=my_range, xmin=0.2, xmax=0.5, color='grey', alpha=0.35)
plt.scatter(df['random'], my_range, color='dimgray', alpha=1, label='Random', marker='|', s=333)
plt.scatter(df['gradients'], my_range, color='red', alpha=ALPHA , label='Gradients', s=SIZE)
plt.scatter(df['deeplift'], my_range, color='green', alpha=ALPHA , label='Deeplift', s=SIZE)
plt.scatter(df['ig'], my_range, color='purple', alpha=ALPHA , label='IG', s=SIZE)
plt.scatter(df['gradientshap'], my_range, color='saddlebrown', alpha=ALPHA , label='Gradientshap', s=SIZE)
plt.scatter(df['scaled attention'], my_range, color='blue', alpha=ALPHA , label='Scaled Attention', s=SIZE)
plt.scatter(df['deepliftshap'], my_range, color='orange', alpha=ALPHA , label='Deepliftshap', s=SIZE)
plt.scatter(df['attention'], my_range, color='gold', alpha=ALPHA , label='Attention', s=SIZE)
plt.scatter(df['lime'], my_range, color='plum', alpha=ALPHA , label='Lime', s=SIZE)
    # # Annotation
    # plt.axhline(0.5, color='green')
    # plt.legend()
# Add title and axis names
# plt.yticks(my_range, df['group'])
#plt.title("AOPC Sufficiency Comparison of Feature Attributes", loc='left')
plt.xlabel('AOPC Sufficiency', fontsize=xlabel_size)
ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])

plt.plot()



plt.subplot(133)

# The horizontal plot is made using the hline function
plt.hlines(y=my_range, xmin=0.2, xmax=0.6, color='grey', alpha=0.35)
plt.scatter(df['random'], my_range, color='dimgray', alpha=1, label='Random', marker='d', s=222)
plt.scatter(df['gradients'], my_range, color='red', alpha=ALPHA , label='Gradients', s=SIZE)
plt.scatter(df['deeplift'], my_range, color='green', alpha=ALPHA , label='Deeplift', s=SIZE)
plt.scatter(df['ig'], my_range, color='purple', alpha=ALPHA , label='IG', s=SIZE)
plt.scatter(df['gradientshap'], my_range, color='saddlebrown', alpha=ALPHA , label='Gradientshap', s=SIZE)
plt.scatter(df['scaled attention'], my_range, color='blue', alpha=ALPHA , label='Scaled Attention', s=SIZE)
plt.scatter(df['deepliftshap'], my_range, color='orange', alpha=ALPHA , label='Deepliftshap', s=SIZE)
plt.scatter(df['attention'], my_range, color='gold', alpha=ALPHA , label='Attention', s=SIZE)
plt.scatter(df['lime'], my_range, color='plum', alpha=ALPHA , label='Lime', s=SIZE)
    # # Annotation
    # plt.axhline(0.5, color='green')
    # plt.legend()
# Add title and axis names
# plt.yticks(my_range, df['group'])
#plt.title("AOPC Sufficiency Comparison of Feature Attributes", loc='left')
plt.xlabel('AOPC Comprehensiveness', fontsize=xlabel_size)
#plt.ylabel('Domain')
# hide y axis for sharing one together
ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])
plt.legend()
plt.plot()



plt.subplots_adjust(
    #left=0.4,
    # bottom=0.1, 
    right=0.7, 
    # top=0.2, 
    wspace=0.01, 
    # hspace=0.4,
    )
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
# Show the graph
plt.show()