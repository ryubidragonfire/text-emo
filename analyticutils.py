# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:16:12 2016
@author: chyam
Purpose: Utilities for analysing cleaned data.
Note: To use plotly:
        import plotly.tools as tls
        tls.set_credentials_file(username='username', api_key='api-key')
"""

#import plotly.plotly as py
#import plotly.offline as py
#import plotly.graph_objs as go
import matplotlib.pyplot as plt
#import squarify

# def draw_pie(label_count, filename):
#     labels = label_count.index.tolist()
#     values = label_count.tolist()
#     trace=go.Pie(labels=labels,values=values)
#     #py.iplot([trace]) # for notebook
#     py.plot([trace], filename=filename)
#     return
    
def draw_PCA(x, y, filename, showStats=False):
    """ Reduce multi-dimention data to 2 dimension using PCA, and visualise.
        x is data points
        y is labels
    """
    from sklearn.decomposition import PCA
    
    # Project multidimension data into 2 dimension
    pca = PCA(2)
    x_proj = pca.fit_transform(x.toarray())
    n_cat = y.max()+1
    # Plot PCA projection
    plt.scatter(x_proj[:, 0], x_proj[:, 1], c=y, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', n_cat))
    plt.title(filename[7:-4])
    plt.colorbar()
    
    if showStats:
        print(pca.explained_variance_)
        print(pca.components_)
        
    if filename:
        plt.savefig(filename)
        
    plt.close()
    return
    
def classification_metrics(y_test, predicted, pos_label, filename):
    from sklearn import metrics
    import numpy as np

    classification_report = metrics.classification_report(y_test, predicted); #print(classification_report)
    confusion_matrix = metrics.confusion_matrix(y_test, predicted); #print(confusion_matrix)
    accuracy_score = metrics.accuracy_score(y_test, predicted)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label)
    auc = metrics.auc(fpr, tpr)

    fname = filename + '-result.txt'
    with open(fname, 'w') as f:
        
        f.write('\n' + 'Classification Report:' + '\n')
        f.writelines(classification_report)
        f.write('\n' + 'Confusion Matrix:' + '\n')
        f.write(np.array2string(confusion_matrix, separator=', '))
        f.write('\n' + 'Accuracy Score:' + '\n')
        f.write(str(accuracy_score))
        f.write('\n' + 'AUC:' + '\n')
        f.write(str(auc))
    return
    
def classification_metrics2(y_test, y_pred, target_names, filename):
    from sklearn import metrics
    import numpy as np

    fname = filename + '-result.txt'
    with open(fname, 'w') as f:
        
        f.write('\n' + 'Classification Report:' + '\n')
        f.writelines(metrics.classification_report(y_test, y_pred, target_names=target_names))
        f.write('\n' + 'Confusion Matrix:' + '\n')
        f.write(np.array2string(metrics.confusion_matrix(y_test, y_pred), separator=','))
        f.write('\n' + 'Accuracy Score:' + '\n')
        f.write(str(metrics.accuracy_score(y_test, y_pred)))
    return

### This does not work
#==============================================================================
# def draw_treemap():
#     
#     x = 0.
#     y = 0.
#     width = 100.
#     height = 100.
#     
#     values = [500, 433, 78, 25, 25, 7]
#     values.sort(reverse=True)
#     
#     normed = squarify.normalize_sizes(values, width, height)
#     ### The error is the line below: object of type 'map' has no len()
#     rects = squarify.squarify(normed, x, y, width, height)
#     
#     # Choose colors from http://colorbrewer2.org/ under "Export"
#     color_brewer = ['rgb(166,206,227)','rgb(31,120,180)','rgb(178,223,138)',
#                     'rgb(51,160,44)','rgb(251,154,153)','rgb(227,26,28)',
#                     'rgb(253,191,111)','rgb(255,127,0)','rgb(202,178,214)',
#                     'rgb(106,61,154)','rgb(255,255,153)','rgb(177,89,40)']
#     shapes = []
#     annotations = []
#     counter = 0
#     
#     for r in rects:
#         shapes.append( 
#             dict(
#                 type = 'rect', 
#                 x0 = r['x'], 
#                 y0 = r['y'], 
#                 x1 = r['x']+r['dx'], 
#                 y1 = r['y']+r['dy'],
#                 line = dict( width = 2 ),
#                 fillcolor = color_brewer[counter]
#             ) 
#         )
#         annotations.append(
#             dict(
#                 x = r['x']+(r['dx']/2),
#                 y = r['y']+(r['dy']/2),
#                 text = values[counter],
#                 showarrow = False
#             )
#         )
#         counter = counter + 1
#         if counter >= len(color_brewer):
#             counter = 0
#     
#     # For hover text
#     trace0 = go.Scatter(
#         x = [ r['x']+(r['dx']/2) for r in rects ], 
#         y = [ r['y']+(r['dy']/2) for r in rects ],
#         text = [ str(v) for v in values ], 
#         mode = 'text',
#     )
#             
#     layout = dict(
#         height=700, 
#         width=700,
#         xaxis=dict(showgrid=False,zeroline=False),
#         yaxis=dict(showgrid=False,zeroline=False),
#         shapes=shapes,
#         annotations=annotations,
#         hovermode='closest'
#     )
#     
#     # With hovertext
#     figure = dict(data=[trace0], layout=layout)
#     
#     # Without hovertext
#     # figure = dict(data=[Scatter()], layout=layout)
#     
#     py.iplot(figure, filename='squarify-treemap')
#     
#     return
#==============================================================================
 
#==============================================================================

### Replaced by df['Label'].value-counts()    
#==============================================================================
# def sample_count_per_label(df):
#     label_count = []
#     uniqueLabels = df.Label.unique()
#     for uLabel in uniqueLabels:
#         count = df.Label.str.contains(uLabel).sum()
#         label_count.append((uLabel, count))
#     return label_count
#==============================================================================
