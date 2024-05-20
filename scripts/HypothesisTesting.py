#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import plotly.express as px
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

def different_means(df, col, cat_col, class0, class1, quantile = True):
    
    '''This function should conduct different mean hypothesis testing which is a method used to compare two distribution means.
    
     Inputs: 'df' the dataframe that carries 
     'col' that will be tested for equal means for the two classes 'class0' and 'class1' in 'cat_col' column
     quantile = True if the it's better to consider the trimmed mean and not the actual mean
     
     Output: Visualization, Summary Statistics, hypothesis testing results
    '''
    
    my_df = df[df[cat_col].isin([class0, class1])][[cat_col, col]]
    
    if quantile == True:
        
        col1 = col +'_' +'10th_quantile'
        col2 = col +'_' +'90th_quantile'

        my_df[col1] = my_df.groupby(cat_col)[col].transform(lambda x: x.quantile(0.25))

        my_df[col2] = my_df.groupby(cat_col)[col].transform(lambda x: x.quantile(0.9))

        my_df = my_df[(my_df[col] > my_df[col1])& (my_df[col] < my_df[col2])]
        
    
    
    ##box plot visualizations
    my_pal = {class0: "red", class1: "green"}
    sns.boxplot(x= col, y= cat_col, data=my_df, palette=my_pal)
    plt.show()
    
    
    ## Let's visualize our data

    fig = px.histogram(my_df, x= col, color= cat_col,
                    barmode="overlay", histnorm = 'probability density', color_discrete_sequence = ['green', 'red'])

    #change color of the plots
    mean1 = my_df[my_df[cat_col] == class0][col].mean()
    mean2 = my_df[my_df[cat_col] == class1][col].mean()
    fig.add_vline(x=mean1, line_width=3, line_dash="dash", line_color="red")
    fig.add_vline(x=mean2, line_width=3, line_dash="dash", line_color="green")
    fig.update_layout(legend=dict(  font=dict(size=20)))
    
    fig.show()
    
    
    ##prepare data for hypothesis testing

    s1 = my_df[my_df[cat_col] == class0][col]
    s2 = my_df[my_df[cat_col] == class1][col]
    
    stat, p = ttest_ind(s1, s2)

    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
         print('Probably the same distribution')
    else:
         print('Probably different distributions')

