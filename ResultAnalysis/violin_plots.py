#!/home/gylai/anaconda3/envs/tf_cpu/bin/python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import rcParams
from matplotlib import font_manager
from copy import deepcopy as kopy
import sys,random
import os

if __name__ == "__main__":

  #---------violin plot------------
  file_path = './violin.xlsx'
  tips = pd.read_excel(file_path)
  sns.set_theme(style="whitegrid",font='FreeSerif')
  # sns.set_style("ticks")
  plt.close('all')
  plt.figure(figsize=(5, 8))
  plt.ylim(0.9,3.1)
  ax=plt.gca()
  y_major_locator=MultipleLocator(1)
  ax.yaxis.set_major_locator(y_major_locator)
  sns.violinplot(x='Algorithm', y='Scott-Knott test ranks', data=tips,\
    order=['I-NSGA-II/LTR','I-MOEA/D/LTR','I-R2-IBEA/LTR'],\
    inner="box",color="silver",cut=0,linewidth=3)

  
  # plt.xlabel()
  plt.ylabel('Scott-Knott test ranks', fontsize=20)
  plt.yticks(fontsize=20)
  # plt.xticks(fontsize=30, rotation=20)
  # ax.axes.get_xaxis().set_visible(False)
  plt.xticks(fontsize=20, rotation=15)
  
  
  plt.savefig("./violin_ltr.pdf")
  import tikzplotlib

  tikzplotlib.save("./violin_ltr.tex")