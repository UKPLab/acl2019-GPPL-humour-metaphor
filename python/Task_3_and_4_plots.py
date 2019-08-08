
# coding: utf-8

# This script plots the results of tasks 3 and 4 for both humour and metaphor datasets. 
# The images are saved to ./results/ .
# 
# The values are output by run_experiments.py -- please see readme.

import pandas as pd, os
import numpy as np
import matplotlib.pyplot as plt

# Evaluation on the dev set
dir = os.path.expanduser('./results/')

x_f_pairs = np.array([2051, 4084, 8144, 13478, 27052, 41028]) / 1000.0
f_pairs = [0.3841120020065945, 0.45111141697116264, 0.4825356421575895, 0.48987434908316246, 0.5317358598378543, 0.5608195982917]

x_f_inst = np.array([2051, 4102, 8205, 13539, 27078, 41028]) / 1000.0
f_inst = [0.44451344133471576, 0.47897780346932894, 0.48857429640014954, 0.5122297067077531, 0.5181752896870239, 0.5608195982917]

x_m_pairs = np.array([1816, 3677, 7304, 12061, 24057, 36255]) / 1000.0
m_pairs = [0.47562396497912296, 0.5434897447328975, 0.5542133099284116, 0.5356443288309666, 0.540546364726225, 0.5623428742619871]

x_m_inst = np.array([1812, 3625, 7251, 11964, 23928, 36255]) / 1000.0
m_inst = [0.4867460938567052, 0.5308507140321809, 0.525365448157127, 0.5679199137572741, 0.5444105142144827, 0.5623428742619871]

plt.figure(figsize=(3,3))
# label refers to the random subsampling process
plt.plot(x_f_inst, f_inst, marker='o', label='annotation', markersize=8, linestyle='--')
plt.plot(x_f_pairs, f_pairs, marker='x', label='pair', markersize=8, linestyle='-')
plt.ylabel(r'$\rho$')
plt.xlabel('1000 pairwise training labels')
plt.legend(loc='best')
plt.tight_layout()
plt.title('Humor')
plt.savefig(dir + '/task3_funniness.pdf')

plt.figure(figsize=(3.2,3))
# label refers to the random subsampling process
plt.plot(x_m_inst, m_inst, marker='o', label='annotation', markersize=8, linestyle='-')
plt.plot(x_m_pairs, m_pairs, marker='x', label='pair', markersize=8, linestyle='-')
plt.ylabel(r'$\rho$')
plt.xlabel('1000 pairwise training labels')
plt.legend(loc='best')
plt.tight_layout()
plt.title('Metaphor')
plt.savefig(dir + '/task3_metaphor.pdf')

# Evaluation on the training instances
dir = os.path.expanduser('./results/')

x_f_pairs = np.array([2015, 4084, 8144, 13478, 27052]) / 1000.0 #, 41028]) / 1000.0
f_bws_pairs = [0.304256, 0.417959, 0.582394, 0.787522, 0.870539]#, 0.940805]

# with posterior correction
f_pairs = [0.5282654237360423, 0.5276139812779103, 0.6342635450438433, 0.6752237217064453, 0.7336785607447495]#, 0.7527702242887844]

x_f_inst = np.array([2015, 4102, 8205, 13539, 27078]) / 1000.0 #, 41028]) / 1000.0
f_bws_inst = [0.413274, 0.531448, 0.686677, 0.787522, 0.870539]#, 0.940805]

# with the posterior correction
f_inst = [0.45057189570725237, 0.556610034226152, 0.684542910592812, 0.7826548397826213, 0.8829260843398532]#, 0.7527702242887844]

x_m_pairs = np.array([1816, 3677, 7304, 12061, 24057]) / 1000.0 #, 36255]) / 1000.0
m_bws_pairs = [0.221087, 0.278819, 0.385386, 0.489894, 0.667966]#, 0.785294]

# after posterior correction
m_pairs = [0.41860634364509947, 0.5131337514468556, 0.557958638385815, 0.6743223831413175, 0.7193919249155734]#, 0.6276521223325572]

x_m_inst = np.array([1812, 3625, 7251, 11964, 23928]) / 1000.0 #, 36255]) / 1000.0
m_bws_inst = [0.264976, 0.360428, 0.471148, 0.547351, 0.698832]#, 0.785294]

# after posterior correction
m_inst = [0.4764033126062197, 0.5353660283406654, 0.5491569559389652, 0.6908024683751538,  0.6043389123673902]#, 0.6276521223325572]

plt.figure(figsize=(3,4))
# label refers to the random subsampling process
plt.plot(x_f_inst, f_inst, marker='o', label='GPPL, annotation', markersize=8, linestyle='-')
plt.plot(x_f_pairs, f_pairs, marker='x', label='GPPL, pair', markersize=8, linestyle=':')
plt.plot(x_f_inst, f_bws_inst, marker='^', label='BWS, annotation', markersize=8, linestyle='-.')
plt.plot(x_f_pairs, f_bws_pairs, marker='>', label='BWS, pair', markersize=8, linestyle='-')

plt.ylabel(r'$\rho$')
plt.xlabel('1000 pairwise training labels')
plt.legend(loc='best')
plt.tight_layout()
plt.title('Humor')
plt.savefig(dir + '/task4_funniness.pdf')


plt.figure(figsize=(3.2,4))
# label refers to the random subsampling process
plt.plot(x_m_inst, m_inst, marker='o', label='GPPL, annotation', markersize=8, linestyle='-')
plt.plot(x_m_pairs, m_pairs, marker='x', label='GPPL, pair', markersize=8, linestyle=':')
plt.plot(x_m_inst, m_bws_inst, marker='^', label='BWS, annotation', markersize=8, linestyle='-.')
plt.plot(x_m_pairs, m_bws_pairs, marker='>', label='BWS, pair', markersize=8, linestyle='-')

plt.ylabel(r'$\rho$')
plt.xlabel('1000 pairwise training labels')
plt.legend(loc='best')
plt.tight_layout()
plt.title('Metaphor')
plt.savefig(dir + '/task4_metaphor.pdf')