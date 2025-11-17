from ast import Tuple
import numpy as np
import random
from scipy.spatial.distance import cdist

# NUM_LISTS = 5
# ELEMENTS_PER_LIST = 10
# a = np.array([
#     [random.randint(0, 3) for _ in range(ELEMENTS_PER_LIST)]
#     for _ in range(NUM_LISTS)
# ])
# print(a)
# entropy_per_gene = []
# for j in range(a.shape[1]):
#     gene_values = a[:, j]
#     unique, counts = np.unique(gene_values, return_counts=True)
#     p = counts / counts.sum()
#     p = p[p > 0]  # حذف احتمال صفر
#     entropy_per_gene.append(-np.sum(p * np.log(p + 1e-12))) 
# print(entropy_per_gene)

a = {
    'a': [],
    'b':[],
    'c':[]
}
print(a.a)