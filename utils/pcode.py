import random
import math
import copy
import numpy as np

def isProperSuperset(list_a, list_b):
    set_a = set(list_a)
    set_b = set(list_b)
    return set_a.issubset(set_b) and len(set_a) < len(set_b)

def findDifferentElements(list_a, list_b):
    set_a = set(list_a)
    set_b = set(list_b)
    different_elements = set_a.symmetric_difference(set_b)
    return list(different_elements)

def flatten(inlist:list = None):
    # if not l or len(l) <= 1:
    #     return None
    # n = np.array(l[0]).reshape(1,-1)

    
    # for i in range(1,len(l)):
    #     print(n, np.array(l[i])[:,np.newaxis])
    #     np.concatenate((n, np.array(l[i]).reshape(1,-1)))
    # return n.ravel()
    if not inlist or len(inlist) <= 1:
        return inlist
    
    new_list = []
    for l in inlist:
        for i in range(1,len(l)):
            new_list.append((l[i-1], l[i]))

    return np.array(new_list)
        

def findLargestGaps(arr, k):
    # 计算并存储间隔及其对应的下标
    gaps_with_indices = [(arr[i+1] - arr[i], i) for i in range(len(arr) - 1)]
    
    # 按间隔从小到大进行排序
    gaps_with_indices.sort(reverse=True)
    
    # 获取k个最大间隔的下标
    largest_k_gap_indices = [index for _, index in gaps_with_indices[:k]]
    
    # 返回排序后的下标
    return sorted(largest_k_gap_indices)

        