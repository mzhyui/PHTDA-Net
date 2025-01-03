import random
import math
import copy
import numpy as np
import perscode
import matplotlib.pyplot as plt

import gudhi as gd
from ripser import Rips

from .load import getGradients, getTotalLength, initDataset, minimizeProduct
from .load import getSamples, getTopofeature, extractWeights

def mergeSublistsWithSharedItems(data):
    merged = True
    while merged:
        merged = False
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if set(data[i]).intersection(data[j]):
                    data[i] = list(set(data[i]).union(data[j]))
                    del data[j]
                    merged = True
                    break
            if merged:
                break
    return data
    

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


def grouping(data, total_nums, normal_nums, mds_results):
     # Step 1: Create Rips complex and compute persistent homology
    rips_complex = gd.RipsComplex(distance_matrix=data, max_edge_length=100)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()

    # Prepare color map for visualization
    colors = plt.cm.get_cmap('tab10', 10)
    color_list = [colors(i) for i in range(10)]

    # Initialize variables for results
    count = 0
    results = []
    dv_seq = []

    # Step 2: Iterate through persistence pairs
    for (birth_value, death_value), d in [(i[1], i[0]) for i in persistence if i[0] == 0]:
        tolerance = 1e-6
        birth_simplices = []
        death_simplices = []
        filtration = simplex_tree.get_filtration()

        # Identify simplices corresponding to birth and death
        for simplex, filtration_value in filtration:
            if abs(filtration_value - birth_value) < tolerance:
                birth_simplices.append(simplex)
            elif abs(filtration_value - death_value) < tolerance or (death_value == np.Inf and d != 0):
                death_simplices.append(simplex)
                
        # Merge points associated with death simplices
        involved_points_d = mergeSublistsWithSharedItems(death_simplices)

        # Append results if valid
        if len(involved_points_d) >= 0:
            print('d', d, count, birth_value, death_value, involved_points_d)
            results.append([death_value, involved_points_d])
            dv_seq.append(death_value)
        count += 1

    # Step 3: Find largest gaps in death values
    print("the gaps:")
    print(findLargestGaps(dv_seq, 3))

    # Analyze merged sublists of points
    a = []
    for [death_value, concern_points] in results:
        for c in concern_points:
            # print(c)
            a.append(c)
    print(a)

    last_merged = None
    last_distance = None
    score = 0
    has_draw = False

    for i in range(min(len(a)-1, len(results)-1), -1, -1):
        warn_flag = False
        f = flatten(a[i:len(a)])
        if (np.unique(f).shape[0] == total_nums):
            continue
        merged = mergeSublistsWithSharedItems(a[i:len(a)])
        print(f"persistence ={results[i][0]}, finding: {a[i]}\n", merged)
        # print(merged, last_merged)
        score = getTotalLength(merged)/normal_nums/2

        newly_merged = []
        newly_added = []
        merging = []

        # pairwise merges
        if (last_merged != None and len(last_merged)):
            print("merging")
            for idxa, list_a in enumerate(last_merged):
                for idxb, list_b in enumerate(merged):
                    # print(list_a, list_b)
                    if (isProperSuperset(list_a, list_b)):
                        newly_merged.append([idxa, idxb])
                        newly_added.append(findDifferentElements(list_a, list_b))
        
            print(newly_added)
            print(newly_merged)

            if (len(newly_merged) >= 2):
                merging = list(set(newly_merged[0] + newly_merged[1]))
                # merging = find_different_elements(newly_merged[0], newly_merged[1])
                print(merging)
                # print(last_distance)
                if(len(merging) and last_distance.shape[0] > max(merging[-1], merging[-2])):
                    print("merging with pbow_d: ",last_distance[merging[-1]][merging[-2]], np.max(last_distance))
                    if (last_distance[merging[-1]][merging[-2]] >= np.max(last_distance)/2 and score > 0.6 ):
                        print("warn")
                        warn_flag = True
                        a[i] = [0, 0]
                        merged = copy.deepcopy(last_merged)

        
        last_merged = copy.deepcopy(merged)


        
        sub_pers = []
        min_length = float('Inf')
        for sub_points in merged:
            if (len(sub_points) <= 1):
                continue
            min_length = min(min_length, len(sub_points))
            sub_diagram = gd.RipsComplex(distance_matrix=np.array(data)[sub_points][:,sub_points], max_edge_length=100)
            sub_simplex_tree = sub_diagram.create_simplex_tree(max_dimension=2)
            sub_persistence = sub_simplex_tree.persistence()
            sub_pd = np.array([[b, d] for dim, (b, d) in sub_persistence if (dim == 0) and (d != float('inf'))])
            sub_pers.append(sub_pd)

        if (len(sub_pers) > 1 and min_length > 1):
            pbow = perscode.PBoW(N = min(5, min_length), normalize = False)
            pbow_diagrams  = pbow.transform(sub_pers)
            print("pbow:")
            print(pbow_diagrams)
            pbow_distance = np.zeros((len(pbow_diagrams),len(pbow_diagrams)))
            for k in range(len(pbow_diagrams)):
                for j in range(len(pbow_diagrams)):
                    pbow_distance[k][j] = np.linalg.norm(pbow_diagrams[k] - pbow_diagrams[j])
            print(pbow_distance)
        
            if (not warn_flag):
                last_distance = copy.deepcopy(pbow_distance)

        print(f"score x{score}: ",end='')
        total_errors = 0
        for j in merged:
            print([np.mean(np.array(j) >= normal_nums)], end=' ')
            total_errors += min(np.sum((np.array(j) >= normal_nums) == 0),np.sum((np.array(j) >= normal_nums) == 1))
        print(f"\ntotal score : {1- total_errors/data.shape[0]}")
        print()

        if (score >= 0.5 and len(merged) > 1):
            plt.clf()
            plt.scatter(mds_results[:, 0], mds_results[:, 1])
            for idx, points in enumerate(mds_results):
                plt.annotate(idx, (points[0], points[1]), textcoords="offset points", xytext=(0,10), ha='center')

            # print(merged)
            draw_point_set = copy.deepcopy(merged)
            for color, layer in enumerate(merged):
                # print(layer)
                for p in range(len(layer)-1):
                    p1 = mds_results[layer[p]]
                    p2 = mds_results[layer[p + 1]]
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c=color_list[color])
            has_draw = True
            print(draw_point_set)
            plt.show()

    group = np.zeros((total_nums,total_nums))
    for i in mergeSublistsWithSharedItems(a)[0]:
        group[i][i] = 1

    return dv_seq

def simpleGrouping(data, n, dbg=False):
    # Create Rips complex and compute persistence diagram
    rips_complex = gd.RipsComplex(distance_matrix=data, max_edge_length=100)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()

    # Initialize variables for clustering
    count = 0
    results = []
    dv_seq = []
    best_result = None

    tolerance = 1e-6
    for (birth_value, death_value), d in [(i[1], i[0]) for i in persistence if i[0] == 0]:
        # Initialize lists to store simplices contributing to births and deaths
        birth_simplices = []
        death_simplices = []
        filtration = simplex_tree.get_filtration()

        # Identify simplices by filtration values
        for simplex, filtration_value in filtration:
            if abs(filtration_value - birth_value) < tolerance:
                birth_simplices.append(simplex)
            elif abs(filtration_value - death_value) < tolerance or (death_value == np.Inf and d != 0):
                death_simplices.append(simplex)

        # Merge points involved in the death event
        involved_points_d = mergeSublistsWithSharedItems(death_simplices)

        # Store results and death values
        if len(involved_points_d) >= 0:
            results.append([death_value, involved_points_d])
            dv_seq.append(death_value)
        count += 1

    a = []
    for [death_value, concern_points] in results:
        for c in concern_points:
            a.append(c)

    if dbg:
        print(results)

    # Initialize variables for iterative merging
    last_merged = None
    last_distance = None
    last_distance_sum = 0
    score = 0

    for i in range(min(len(a)-1, len(results)-1), -1, -1):
        warn_flag = False
        f = flatten(a[i:len(a)])
        if np.unique(f).shape[0] == len(data):
            continue

        # Merge sublists from current iteration
        merged = mergeSublistsWithSharedItems(a[i:len(a)])

        # Identify newly merged clusters
        newly_merged = []
        newly_added = []
        merging = []
        if last_merged is not None and len(last_merged):
            for idxa, list_a in enumerate(last_merged):
                for idxb, list_b in enumerate(merged):
                    if isProperSuperset(list_a, list_b):
                        newly_merged.append([idxa, idxb])
                        newly_added.append(findDifferentElements(list_a, list_b))

            if len(newly_merged) >= 2:
                merging = list(set(newly_merged[0] + newly_merged[1]))
                if len(merging) and last_distance.shape[0] > max(merging[-1], merging[-2]):
                    if last_distance[merging[-1]][merging[-2]] >= np.max(last_distance) / 2 and len(merging) >= 5:
                        warn_flag = True
                        a[i] = [0, 0]
                        merged = copy.deepcopy(last_merged)

        if dbg:
            print(merged)
        last_merged = copy.deepcopy(merged)

        # Evaluate clusters using persistence bag-of-words (PBoW)
        sub_pers = []
        min_length = float('Inf')
        pbow_distance = None
        for sub_points in merged:
            if len(sub_points) <= 1:
                continue
            min_length = min(min_length, len(sub_points))
            sub_diagram = gd.RipsComplex(distance_matrix=np.array(data)[sub_points][:, sub_points], max_edge_length=100)
            sub_simplex_tree = sub_diagram.create_simplex_tree(max_dimension=2)
            sub_persistence = sub_simplex_tree.persistence()
            sub_pd = np.array([[b, d] for dim, (b, d) in sub_persistence if (dim == 0) and (d != float('inf'))])
            sub_pers.append(sub_pd)

        if len(sub_pers) > 1 and min_length > 1:
            pbow = perscode.PBoW(N=min(5, min_length), normalize=False)
            pbow_diagrams = pbow.transform(sub_pers)
            pbow_distance = np.zeros((len(pbow_diagrams), len(pbow_diagrams)))
            for k in range(len(pbow_diagrams)):
                for j in range(len(pbow_diagrams)):
                    pbow_distance[k][j] = np.linalg.norm(pbow_diagrams[k] - pbow_diagrams[j])

            if not warn_flag:
                last_distance = copy.deepcopy(pbow_distance)

        # Save the best result
        if dbg:
            print(best_result)
        if pbow_distance is not None and len(pbow_distance) == n and np.sum(pbow_distance) >= last_distance_sum:
            best_result = merged
        last_distance_sum = np.sum(pbow_distance) or 0

    # Handle ungrouped points
    assert best_result is not None
    remained_points = [points for points in range(0, len(data)) if points not in [item for row in best_result for item in row]]
    best_result.append(remained_points)

    # Assign labels to data points
    label = np.zeros(len(data))
    for idx, group in enumerate(best_result):
        for number_id in group:
            label[number_id] = idx

    return label.astype(np.int8)


def simpleGroupingVectors(data, n):
    def convert_to_np_array(param):
        # 检查传入的参数是否为list类型
        if isinstance(param, list):
            # 如果是，将其转换为NumPy数组
            return np.array(param)
        else:
            # 如果不是，返回原始参数
            return param
    data = convert_to_np_array(data)
    rips_complex = gd.RipsComplex(points=data, max_edge_length=100)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()

    # colors = plt.cm.get_cmap('tab10', 10)
    # color_list = [colors(i) for i in range(10)]

    
    count = 0
    results = []
    dv_seq = []
    best_result = None


    for (birth_value, death_value), d in [(i[1], i[0]) for i in persistence if i[0] == 0]:
        tolerance = 1e-6
        birth_simplices = []
        death_simplices = []
        filtration = simplex_tree.get_filtration()

        for simplex, filtration_value in filtration:
            # print(filtration_value, birth_value)
            if abs(filtration_value - birth_value) < tolerance:
                # print(simplex)
                birth_simplices.append(simplex)
            elif abs(filtration_value - death_value) < tolerance or (death_value == np.Inf and d != 0):
                death_simplices.append(simplex)
                
        involved_points_d = (mergeSublistsWithSharedItems(death_simplices))

        if len(involved_points_d) >= 0 : 
            # print('d', d, count, birth_value, death_value, involved_points_d) 
            results.append([death_value, involved_points_d])
            dv_seq.append(death_value)
        count += 1
    # print("the gaps:")
    # print(findLargestGaps(dv_seq, 3))
    
    a = []
    for [death_value, concern_points] in results:
        for c in concern_points:
            # print(c)
            a.append(c)
            
    # print(a)
    last_merged = None
    last_distance = None
    last_distance_sum = 0
    score = 0
    # has_draw = False
    for i in range(min(len(a)-1, len(results)-1), -1, -1):
        warn_flag = False
        f = flatten(a[i:len(a)])
        if (np.unique(f).shape[0] == len(data)):
            continue
        merged = mergeSublistsWithSharedItems(a[i:len(a)])
        # print(f"persistence ={results[i][0]}, finding: {a[i]}\n", merged)
        # print(merged, last_merged)

        newly_merged = []
        newly_added = []
        merging = []

        if (last_merged != None and len(last_merged)):
            # print("merging")
            for idxa, list_a in enumerate(last_merged):
                for idxb, list_b in enumerate(merged):
                    # print(list_a, list_b)
                    if (isProperSuperset(list_a, list_b)):
                        newly_merged.append([idxa, idxb])
                        newly_added.append(findDifferentElements(list_a, list_b))
        
            # print(newly_added)
            # print(newly_merged)

            if (len(newly_merged) >= 2):
                merging = list(set(newly_merged[0] + newly_merged[1]))
                # merging = find_different_elements(newly_merged[0], newly_merged[1])
                # print(merging)
                # print(last_distance)
                if(len(merging) and last_distance.shape[0] > max(merging[-1], merging[-2])):
                    # print("merging with pbow_d: ",last_distance[merging[-1]][merging[-2]], np.max(last_distance))
                    if (last_distance[merging[-1]][merging[-2]] >= np.max(last_distance)/2 and len(merging) >= 5):
                        # print("warn")
                        warn_flag = True
                        a[i] = [0, 0]
                        merged = copy.deepcopy(last_merged)

        
        last_merged = copy.deepcopy(merged)


        
        sub_pers = []
        min_length = float('Inf')
        pbow_distance = None
        for sub_points in merged:
            if (len(sub_points) <= 1):
                continue
            min_length = min(min_length, len(sub_points))
            sub_diagram = gd.RipsComplex(points=data[sub_points], max_edge_length=100)
            sub_simplex_tree = sub_diagram.create_simplex_tree(max_dimension=2)
            sub_persistence = sub_simplex_tree.persistence()
            sub_pd = np.array([[b, d] for dim, (b, d) in sub_persistence if (dim == 0) and (d != float('inf'))])
            sub_pers.append(sub_pd)

        if (len(sub_pers) > 1 and min_length > 1):
            pbow = perscode.PBoW(N = min(5, min_length), normalize = False)
            pbow_diagrams  = pbow.transform(sub_pers)
            # print("pbow:")
            # print(pbow_diagrams)
            pbow_distance = np.zeros((len(pbow_diagrams),len(pbow_diagrams)))
            for k in range(len(pbow_diagrams)):
                for j in range(len(pbow_diagrams)):
                    pbow_distance[k][j] = np.linalg.norm(pbow_diagrams[k] - pbow_diagrams[j])
            # print(pbow_distance)
        
            if (not warn_flag):
                last_distance = copy.deepcopy(pbow_distance)

        # print(f"score x{score}: ",end='')
        # total_errors = 0
        # for j in merged:
        #     # print([np.mean(np.array(j) >= normal_nums)], end=' ')
        #     total_errors += min(np.sum((np.array(j) >= normal_nums) == 0),np.sum((np.array(j) >= normal_nums) == 1))
        # print(f"\ntotal score : {1- total_errors/data.shape[0]}")
        # print()

        # save best result
        if pbow_distance is not None and len(pbow_distance) == n and pbow_distance is not None and np.sum(pbow_distance) > last_distance_sum:
            best_result = merged
        last_distance_sum = np.sum(pbow_distance) or 0
        
        

    # if (has_draw):
    #     pass

    # group = np.zeros((total_nums,total_nums))
    # for i in mergeSublistsWithSharedItems(a)[0]:
    #     group[i][i] = 1
    assert best_result is not None
    remained_points = [points for points in range(0,len(data)) if points not in [item for row in best_result for item in row]]
    # remained_diagram = gd.RipsComplex(distance_matrix=np.array(data)[remained_points][:,remained_points], max_edge_length=100)

    best_result.append(remained_points)
    # print(remained_points)

    label = np.zeros(len(data))
    for idx, group in enumerate(best_result):
        for number_id in group:
            label[number_id] = idx

    return label.astype(np.int8)