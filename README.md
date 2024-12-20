# PHTDA-Net
This Project is under development. No assurance or warranty is provided.

# Table of Contents
[CKA](https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment)


# Introduction

<a id="Grouping"></a>
## Grouping(Simple)
Persistence Calculation: Calculates the persistence diagram from the Rips complex.

Filtration Analysis: Identifies simplices corresponding to specific birth and death filtration values.

Cluster Analysis: Iteratively merges sublists and uses Persistence Bag-of-Words (PBoW) to evaluate the quality of clusters.

Best Group Selection: Determines the best clustering result based on the computed distances and persistence features.

Final Labeling: Assigns labels to the data points corresponding to the identified clusters.


## phcluster
### Prepare Data
1. **Initialization**:
   - Get the current date and time.
   - Initialize `corr_set` to store correlations across rounds.
   - Set the number of normal and attack clients (`normal_nums` and `attack_nums`).
   - Compute the total number of clients (`total_nums`).

2. **Iterate Over Rounds**:
   - For each round in `round_set`:
     - Combine normal and attack model paths for the round.
     - Find the global model path corresponding to the round.

3. **Dataset and Model Initialization**:
   - Initialize the dataset, model, and dataloader.
   - Compute gradients for the global model.

4. **Compute Local Gradients**:
   - For each model in the combined path set:
     - Load the model weights.
     - Compute and store gradients.

5. **Pairwise Gradient Correlation**:
   - Iterate through local gradients in pairs:
     - Reshape gradients for each layer.
     - Compute Linear and Kernel Centered Kernel Alignment (CKA) metrics.
     - Average the CKA metrics and store the correlation in a matrix.

6. **Store and Output Results**:
   - Append the correlation matrix for the round to `corr_set`.
   - Print the size of `corr_set` and the shape of the first correlation matrix.

7. **Save Results**:
    - Save the correlation matrices with `np.savetxt()`.

### Compute PH-Cluster

1. **Iterate Through Correlation Data**:
   - For each round in `corr_set`:
     - Compute the dissimilarity matrix as `1 - corr`.

2. **Heatmap Visualization**:
   - Plot the dissimilarity matrix using a heatmap.

3. **Dimensionality Reduction with MDS**:
   - Initialize MDS with `n_components=2` and `dissimilarity='precomputed'`.
   - Fit and transform the dissimilarity matrix to obtain 2D coordinates.

4. **Scatter Plot of MDS Results**:
   - Scatter plot the MDS results.
   - Annotate each point with its index.

5. **Grouping**:
   - Apply a [grouping function](#Grouping) based on dissimilarity data, total clients, and MDS results.
   - Store the result in `dv_seq`.

6. **Kernel Density Estimation**:
   - Filter `dv_seq` to exclude infinite values and reshape it for KDE.
   - Initialize KDE with a Gaussian kernel and bandwidth of 0.01.
   - Fit the KDE model and compute density estimation on sampled data.

7. **Plot Density Estimation**:
   - Plot the estimated density using a Gaussian kernel.
   - Label the axes and add a legend for clarity.

### Trace Back

Evaluate the performance of the PH-Cluster algorithm along with K-means and DBSCAN clustering methods.
