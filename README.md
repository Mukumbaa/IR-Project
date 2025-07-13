# Bipartite Reputation-Based Ranking System

This Python script implements a reputation-aware ranking algorithm designed to evaluate items based on user ratings while identifying and filtering unreliable users. It includes clustering of users based on rating similarity and offers both user-specific and user-agnostic ranking methods.

## Overview

The script is organized into the following key components:

1. **Dataset Handling**
2. **Bipartite Reputation-Based Ranking**
3. **User Similarity & Clustering**
4. **Cluster-Based Ranking**
5. **User-Agnostic Ranking (UARS)**
6. **Result Display Functions**
7. **Synthetic Data Generation**

---

## Classes & Functions

### `class Dataset`

Initializes and preprocesses the dataset of user-item ratings.

#### Parameters:

* `ratings: List[Tuple[int, int, float]]`
  A list of tuples `(user_id, item_id, rating)`.

#### Attributes:

* `R`: Normalized ratings by user and item.
* `Iu`: Set of items rated by each user.
* `Ui`: Set of users that rated each item.
* `min_rating`, `max_rating`: Track rating bounds for normalization.

#### Methods:

* `normalize(r: float) -> float`
  Scales rating to a normalized range \[0, 1].

* `denormalize(r: float) -> float`
  Converts normalized ratings back to the original scale.

---

### `bipartite_ranking(dataset, lambda_=0.5, max_iter=100, tol=1e-4)`

Implements Bipartite Reputation-Based Ranking based on iterative updates of user reputations and item rankings.

#### Parameters:

* `dataset`: An instance of `Dataset`.
* `lambda_`: Sensitivity to user error in reputation computation.
* `max_iter`: Max number of iterations.
* `tol`: Convergence tolerance.

#### Returns:

* `r`: Dictionary of item rankings.
* `c`: Dictionary of user reputations.

---

### `linear_similarity(u_ratings, v_ratings, delta_r=1.0)`

Computes linear similarity between two users based on their common item ratings.

#### Parameters:

* `u_ratings`, `v_ratings`: Dicts of item ratings for users `u` and `v`.
* `delta_r`: Normalization constant.

#### Returns:

* Similarity score in \[0,1].

---

### `build_similarity_graph(dataset, alpha=0.95)`

Builds a user similarity graph where an edge is added between users with similarity > `alpha`.

#### Returns:

* A dictionary representing the adjacency list of the graph.

---

### `connected_components(graph)`

Finds all connected components (clusters) in a user similarity graph using DFS.

#### Returns:

* List of user clusters (lists of user IDs).

---

### `cluster_based_ranking(dataset, alpha=0.75, lambda_=0.5)`

Applies bipartite ranking within each user cluster.

#### Parameters:

* `alpha`: Similarity threshold for connecting users in the graph.
* `lambda_`: Passed to the bipartite ranking function.

#### Returns:

* `cluster_rankings`: List of cluster results with rankings.
* `fallback_ranking`: Weighted average item rankings for users outside clusters.
* `outside_users`: List of user IDs not included in any cluster.

---

### `uars_ranking(dataset)`

Implements **UARS** (User-Agnostic Ranking System). Filters outliers and returns a robust average per item.

#### Returns:

* Dictionary of item rankings.

---

### `print_top_bottom(rankings, N=5)`

Prints the top `N` and bottom `N` items from a ranking.

---

### `print_tb(cluster_results, fallback_ranking, uars, outside_users, num=5)`

Prints summary top/bottom rankings per cluster, fallback group, and UARS method.

---

### `print_all(cluster_results, fallback_ranking, uars, outside_users)`

Prints **all** item rankings for each cluster, fallback group, and UARS.

---

### `generate_grouped_ratings() -> List[Tuple[int, int, int]]`

Creates synthetic user-item rating data for testing.

---
