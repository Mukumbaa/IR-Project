import numpy as np
import random
from collections import defaultdict
from typing import List, Dict, Tuple

# 
# Step 1: Load and normalize ratings
# 
class Dataset:
    def __init__(self, ratings: List[Tuple[int, int, float]]):
        self.ratings = ratings
        self.users = sorted(set(u for u, _, _ in ratings))
        self.items = sorted(set(i for _, i, _ in ratings))
        self.R = defaultdict(dict)
        self.Iu = defaultdict(set)
        self.Ui = defaultdict(set)
        self.min_rating = min(r for _, _, r in ratings)
        self.max_rating = max(r for _, _, r in ratings)

        for u, i, r in ratings:
            r_norm = self.normalize(r)
            self.R[u][i] = r_norm
            self.Iu[u].add(i)
            self.Ui[i].add(u)

    def normalize(self, r: float) -> float:
        return (r - self.min_rating + 1) / (self.max_rating - self.min_rating + 1)

    def denormalize(self, r: float) -> float:
        return r * (self.max_rating - self.min_rating + 1) + self.min_rating - 1

# 
# Step 2: Bipartite Reputation-Based Ranking (Eq. 3 + 4)
# 
def bipartite_ranking(dataset: Dataset, lambda_: float = 0.5, max_iter: int = 100, tol: float = 1e-4):
    users, items = dataset.users, dataset.items
    c = {u: 1.0 for u in users}  # Initialize reputations
    r = {i: 0.5 for i in items}  # Initialize rankings

    for it in range(max_iter):
        # Update rankings (Eq. 3)
        r_new = {}
        for i in items:
            numer = sum(c[u] * dataset.R[u][i] for u in dataset.Ui[i])
            denom = sum(c[u] for u in dataset.Ui[i])
            r_new[i] = numer / denom if denom != 0 else 0.5

        # Update reputations (Eq. 4)
        c_new = {}
        for u in users:
            if not dataset.Iu[u]:
                c_new[u] = 1.0
                continue
            err = sum(abs(dataset.R[u][i] - r_new[i]) for i in dataset.Iu[u]) / len(dataset.Iu[u])
            c_new[u] = 1 - lambda_ * err

        # Convergence check
        delta = max(abs(r[i] - r_new[i]) for i in items)
        r, c = r_new, c_new
        if delta < tol:
            break

    return r, c

# 
# Step 3: User Similarity and Clustering (Algorithm 1 in TKDD)
# 
def linear_similarity(u_ratings: Dict[int, float], v_ratings: Dict[int, float], delta_r: float = 1.0) -> float:
    common = set(u_ratings) & set(v_ratings)
    if not common:
        return 0.0
    diff = sum(abs(u_ratings[i] - v_ratings[i]) for i in common)
    return 1 - (diff / (delta_r * len(common)))

def build_similarity_graph(dataset: Dataset, alpha: float = 0.95) -> Dict[int, List[int]]:
    graph = defaultdict(list)
    users = dataset.users
    for i, u in enumerate(users):
        for v in users[i+1:]:
            sim = linear_similarity(dataset.R[u], dataset.R[v])
            if sim > alpha:
                graph[u].append(v)
                graph[v].append(u)
    return graph

def connected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    visited = set()
    components = []
    for node in graph:
        if node not in visited:
            stack = [node]
            component = []
            while stack:
                u = stack.pop()
                if u not in visited:
                    visited.add(u)
                    component.append(u)
                    stack.extend(graph[u])
            components.append(component)
    return components

# 
# Step 4: Apply bipartite ranking within each cluster
# 
def cluster_based_ranking(dataset: Dataset, alpha: float = 0.75, lambda_: float = 0.5):
    graph = build_similarity_graph(dataset, alpha)
    clusters = connected_components(graph)
    cluster_rankings = []
    all_cluster_users = set()
    cluster_item_scores = defaultdict(list)

    for cluster in clusters:
        if len(cluster) < 10:
            continue  # Discard small clusters
        all_cluster_users.update(cluster)
        cluster_ratings = [(u, i, r) for u in cluster for i, r in dataset.R[u].items()]
        sub_dataset = Dataset(cluster_ratings)
        r, _ = bipartite_ranking(sub_dataset, lambda_)
        for i, score in r.items():
            cluster_item_scores[i].append(score)
        cluster_rankings.append((cluster, r))

    # Step 7: Weighted average fallback for users outside clusters
    fallback_ranking = {}
    for i, scores in cluster_item_scores.items():
        fallback_ranking[i] = sum(scores) / len(scores)

    outside_users = [u for u in dataset.users if u not in all_cluster_users]
    return cluster_rankings, fallback_ranking, outside_users

# 
# Step 5: UARS (User-Agnostic Ranking System from SIGIR)
# 
def uars_ranking(dataset: Dataset) -> Dict[int, float]:
    rankings = {}
    for i in dataset.items:
        ratings = [dataset.R[u][i] for u in dataset.Ui[i]]
        while True:
            mu = np.mean(ratings)
            sigma = np.std(ratings, ddof=1) if len(ratings) > 1 else 0
            filtered = [r for r in ratings if (r - mu) ** 2 <= sigma]
            if len(filtered) == len(ratings):
                break
            ratings = filtered
        rankings[i] = np.mean(ratings) if ratings else 0.5
    return rankings

# 
# Example Usage Placeholder
# 
def print_top_bottom(rankings, N = 5):

    if rankings:
        sorted_items = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop {N} Items:")
        for i, score in sorted_items[:N]:
            print(f"  Item {i}: {score:.4f} (denorm: {dataset.denormalize(score):.2f})")

        print(f"Bottom {N} Items:")
        for i, score in sorted_items[-N:]:
            print(f"  Item {i}: {score:.4f} (denorm: {dataset.denormalize(score):.2f})")
    else:
        print("No rankings available")

def print_tb(cluster_results, fallback_ranking, uars, outside_users, num = 5):
    for idx, (cluster, rankings) in enumerate(cluster_results):
        print(f"\nCluster {idx+1} (Users: {cluster})[{len(cluster)}]")
        print_top_bottom(rankings, N = num)

    print("\nFallback Ranking for Users Outside Clusters:")
    print_top_bottom(fallback_ranking,N = num)

    print(f"\nUsers outside clusters: {outside_users}[{len(outside_users)}]")

    print("\nUARS Global Rankings:")
    uars = uars_ranking(dataset)
    print_top_bottom(uars,N = num)

def print_all(cluster_results, fallback_ranking, uars, outside_users):

    for idx, (cluster, rankings) in enumerate(cluster_results):
        print(f"\nCluster {idx+1} (Users: {cluster})")
        for i, score in rankings.items():
            print(f"Item {i}: {score:.4f} (denorm: {dataset.denormalize(score):.2f})")

    print("\nFallback Ranking for Users Outside Clusters:")
    for i, score in fallback_ranking.items():
        print(f"Item {i}: {score:.4f} (denorm: {dataset.denormalize(score):.2f})")

    print(f"\nUsers outside clusters: {outside_users}")

    print("\nUARS Global Rankings:")
    for i, score in uars.items():
        print(f"Item {i}: {score:.4f} (denorm: {dataset.denormalize(score):.2f})")

def generate_grouped_ratings() -> List[Tuple[int, int, int]]:
    random.seed() # 42
    raw_ratings = []
    
    min_items = 101 # 101
    max_items = 250 # 180 250
    min_users = 1 # 1
    max_users = 201 # 101 201
    num_items_preferred = 30 # 15 30
    num_items_non_preferred = 60 # 5 60
    num_random_users = 50 # 5 50
    
    group1_items = range(min_items, (min_items + max_items)//2)
    group2_items = range((min_items + max_items)//2 + 1, max_items)
    all_items = range(min_items, max_items)
    
    # Group 1 (users 1-50)
    for user_id in range(min_users, (min_users + max_users)//2):
        preferred = random.sample(group1_items, k=num_items_preferred)
        non_preferred = random.sample(group2_items, k=num_items_non_preferred)
        for i in preferred:
            raw_ratings.append((user_id, i, random.randint(9, 10)))
        for i in non_preferred:
            raw_ratings.append((user_id, i, random.randint(1, 3)))
    
    # Group 2 (users 51-100)
    for user_id in range((min_users + max_users)//2, max_users):
        preferred = random.sample(group2_items, k=num_items_preferred)
        non_preferred = random.sample(group1_items, k=num_items_non_preferred)
        for i in preferred:
            raw_ratings.append((user_id, i, random.randint(9, 10)))
        for i in non_preferred:
            raw_ratings.append((user_id, i, random.randint(1, 3)))


    # NEW
    # random vote added on all items
    # for user_id in range((max_users * 40)//100, (max_users * 50)//100):
    #     ri = random.sample(all_items, k=5)
    #     for i in ri:
    #         raw_ratings.append((user_id, i, random.randint(1,10)))

    # random 
    for user_id in range(max_items, max_items+num_random_users):
        items = random.sample(all_items, k=20) # 20
        for i in items:
            raw_ratings.append((user_id, i, random.randint(1, 10)))
    
    return raw_ratings

# def generate_grouped_ratings() -> List[Tuple[int, int, int]]:
#     random.seed(42)
#     raw_ratings = []
#     group1_items = range(101, 140)
#     group2_items = range(140, 180)
#     all_items = range(101, 180)
    
#     # Group 1 (users 1-50)
#     for user_id in range(1, 51):
#         preferred = random.sample(group1_items, k=15)
#         non_preferred = random.sample(group2_items, k=5)
#         for i in preferred:
#             raw_ratings.append((user_id, i, random.randint(7, 10)))
#         for i in non_preferred:
#             raw_ratings.append((user_id, i, random.randint(1, 3)))
    
#     # Group 2 (users 51-100)
#     for user_id in range(51, 101):
#         preferred = random.sample(group2_items, k=15)
#         non_preferred = random.sample(group1_items, k=5)
#         for i in preferred:
#             raw_ratings.append((user_id, i, random.randint(8, 10)))
#         for i in non_preferred:
#             raw_ratings.append((user_id, i, random.randint(1, 4)))
#     # NEW
#     # random vote added on all items
#     for user_id in range(40, 60):
#         ri = random.sample(all_items, k=5)
#         for i in ri:
#             raw_ratings.append((user_id, i, random.randint(1,10)))

#     random 
#     for user_id in range(101, 106):
#         items = random.sample(range(101, 180), k=20)
#         for i in items:
#             raw_ratings.append((user_id, i, random.randint(1, 10)))
    
#     return raw_ratings
if __name__ == "__main__":
    # synthetic ratings: (user_id, item_id, rating)

    # num_users = 100
    # num_items = 40
    # raw_ratings = []

    # random.seed(42)
    # for user_id in range(1, num_users + 1):
    #     rated_items = random.sample(range(101, 101 + num_items), k=random.randint(10, 30))
    #     for item_id in rated_items:
    #         rating = random.randint(1, 10)
    #         raw_ratings.append((user_id, item_id, rating))


    dataset = Dataset(generate_grouped_ratings())
    cluster_results, fallback_ranking, outside_users = cluster_based_ranking(dataset, alpha=0.937) # , 91 935
    uars = uars_ranking(dataset)
    
    print_tb(cluster_results, fallback_ranking, uars, outside_users)
    # print_all(cluster_results, fallback_ranking, uars, outside_users)
