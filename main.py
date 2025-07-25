import numpy as np
import random
from collections import defaultdict
from typing import List, Dict, Tuple

# 
# STEP 1: DATASET LOADING AND NORMALIZATION
# 

class Dataset:
    """
    Classe per gestire il dataset di ratings degli utenti.
    Normalizza i ratings e crea strutture dati per accesso rapido.
    """
    def __init__(self, ratings: List[Tuple[int, int, float]]):
        """
        Inizializza il dataset con una lista di ratings.
        
        Args:
            ratings: Lista di tuple (user_id, item_id, rating)
        """
        self.ratings = ratings
        # Estrae tutti gli utenti e item unici e li ordina
        self.users = sorted(set(u for u, _, _ in ratings))
        self.items = sorted(set(i for _, i, _ in ratings))
        
        # Strutture dati per accesso rapido:
        self.R = defaultdict(dict)      # R[user][item] = rating normalizzato
        self.Iu = defaultdict(set)      # Iu[user] = set di item valutati dall'utente
        self.Ui = defaultdict(set)      # Ui[item] = set di utenti che hanno valutato l'item
        
        # Trova il range dei ratings per la normalizzazione
        self.min_rating = min(r for _, _, r in ratings)
        self.max_rating = max(r for _, _, r in ratings)

        # Popola le strutture dati normalizzando i ratings
        for u, i, r in ratings:
            r_norm = self.normalize(r)
            self.R[u][i] = r_norm
            self.Iu[u].add(i)
            self.Ui[i].add(u)

    def normalize(self, r: float) -> float:
        """
        Normalizza un rating nel range [0, 1].
        Formula: (r - min + 1) / (max - min + 1)
        """
        return (r - self.min_rating + 1) / (self.max_rating - self.min_rating + 1)

    def denormalize(self, r: float) -> float:
        """
        Converte un rating normalizzato al range originale.
        """
        return r * (self.max_rating - self.min_rating + 1) + self.min_rating - 1

# 
# STEP 2: BIPARTITE REPUTATION-BASED RANKING
# 

def bipartite_ranking(dataset: Dataset, lambda_: float = 0.5, max_iter: int = 100, tol: float = 1e-4):
    """
    Implementa l'algoritmo di Bipartite Reputation-Based Ranking.
    
    Questo algoritmo funziona in modo iterativo:
    1. Calcola il ranking degli item basato sulla reputazione degli utenti
    2. Aggiorna la reputazione degli utenti basata sull'accuratezza delle loro valutazioni
    
    Args:
        dataset: Dataset con i ratings
        lambda_: Parametro di penalizzazione per l'errore (0-1)
        max_iter: Numero massimo di iterazioni
        tol: Tolleranza per la convergenza
    
    Returns:
        Tuple (rankings_item, reputazioni_utenti)
    """
    users, items = dataset.users, dataset.items
    
    # Inizializzazione:
    c = {u: 1.0 for u in users}    # Reputazione iniziale degli utenti (tutti uguali)
    r = {i: 0.5 for i in items}    # Ranking iniziale degli item (neutro)

    for it in range(max_iter):
        # STEP A: Aggiorna i ranking degli item (Equazione 3, riferimento email)
        r_new = {}
        for i in items:
            # Il ranking di un item è la media pesata dei suoi ratings,
            # dove il peso è la reputazione dell'utente che ha dato il rating
            numer = sum(c[u] * dataset.R[u][i] for u in dataset.Ui[i])  # Somma pesata
            denom = sum(c[u] for u in dataset.Ui[i])                    # Somma dei pesi
            r_new[i] = numer / denom if denom != 0 else 0.5

        # STEP B: Aggiorna le reputazioni degli utenti (Equazione 4, riferimento email)
        c_new = {}
        for u in users:
            if not dataset.Iu[u]:  # Se l'utente non ha valutato nessun item
                c_new[u] = 1.0
                continue
            
            # Calcola l'errore medio dell'utente rispetto ai ranking correnti
            err = sum(abs(dataset.R[u][i] - r_new[i]) for i in dataset.Iu[u]) / len(dataset.Iu[u])
            
            # La reputazione diminuisce proporzionalmente all'errore
            c_new[u] = 1 - lambda_ * err

        # STEP C: Controllo convergenza
        # Se i ranking non cambiano significativamente, l'algoritmo è converso
        delta = max(abs(r[i] - r_new[i]) for i in items)
        r, c = r_new, c_new
        if delta < tol:
            break

    return r, c

# 
# STEP 3: USER SIMILARITY AND CLUSTERING
# 

def linear_similarity(u_ratings: Dict[int, float], v_ratings: Dict[int, float], delta_r: float = 1.0) -> float:
    """
    Calcola la similarità lineare tra due utenti basata sui loro ratings comuni.
    
    Formula: 1 - (somma_differenze_assolute) / (delta_r * numero_item_comuni)
    
    Args:
        u_ratings: Dizionario dei ratings dell'utente u
        v_ratings: Dizionario dei ratings dell'utente v  
        delta_r: Range massimo di differenza (default 1.0 per ratings normalizzati)
    
    Returns:
        Valore di similarità tra 0 e 1 (1 = identici, 0 = completamente diversi)
    """
    # Trova gli item valutati da entrambi gli utenti
    common = set(u_ratings) & set(v_ratings)
    if not common:
        return 0.0  # Nessun item in comune
    
    # Calcola la somma delle differenze assolute sui ratings comuni
    diff = sum(abs(u_ratings[i] - v_ratings[i]) for i in common)
    
    # Converte in similarità: più piccola è la differenza, più alta è la similarità
    return 1 - (diff / (delta_r * len(common)))

def build_similarity_graph(dataset: Dataset, alpha: float = 0.95) -> Dict[int, List[int]]:
    """
    Costruisce un grafo di similarità tra utenti.
    
    Due utenti sono connessi nel grafo se la loro similarità supera la soglia alpha.
    
    Args:
        dataset: Dataset con i ratings
        alpha: Soglia di similarità (0-1). Valori alti = clusters più stretti
    
    Returns:
        Grafo come dizionario: {utente: [lista_utenti_simili]}
    """
    graph = defaultdict(list)
    users = dataset.users
    
    # Confronta ogni coppia di utenti
    for i, u in enumerate(users):
        for v in users[i+1:]:  # Evita confronti duplicati
            sim = linear_similarity(dataset.R[u], dataset.R[v])
            if sim > alpha:  # Se abbastanza simili, li connette nel grafo
                graph[u].append(v)
                graph[v].append(u)
    return graph

def connected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    Trova le componenti connesse del grafo (i clusters di utenti simili).
    
    Usa un algoritmo di visita in profondità (DFS) per trovare tutti i nodi
    raggiungibili da ogni nodo non ancora visitato.
    
    Args:
        graph: Grafo di adiacenza
    
    Returns:
        Lista di clusters, dove ogni cluster è una lista di user_id
    """
    visited = set()
    components = []
    
    for node in graph:
        if node not in visited:
            # Inizia una nuova componente connessa
            stack = [node]
            component = []
            
            # DFS per visitare tutti i nodi connessi
            while stack:
                u = stack.pop()
                if u not in visited:
                    visited.add(u)
                    component.append(u)
                    stack.extend(graph[u])  # Aggiunge i vicini allo stack
            
            components.append(component)
    return components

# 
# STEP 4: CLUSTER-BASED RANKING
# 

def cluster_based_ranking(dataset: Dataset, alpha: float = 0.75, lambda_: float = 0.5):
    """
    Applica il bipartite ranking separatamente a ogni cluster di utenti simili.
    
    Processo:
    1. Costruisce grafo di similarità tra utenti
    2. Trova clusters (componenti connesse)  
    3. Applica bipartite ranking su ogni cluster con ≥10 utenti
    4. Calcola ranking di fallback per utenti fuori dai clusters
    
    Args:
        dataset: Dataset completo
        alpha: Soglia di similarità per formare clusters
        lambda_: Parametro di penalizzazione per bipartite ranking
    
    Returns:
        Tuple (cluster_rankings, fallback_ranking, outside_users)
    """
    # Costruisce grafo di similarità e trova clusters
    graph = build_similarity_graph(dataset, alpha)
    clusters = connected_components(graph)
    
    cluster_rankings = []
    all_cluster_users = set()
    cluster_item_scores = defaultdict(list)  # Per calcolare fallback

    # Processa ogni cluster
    for cluster in clusters:
        if len(cluster) < 5:  # Scarta clusters troppo piccoli (rumore)
            continue
            
        all_cluster_users.update(cluster)
        
        # Crea un sub-dataset solo con i ratings degli utenti nel cluster
        cluster_ratings = [(u, i, r) for u in cluster for i, r in dataset.R[u].items()]
        sub_dataset = Dataset(cluster_ratings)
        
        # Applica bipartite ranking al cluster
        r, _ = bipartite_ranking(sub_dataset, lambda_)
        
        # Memorizza i risultati
        for i, score in r.items():
            cluster_item_scores[i].append(score)
        cluster_rankings.append((cluster, r))

    # Calcola ranking di fallback come media dei rankings dei clusters
    fallback_ranking = {}
    for i, scores in cluster_item_scores.items():
        fallback_ranking[i] = sum(scores) / len(scores)

    # Identifica utenti non inclusi in nessun cluster valido
    outside_users = [u for u in dataset.users if u not in all_cluster_users]
    
    return cluster_rankings, fallback_ranking, outside_users

# 
# STEP 5: UARS (USER-AGNOSTIC RANKING SYSTEM)
# 

def uars_ranking(dataset: Dataset) -> Dict[int, float]:
    """
    Implementa UARS (User-Agnostic Ranking System) per ranking globale.
    
    Per ogni item, rimuove iterativamente i ratings anomali (outliers) e 
    calcola la media dei ratings rimanenti.
    
    Processo:
    1. Per ogni item, calcola media e deviazione standard dei suoi ratings
    2. Rimuove ratings che si discostano troppo dalla media
    3. Ripete fino a quando non ci sono più outliers
    4. Il ranking finale è la media dei ratings "puliti"
    
    Args:
        dataset: Dataset con i ratings
    
    Returns:
        Dizionario {item_id: ranking_score}
    """
    rankings = {}
    
    for i in dataset.items:
        # Ottiene tutti i ratings per questo item
        ratings = [dataset.R[u][i] for u in dataset.Ui[i]]
        
        # Rimozione iterativa degli outliers
        while True:
            mu = np.mean(ratings)  # Media corrente
            sigma = np.std(ratings, ddof=1) if len(ratings) > 1 else 0  # Deviazione standard
            
            # Mantiene solo i ratings entro una deviazione standard dalla media
            filtered = [r for r in ratings if (r - mu) ** 2 <= sigma]
            
            if len(filtered) == len(ratings):
                # Non ci sono più outliers da rimuovere
                break
            ratings = filtered
        
        # Il ranking è la media dei ratings "puliti"
        rankings[i] = np.mean(ratings) if ratings else 0.5
    
    return rankings

# 
# UTILITY FUNCTIONS FOR OUTPUT
# 

def print_top_bottom(rankings, N=5):
    """
    Stampa i top N e bottom N item da un dizionario di rankings.
    """
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

def print_tb(cluster_results, fallback_ranking, uars, outside_users, num=5):
    """
    Stampa un riepilogo dei risultati di tutti gli algoritmi.
    """
    # Risultati per ogni cluster
    for idx, (cluster, rankings) in enumerate(cluster_results):
        print(f"\nCluster {idx+1} (Users: {cluster})[{len(cluster)}]")
        print_top_bottom(rankings, N=num)

    # Ranking di fallback
    print("\nFallback Ranking for Users Outside Clusters:")
    print_top_bottom(fallback_ranking, N=num)

    print(f"\nUsers outside clusters: {outside_users}[{len(outside_users)}]")

    # Ranking globale UARS
    print("\nUARS Global Rankings:")
    uars = uars_ranking(dataset)
    print_top_bottom(uars, N=num)

def print_all(cluster_results, fallback_ranking, uars, outside_users):
    """
    Stampa tutti i risultati in dettaglio (tutti gli item).
    """
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

# 
# SYNTHETIC DATA GENERATION
# 

def generate_grouped_ratings() -> List[Tuple[int, int, int]]:
    """
    Genera dati sintetici per testare gli algoritmi.
    
    Crea due gruppi di utenti con preferenze opposte:
    - Gruppo 1: preferisce item 101-175, non gradisce item 176-250
    - Gruppo 2: preferisce item 176-250, non gradisce item 101-175
    - Utenti casuali: danno ratings random su item casuali
    
    Questo setup permette di testare se gli algoritmi riescono a identificare
    i clusters e dare ranking appropriati per ogni gruppo.
    """
    random.seed()  # Seed casuale per variabilità
    raw_ratings = []
    
    # Parametri di configurazione
    min_items = 101
    max_items = 250
    min_users = 1
    max_users = 201
    num_items_preferred = 30      # Item che ogni utente valuta positivamente
    num_items_non_preferred = 60  # Item che ogni utente valuta negativamente
    num_random_users = 50         # Utenti con comportamento casuale
    
    # Divide gli item in due gruppi
    group1_items = range(min_items, (min_items + max_items)//2)          # 101-175
    group2_items = range((min_items + max_items)//2 + 1, max_items)      # 176-250
    all_items = range(min_items, max_items)
    
    #  GRUPPO 1: Utenti 1-100 
    # Preferiscono item del gruppo 1, non gradiscono item del gruppo 2
    for user_id in range(min_users, (min_users + max_users)//2):
        # Seleziona casualmente alcuni item preferiti e non preferiti
        preferred = random.sample(group1_items, k=num_items_preferred)
        non_preferred = random.sample(group2_items, k=num_items_non_preferred)
        
        # Assegna ratings alti agli item preferiti
        for i in preferred:
            raw_ratings.append((user_id, i, random.randint(9, 10)))
        # Assegna ratings bassi agli item non preferiti    
        for i in non_preferred:
            raw_ratings.append((user_id, i, random.randint(1, 3)))
    
    #  GRUPPO 2: Utenti 101-200 
    # Preferenze opposte al gruppo 1
    for user_id in range((min_users + max_users)//2, max_users):
        preferred = random.sample(group2_items, k=num_items_preferred)
        non_preferred = random.sample(group1_items, k=num_items_non_preferred)
        
        for i in preferred:
            raw_ratings.append((user_id, i, random.randint(9, 10)))
        for i in non_preferred:
            raw_ratings.append((user_id, i, random.randint(1, 3)))

    #  UTENTI CASUALI 
    # Aggiunge "rumore" al dataset con utenti che hanno comportamento casuale
    for user_id in range(max_items, max_items + num_random_users):
        items = random.sample(all_items, k=20)
        for i in items:
            raw_ratings.append((user_id, i, random.randint(1, 10)))
    
    return raw_ratings

# 
# MAIN 
# 

def load_movielens_sample(path: str = "ml-1m/ratings.dat", sample_size: int = 5000):
    ratings = load_movielens_1m(path)
    return random.sample(ratings, k=sample_size)

def load_movielens_1m(path: str = "ml-1m/ratings.dat") -> List[Tuple[int, int, int]]:
    """
    Carica i dati MovieLens 1M dal file ratings.dat.
    
    Returns:
        Lista di tuple (user_id, item_id, rating)
    """
    ratings = []
    with open(path, "r") as f:
        for line in f:
            user_id, item_id, rating, _ = line.strip().split("::")
            ratings.append((int(user_id), int(item_id), int(rating)))
    return ratings
if __name__ == "__main__":
    """
    Esecuzione principale che:
    1. Genera dati sintetici
    2. Applica clustering e bipartite ranking
    3. Calcola UARS ranking globale  
    4. Stampa e confronta i risultati
    """
    
    # Genera dataset sintetico
    # dataset = Dataset(generate_grouped_ratings())
    
    dataset = Dataset(load_movielens_sample("ml-1m/ratings.dat"))
    # dataset = Dataset(generate_grouped_ratings())
    # Applica cluster-based ranking
    # alpha=0.937: soglia alta per clusters molto simili
    cluster_results, fallback_ranking, outside_users = cluster_based_ranking(dataset, alpha=0.95)
    
    # Calcola ranking globale con UARS
    uars = uars_ranking(dataset)
    
    # Stampa risultati riassuntivi
    print_tb(cluster_results, fallback_ranking, uars, outside_users)
    
    # Per stampare tutti i dettagli, decommentare la riga seguente:
    # print_all(cluster_results, fallback_ranking, uars, outside_users)
