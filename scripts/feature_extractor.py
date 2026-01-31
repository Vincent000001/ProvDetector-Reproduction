"""
ProvDetector Feature Extractor - Following the paper methodology
Implements: Path-to-Sentence, Doc2Vec, Frequency Database, Rarest Path
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
import networkx as nx
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import os
import pickle


class ProvDetectorFeatureExtractor:
    """
    ProvDetector feature extractor following the paper methodology:
    1. Extract causal paths from provenance graphs
    2. Convert paths to sentences
    3. Build frequency database
    4. Select rarest paths
    5. Use Doc2Vec for embedding
    """
    
    def __init__(self, frequency_db_path: str = None, doc2vec_model_path: str = None):
        """
        Initialize ProvDetector feature extractor
        
        Args:
            frequency_db_path: Path to frequency database (for inference)
            doc2vec_model_path: Path to Doc2Vec model (for inference)
        """
        self.frequency_db = {}  # Path -> frequency count
        self.doc2vec_model = None
        self.frequency_db_path = frequency_db_path
        self.doc2vec_model_path = doc2vec_model_path
        
        # Load models if provided
        if frequency_db_path and os.path.exists(frequency_db_path):
            with open(frequency_db_path, 'rb') as f:
                self.frequency_db = pickle.load(f)
        
        if doc2vec_model_path and os.path.exists(doc2vec_model_path):
            self.doc2vec_model = Doc2Vec.load(doc2vec_model_path)
    
    def extract_paths_from_graph(self, df: pd.DataFrame, max_path_length: int = 6, verbose: bool = False) -> List[List[str]]:
        """
        Extract causal paths from provenance graph
        
        Args:
            df: Provenance graph DataFrame
            max_path_length: Maximum path length to extract
            verbose: Print progress messages
            
        Returns:
            List of paths, where each path is a list of node IDs
        """
        paths = []
        
        try:
            # Build directed graph
            G = nx.DiGraph()
            
            # Add edges with attributes
            for _, row in df.iterrows():
                source = str(row['sourceId'])
                dest = str(row['destinationId'])
                action = str(row.get('action', ''))
                
                if source != 'graph_head' and dest != 'graph_head':
                    # Store action as edge attribute
                    G.add_edge(source, dest, action=action)
            
            if G.number_of_nodes() == 0:
                return paths
            
            # Extract paths starting from root nodes (nodes with in-degree 0)
            root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
            
            # If no root nodes, use all nodes as potential starts
            if not root_nodes:
                root_nodes = list(G.nodes())[:10]  # Limit to avoid explosion
            
            # Extract paths using DFS
            # Extract paths from root nodes (limit to avoid explosion but get good coverage)
            max_roots = min(15, len(root_nodes))  # Balanced: good coverage, reasonable speed
            max_paths_per_graph = 600  # Balanced: good coverage, reasonable speed
            for root in root_nodes[:max_roots]:
                if len(paths) >= max_paths_per_graph:
                    break
                self._extract_paths_dfs(G, root, [], paths, max_path_length, max_paths_per_graph)
        
        except Exception as e:
            print(f"Error extracting paths: {e}")
        
        return paths
    
    def _extract_paths_dfs(self, G: nx.DiGraph, node: str, current_path: List[str], 
                           all_paths: List[List[str]], max_length: int, max_paths: int = 500):
        """DFS to extract paths from graph with early stopping"""
        # Early stopping if we have enough paths
        if len(all_paths) >= max_paths:
            return
        
        if len(current_path) >= max_length:
            all_paths.append(current_path.copy())
            return
        
        if node in current_path:  # Avoid cycles
            all_paths.append(current_path.copy())
            return
        
        current_path.append(node)
        
        # Get neighbors
        neighbors = list(G.successors(node))
        
        if not neighbors:
            # Leaf node - save path
            all_paths.append(current_path.copy())
        else:
            # Continue to neighbors
            # Follow more branches for better path diversity
            max_branches = min(6, len(neighbors))  # Balanced: good diversity, reasonable speed
            for neighbor in neighbors[:max_branches]:
                if len(all_paths) >= max_paths:
                    break
                self._extract_paths_dfs(G, neighbor, current_path, all_paths, max_length, max_paths)
        
        current_path.pop()
    
    def path_to_sentence(self, path: List[str], df: pd.DataFrame) -> str:
        """
        Convert a path to a sentence representation
        Following ProvDetector paper: path -> sentence with node types and actions
        
        Args:
            path: List of node IDs
            df: Original DataFrame for node information
            
        Returns:
            Sentence string representation
        """
        if len(path) < 2:
            return ""
        
        sentence_tokens = []
        
        # Create a mapping of node IDs to their types
        node_types = {}
        for _, row in df.iterrows():
            source = str(row['sourceId'])
            dest = str(row['destinationId'])
            node_types[source] = row.get('sourceType', 'unknown')
            node_types[dest] = row.get('destinationType', 'unknown')
        
        # Convert path to sentence
        for i in range(len(path) - 1):
            source = path[i]
            dest = path[i + 1]
            
            # Get edge action
            edge_rows = df[(df['sourceId'] == source) & (df['destinationId'] == dest)]
            if len(edge_rows) > 0:
                action = edge_rows.iloc[0].get('action', 'connect')
            else:
                action = 'connect'
            
            source_type = node_types.get(source, 'unknown')
            dest_type = node_types.get(dest, 'unknown')
            
            # Create sentence token: source_type-action-dest_type
            token = f"{source_type}_{action}_{dest_type}"
            sentence_tokens.append(token)
        
        return " ".join(sentence_tokens)
    
    def build_frequency_database(self, all_paths_sentences: List[str]):
        """
        Build frequency database from all path sentences
        This tracks how often each path pattern appears
        
        Args:
            all_paths_sentences: List of path sentences from all training samples
        """
        # Count frequency of each path sentence
        path_counts = Counter(all_paths_sentences)
        self.frequency_db = dict(path_counts)
    
    def get_rarest_paths(self, paths_sentences: List[str], top_k: int = 50, 
                        return_weights: bool = False, ensure_diversity: bool = True) -> List[str]:
        """
        Select rarest paths based on frequency database
        Following ProvDetector: rare paths are more indicative of malicious behavior
        
        Args:
            paths_sentences: List of path sentences for current sample
            top_k: Number of rarest paths to select
            return_weights: If True, return (paths, weights) tuple
            ensure_diversity: If True, ensure selected paths are diverse (not too similar)
            
        Returns:
            List of rarest path sentences, or (paths, weights) if return_weights=True
        """
        if not self.frequency_db:
            # If no frequency DB, return all paths
            if return_weights:
                return paths_sentences[:top_k], [1.0] * min(len(paths_sentences), top_k)
            return paths_sentences[:top_k]
        
        # Calculate rarity scores (lower frequency = rarer)
        # Use TF-IDF style: log(1 + total_paths / frequency)
        total_paths = sum(self.frequency_db.values())
        path_rarity = []
        for path in paths_sentences:
            frequency = self.frequency_db.get(path, 0)
            # TF-IDF style rarity: log(1 + total/freq) - higher for rarer paths
            if frequency > 0:
                rarity_score = np.log(1 + total_paths / frequency)
            else:
                rarity_score = np.log(1 + total_paths)  # Maximum rarity for unseen paths
            
            # Also consider path length (longer paths might be more informative)
            path_length = len(path.split())
            length_bonus = 1.0 + 0.1 * path_length  # Slight bonus for longer paths
            
            final_score = rarity_score * length_bonus
            path_rarity.append((path, final_score, frequency, rarity_score))
        
        # Sort by rarity (rarest first)
        path_rarity.sort(key=lambda x: x[1], reverse=True)
        
        # Select diverse paths if requested
        if ensure_diversity and len(path_rarity) > top_k:
            selected_paths = []
            selected_weights = []
            seen_tokens = set()
            
            for path, score, freq, rarity in path_rarity:
                if len(selected_paths) >= top_k:
                    break
                
                # Check diversity: don't select paths that are too similar
                path_tokens = set(path.split())
                overlap = len(path_tokens & seen_tokens) / max(len(path_tokens), 1)
                
                # Allow some overlap but prefer diverse paths
                if overlap < 0.7 or len(selected_paths) < top_k // 2:
                    selected_paths.append(path)
                    selected_weights.append(rarity)  # Use original rarity as weight
                    seen_tokens.update(path_tokens)
            
            # Fill remaining slots with rarest paths if needed
            for path, score, freq, rarity in path_rarity:
                if len(selected_paths) >= top_k:
                    break
                if path not in selected_paths:
                    selected_paths.append(path)
                    selected_weights.append(rarity)
        else:
            selected_paths = [path for path, _, _, _ in path_rarity[:top_k]]
            selected_weights = [rarity for _, _, _, rarity in path_rarity[:top_k]]
        
        # Normalize weights
        if selected_weights:
            max_weight = max(selected_weights)
            if max_weight > 0:
                selected_weights = [w / max_weight for w in selected_weights]
        
        if return_weights:
            return selected_paths, selected_weights
        return selected_paths
    
    def train_doc2vec(self, all_paths_sentences: List[str], vector_size: int = 100, 
                     epochs: int = 20, min_count: int = 1):
        """
        Train Doc2Vec model on path sentences
        Following ProvDetector: use Doc2Vec to embed paths into vectors
        
        Args:
            all_paths_sentences: List of all path sentences from training data
            vector_size: Dimension of embedding vectors
            epochs: Number of training epochs
            min_count: Minimum word count
        """
        # Create TaggedDocuments for Doc2Vec
        documents = [TaggedDocument(words=path.split(), tags=[i]) 
                    for i, path in enumerate(all_paths_sentences)]
        
        # Train Doc2Vec model with improved hyperparameters
        self.doc2vec_model = Doc2Vec(
            documents=documents,
            vector_size=vector_size,
            window=8,  # Increased window for better context
            min_count=min_count,
            workers=4,
            epochs=epochs,
            dm=1,  # PV-DM model
            dbow_words=0,  # Don't train word vectors in DBOW mode
            alpha=0.025,  # Initial learning rate
            min_alpha=0.0001,  # Minimum learning rate
            negative=5,  # Negative sampling
            hs=0,  # Use negative sampling instead of hierarchical softmax
            sample=1e-4,  # Subsampling threshold
            ns_exponent=0.75  # Negative sampling exponent
        )
    
    def extract_features(self, df: pd.DataFrame, use_rarest: bool = True, 
                        top_k_paths: int = 50, aggregation: str = 'enhanced',
                        use_weighted: bool = True) -> np.ndarray:
        """
        Extract features using ProvDetector methodology
        
        Args:
            df: Provenance graph DataFrame
            use_rarest: Whether to use rarest paths
            top_k_paths: Number of paths to use
            aggregation: How to aggregate path embeddings ('mean', 'max', 'concat', 'enhanced')
            
        Returns:
            Feature vector (Doc2Vec embeddings)
        """
        # Extract paths
        paths = self.extract_paths_from_graph(df)
        
        if not paths:
            # Return zero vector if no paths
            if self.doc2vec_model:
                vec_size = self.doc2vec_model.vector_size
            else:
                vec_size = 100
            if aggregation == 'concat' or aggregation == 'enhanced':
                return np.zeros(vec_size * top_k_paths)
            return np.zeros(vec_size)
        
        # Convert paths to sentences
        paths_sentences = [self.path_to_sentence(path, df) for path in paths]
        paths_sentences = [p for p in paths_sentences if p]  # Remove empty
        
        if not paths_sentences:
            if self.doc2vec_model:
                vec_size = self.doc2vec_model.vector_size
            else:
                vec_size = 100
            if aggregation == 'concat' or aggregation == 'enhanced':
                return np.zeros(vec_size * top_k_paths)
            return np.zeros(vec_size)
        
        # Select rarest paths if enabled
        if use_rarest and self.frequency_db:
            if use_weighted:
                selected_paths, path_weights = self.get_rarest_paths(
                    paths_sentences, top_k_paths, return_weights=True, ensure_diversity=True
                )
            else:
                selected_paths = self.get_rarest_paths(paths_sentences, top_k_paths, 
                                                      return_weights=False, ensure_diversity=True)
                path_weights = [1.0] * len(selected_paths)
        else:
            selected_paths = paths_sentences[:top_k_paths]
            path_weights = [1.0] * len(selected_paths)
        
        # Get Doc2Vec embeddings
        if self.doc2vec_model:
            embeddings = []
            weights = []
            for path, weight in zip(selected_paths, path_weights):
                words = path.split()
                if words:
                    try:
                        embedding = self.doc2vec_model.infer_vector(words, epochs=15, alpha=0.025)
                        embeddings.append(embedding)
                        weights.append(weight)
                    except:
                        continue
            
            if not embeddings:
                vec_size = self.doc2vec_model.vector_size
                if aggregation == 'concat' or aggregation == 'enhanced':
                    return np.zeros(vec_size * top_k_paths)
                return np.zeros(vec_size)
            
            # Normalize weights
            if weights and sum(weights) > 0:
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize to sum to 1
            else:
                weights = np.ones(len(embeddings)) / len(embeddings)
            
            embeddings = np.array(embeddings)
            
            # Aggregate embeddings based on method
            if aggregation == 'mean':
                if use_weighted:
                    feature_vector = np.average(embeddings, axis=0, weights=weights)
                else:
                    feature_vector = np.mean(embeddings, axis=0)
            elif aggregation == 'max':
                feature_vector = np.max(embeddings, axis=0)
            elif aggregation == 'concat':
                # Concatenate all embeddings (pad or truncate to top_k_paths)
                while len(embeddings) < top_k_paths:
                    embeddings = np.vstack([embeddings, np.zeros((1, self.doc2vec_model.vector_size))])
                embeddings = embeddings[:top_k_paths]
                feature_vector = embeddings.flatten()
            elif aggregation == 'enhanced':
                # Enhanced: weighted mean + max + std + weighted first/last
                if use_weighted:
                    weighted_mean = np.average(embeddings, axis=0, weights=weights)
                else:
                    weighted_mean = np.mean(embeddings, axis=0)
                max_vec = np.max(embeddings, axis=0)
                std_vec = np.std(embeddings, axis=0)
                min_vec = np.min(embeddings, axis=0)  # Add min for more information
                
                # Add first and last if we have multiple embeddings
                if len(embeddings) >= 2:
                    # Weight first and last by their weights
                    first_vec = embeddings[0] * weights[0] if use_weighted else embeddings[0]
                    last_vec = embeddings[-1] * weights[-1] if use_weighted else embeddings[-1]
                    feature_vector = np.concatenate([weighted_mean, max_vec, min_vec, std_vec, first_vec, last_vec])
                else:
                    # If only one embedding, use mean + max + min + std
                    feature_vector = np.concatenate([weighted_mean, max_vec, min_vec, std_vec])
            else:
                if use_weighted:
                    feature_vector = np.average(embeddings, axis=0, weights=weights)
                else:
                    feature_vector = np.mean(embeddings, axis=0)
        else:
            # If no model, return zero vector
            vec_size = 100
            if aggregation == 'concat' or aggregation == 'enhanced':
                return np.zeros(vec_size * top_k_paths)
            feature_vector = np.zeros(vec_size)
        
        return feature_vector
    
    def save_models(self, frequency_db_path: str, doc2vec_model_path: str):
        """Save frequency database and Doc2Vec model"""
        with open(frequency_db_path, 'wb') as f:
            pickle.dump(self.frequency_db, f)
        
        if self.doc2vec_model:
            self.doc2vec_model.save(doc2vec_model_path)

