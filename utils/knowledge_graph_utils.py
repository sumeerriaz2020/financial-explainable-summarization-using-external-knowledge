"""
Knowledge Graph Query and Traversal Utilities
==============================================

Utilities for querying, traversing, and analyzing knowledge graphs
including path finding, subgraph extraction, and graph algorithms.

Reference: Algorithm 2, Lines 6-8 (multi-hop reasoning)
"""

import networkx as nx
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Path:
    """Path in knowledge graph"""
    nodes: List[str]
    edges: List[Tuple[str, str, Dict]]
    length: int
    confidence: float


class KnowledgeGraphUtils:
    """Utilities for knowledge graph operations"""
    
    def __init__(self, graph: Optional[nx.DiGraph] = None):
        """
        Initialize KG utilities
        
        Args:
            graph: NetworkX directed graph (optional)
        """
        self.graph = graph or nx.DiGraph()
        logger.info(f"KG Utils initialized with {self.graph.number_of_nodes()} nodes")
    
    def add_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        **attributes
    ):
        """
        Add triple to knowledge graph
        
        Args:
            subject: Subject node
            predicate: Relationship type
            object: Object node
            confidence: Confidence score
            **attributes: Additional edge attributes
        """
        self.graph.add_edge(
            subject,
            object,
            relation=predicate,
            confidence=confidence,
            **attributes
        )
    
    def get_neighbors(
        self,
        node: str,
        direction: str = 'out',
        relation_type: Optional[str] = None
    ) -> List[str]:
        """
        Get neighbors of a node
        
        Args:
            node: Source node
            direction: 'out' (successors) or 'in' (predecessors) or 'both'
            relation_type: Optional filter by relation type
            
        Returns:
            List of neighbor nodes
        """
        if node not in self.graph:
            return []
        
        if direction == 'out':
            neighbors = list(self.graph.successors(node))
        elif direction == 'in':
            neighbors = list(self.graph.predecessors(node))
        else:  # both
            neighbors = list(self.graph.successors(node)) + \
                       list(self.graph.predecessors(node))
        
        # Filter by relation type if specified
        if relation_type:
            filtered = []
            for neighbor in neighbors:
                if direction in ['out', 'both']:
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    if edge_data and edge_data.get('relation') == relation_type:
                        filtered.append(neighbor)
                if direction in ['in', 'both']:
                    edge_data = self.graph.get_edge_data(neighbor, node)
                    if edge_data and edge_data.get('relation') == relation_type:
                        filtered.append(neighbor)
            neighbors = filtered
        
        return neighbors
    
    def find_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
        max_paths: int = 10
    ) -> List[Path]:
        """
        Find paths between two nodes
        
        Args:
            source: Source node
            target: Target node
            max_hops: Maximum path length
            max_paths: Maximum number of paths to return
            
        Returns:
            List of paths
        """
        if source not in self.graph or target not in self.graph:
            return []
        
        try:
            # Find all simple paths
            all_paths = nx.all_simple_paths(
                self.graph,
                source,
                target,
                cutoff=max_hops
            )
            
            paths = []
            for i, node_path in enumerate(all_paths):
                if i >= max_paths:
                    break
                
                # Extract edges
                edges = []
                for j in range(len(node_path) - 1):
                    u, v = node_path[j], node_path[j+1]
                    edge_data = self.graph.get_edge_data(u, v)
                    edges.append((u, v, edge_data or {}))
                
                # Compute path confidence (product of edge confidences)
                confidence = 1.0
                for _, _, data in edges:
                    confidence *= data.get('confidence', 1.0)
                
                path = Path(
                    nodes=node_path,
                    edges=edges,
                    length=len(node_path) - 1,
                    confidence=confidence
                )
                paths.append(path)
            
            # Sort by confidence
            paths.sort(key=lambda p: p.confidence, reverse=True)
            
            return paths
        
        except nx.NetworkXNoPath:
            return []
    
    def multi_hop_reasoning(
        self,
        start_nodes: List[str],
        num_hops: int = 3,
        relation_types: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Perform multi-hop reasoning from start nodes
        
        Implements multi-hop traversal from Algorithm 2, Lines 6-8
        
        Args:
            start_nodes: Starting nodes
            num_hops: Number of hops
            relation_types: Optional filter by relation types
            
        Returns:
            Dict mapping hop number to reachable nodes
        """
        reachable = {0: set(start_nodes)}
        
        for hop in range(1, num_hops + 1):
            current_level = set()
            
            for node in reachable[hop - 1]:
                # Get neighbors
                neighbors = self.get_neighbors(node, direction='out')
                
                # Filter by relation type
                if relation_types:
                    filtered_neighbors = []
                    for neighbor in neighbors:
                        edge_data = self.graph.get_edge_data(node, neighbor)
                        if edge_data and edge_data.get('relation') in relation_types:
                            filtered_neighbors.append(neighbor)
                    neighbors = filtered_neighbors
                
                current_level.update(neighbors)
            
            reachable[hop] = current_level
        
        # Convert to lists
        return {hop: list(nodes) for hop, nodes in reachable.items()}
    
    def extract_subgraph(
        self,
        nodes: List[str],
        k_hop: int = 1,
        max_nodes: int = 100
    ) -> nx.DiGraph:
        """
        Extract k-hop subgraph around nodes
        
        Args:
            nodes: Center nodes
            k_hop: Number of hops to include
            max_nodes: Maximum nodes in subgraph
            
        Returns:
            Subgraph
        """
        # Start with center nodes
        subgraph_nodes = set(nodes)
        
        # Add k-hop neighbors
        for _ in range(k_hop):
            new_nodes = set()
            for node in list(subgraph_nodes):
                if node in self.graph:
                    new_nodes.update(self.graph.successors(node))
                    new_nodes.update(self.graph.predecessors(node))
            
            subgraph_nodes.update(new_nodes)
            
            # Limit size
            if len(subgraph_nodes) > max_nodes:
                break
        
        # Extract subgraph
        subgraph = self.graph.subgraph(list(subgraph_nodes)[:max_nodes]).copy()
        
        logger.info(f"Extracted subgraph: {subgraph.number_of_nodes()} nodes, "
                   f"{subgraph.number_of_edges()} edges")
        
        return subgraph
    
    def compute_centrality(
        self,
        metric: str = 'degree'
    ) -> Dict[str, float]:
        """
        Compute node centrality
        
        Args:
            metric: Centrality metric ('degree', 'betweenness', 'pagerank')
            
        Returns:
            Dict mapping nodes to centrality scores
        """
        if metric == 'degree':
            return dict(self.graph.degree())
        elif metric == 'betweenness':
            return nx.betweenness_centrality(self.graph)
        elif metric == 'pagerank':
            return nx.pagerank(self.graph)
        else:
            raise ValueError(f"Unknown centrality metric: {metric}")
    
    def find_communities(self) -> List[Set[str]]:
        """
        Detect communities in graph
        
        Returns:
            List of community node sets
        """
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        # Use Louvain algorithm
        from networkx.algorithms.community import louvain_communities
        
        communities = louvain_communities(undirected)
        
        logger.info(f"Found {len(communities)} communities")
        return communities
    
    def get_graph_statistics(self) -> Dict:
        """Get graph statistics"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph)
        }
        
        # Average degree
        if stats['num_nodes'] > 0:
            stats['avg_degree'] = stats['num_edges'] / stats['num_nodes']
        else:
            stats['avg_degree'] = 0
        
        # Clustering coefficient (for undirected)
        try:
            stats['clustering_coefficient'] = nx.average_clustering(
                self.graph.to_undirected()
            )
        except:
            stats['clustering_coefficient'] = 0.0
        
        return stats


class KnowledgeGraphQuery:
    """SPARQL-like query interface for knowledge graph"""
    
    def __init__(self, kg_utils: KnowledgeGraphUtils):
        """
        Initialize query interface
        
        Args:
            kg_utils: KnowledgeGraphUtils instance
        """
        self.kg = kg_utils
    
    def select(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Query triples matching pattern
        
        Args:
            subject: Subject pattern (None for wildcard)
            predicate: Predicate pattern (None for wildcard)
            object: Object pattern (None for wildcard)
            
        Returns:
            List of matching (subject, predicate, object) triples
        """
        results = []
        
        for u, v, data in self.kg.graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            
            # Check pattern match
            if subject is not None and u != subject:
                continue
            if predicate is not None and relation != predicate:
                continue
            if object is not None and v != object:
                continue
            
            results.append((u, relation, v))
        
        return results
    
    def count(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None
    ) -> int:
        """Count matching triples"""
        return len(self.select(subject, predicate, object))
    
    def distinct(
        self,
        variable: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None
    ) -> Set[str]:
        """
        Get distinct values for a variable
        
        Args:
            variable: 'subject', 'predicate', or 'object'
            subject, predicate, object: Pattern constraints
            
        Returns:
            Set of distinct values
        """
        triples = self.select(subject, predicate, object)
        
        if variable == 'subject':
            return {t[0] for t in triples}
        elif variable == 'predicate':
            return {t[1] for t in triples}
        elif variable == 'object':
            return {t[2] for t in triples}
        else:
            raise ValueError(f"Unknown variable: {variable}")


# Example usage
if __name__ == "__main__":
    print("Knowledge Graph Utilities")
    print("=" * 70)
    
    # Create sample knowledge graph
    print("\nBuilding sample knowledge graph...")
    kg = KnowledgeGraphUtils()
    
    # Add triples (financial entities and relationships)
    kg.add_triple("Apple", "hasProduct", "iPhone", confidence=0.95)
    kg.add_triple("Apple", "hasCEO", "Tim Cook", confidence=1.0)
    kg.add_triple("Apple", "hasRevenue", "Revenue_Q4", confidence=0.90)
    kg.add_triple("Revenue_Q4", "hasValue", "$89.5B", confidence=0.95)
    kg.add_triple("iPhone", "contributesTo", "Revenue_Q4", confidence=0.85)
    kg.add_triple("Tim Cook", "announced", "Earnings_Report", confidence=0.90)
    kg.add_triple("Earnings_Report", "mentions", "Revenue_Q4", confidence=0.95)
    kg.add_triple("Apple", "competesWith", "Samsung", confidence=0.80)
    kg.add_triple("Samsung", "hasProduct", "Galaxy", confidence=0.90)
    
    print(f"Graph created: {kg.graph.number_of_nodes()} nodes, "
          f"{kg.graph.number_of_edges()} edges")
    
    # Test neighbor finding
    print(f"\n{'=' * 70}")
    print("NEIGHBOR FINDING:")
    print(f"{'=' * 70}")
    
    neighbors = kg.get_neighbors("Apple", direction='out')
    print(f"\nApple's outgoing neighbors: {neighbors}")
    
    # Test path finding
    print(f"\n{'=' * 70}")
    print("PATH FINDING:")
    print(f"{'=' * 70}")
    
    paths = kg.find_paths("Apple", "$89.5B", max_hops=3)
    print(f"\nPaths from Apple to $89.5B: {len(paths)} found")
    
    for i, path in enumerate(paths[:3], 1):
        print(f"\nPath {i} (confidence: {path.confidence:.2f}):")
        print(f"  {' -> '.join(path.nodes)}")
    
    # Test multi-hop reasoning
    print(f"\n{'=' * 70}")
    print("MULTI-HOP REASONING:")
    print(f"{'=' * 70}")
    
    reachable = kg.multi_hop_reasoning(["Apple"], num_hops=3)
    
    for hop, nodes in reachable.items():
        print(f"\nHop {hop}: {len(nodes)} nodes")
        if nodes:
            print(f"  {nodes[:5]}")  # Show first 5
    
    # Test subgraph extraction
    print(f"\n{'=' * 70}")
    print("SUBGRAPH EXTRACTION:")
    print(f"{'=' * 70}")
    
    subgraph = kg.extract_subgraph(["Apple", "Revenue_Q4"], k_hop=1)
    print(f"\nExtracted subgraph: {subgraph.number_of_nodes()} nodes, "
          f"{subgraph.number_of_edges()} edges")
    
    # Test centrality
    print(f"\n{'=' * 70}")
    print("CENTRALITY ANALYSIS:")
    print(f"{'=' * 70}")
    
    centrality = kg.compute_centrality(metric='degree')
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 nodes by degree centrality:")
    for node, score in sorted_nodes[:5]:
        print(f"  {node}: {score}")
    
    # Test query interface
    print(f"\n{'=' * 70}")
    print("QUERY INTERFACE:")
    print(f"{'=' * 70}")
    
    query = KnowledgeGraphQuery(kg)
    
    # Find all products
    products = query.distinct('object', predicate='hasProduct')
    print(f"\nAll products: {products}")
    
    # Count relationships
    count = query.count(subject="Apple")
    print(f"Apple has {count} direct relationships")
    
    # Statistics
    stats = kg.get_graph_statistics()
    print(f"\n{'=' * 70}")
    print("GRAPH STATISTICS:")
    print(f"{'=' * 70}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n{'=' * 70}")
    print("KG utilities ready!")
    print(f"{'=' * 70}")
