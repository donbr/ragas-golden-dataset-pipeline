import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from collections import Counter
import argparse
from kg_utils import load_kg_json, get_kg_stats, setup_common_args, PROCESSED_DIR

# Handle the pyvis import gracefully
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("Warning: pyvis package not available. Interactive visualization will be disabled.")

from ragas.testset.graph import KnowledgeGraph

def load_knowledge_graph(kg_path):
    """Load knowledge graph from JSON file."""
    try:
        # Using RAGAS's KnowledgeGraph.load method
        kg = KnowledgeGraph.load(kg_path)
        print(f"Successfully loaded knowledge graph from {kg_path}")
        return kg
    except Exception as e:
        print(f"Error loading knowledge graph with RAGAS: {e}")
        
        # Fallback to manual JSON parsing
        try:
            data = load_kg_json(kg_path)
            print(f"Loaded knowledge graph as raw JSON")
            return data
        except Exception as e:
            print(f"Error loading knowledge graph as JSON: {e}")
            return None

def analyze_kg_structure(kg):
    """Analyze the basic structure of the knowledge graph."""
    if hasattr(kg, 'nodes') and hasattr(kg, 'relationships'):
        # Using RAGAS KnowledgeGraph object
        nodes = kg.nodes
        relationships = kg.relationships
        
        print(f"\nKnowledge Graph Structure:")
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of relationships: {len(relationships)}")
        
        # Analyze node properties
        if nodes:
            sample_node = nodes[0]
            print(f"\nNode properties: {list(sample_node.properties.keys())}")
            
            # Count property types
            property_counts = Counter()
            for node in nodes:
                for prop in node.properties:
                    property_counts[prop] += 1
            
            print("\nProperty distribution:")
            for prop, count in property_counts.most_common():
                print(f"  - {prop}: {count} nodes ({count/len(nodes)*100:.1f}%)")
        
        # Analyze relationship types
        if relationships:
            rel_types = Counter([rel.type for rel in relationships])
            print("\nRelationship types:")
            for rel_type, count in rel_types.most_common():
                print(f"  - {rel_type}: {count} relationships")
                
            # Sample relationship properties
            sample_rel = relationships[0]
            print(f"\nSample relationship properties: {list(sample_rel.properties.keys())}")
    else:
        # Raw JSON data - use kg_utils for statistics
        stats = get_kg_stats(kg)
        print("\nRaw JSON structure:")
        print(f"Number of nodes: {stats['num_nodes']}")
        print(f"Number of relationships: {stats['num_relationships']}")
        
        # Print relationship types if available
        if 'relationship_types' in stats:
            print("\nRelationship types:")
            for rel_type, count in stats['relationship_types'].items():
                print(f"  - {rel_type}: {count} relationships")

def create_networkx_graph(kg):
    """Convert KG to NetworkX graph for analysis and visualization."""
    G = nx.Graph()
    
    if hasattr(kg, 'nodes') and hasattr(kg, 'relationships'):
        # Add nodes
        for node in kg.nodes:
            node_id = node.id
            G.add_node(node_id, **{k: str(v)[:50] for k, v in node.properties.items()})
            
        # Add edges
        for rel in kg.relationships:
            source_id = rel.source.id
            target_id = rel.target.id
            G.add_edge(source_id, target_id, type=rel.type, 
                       **{k: str(v)[:50] for k, v in rel.properties.items()})
    else:
        # Try to parse raw JSON
        if 'nodes' in kg and 'relationships' in kg:
            # Add nodes
            for node in kg['nodes']:
                if 'id' in node:
                    node_id = node['id']
                    properties = node.get('properties', {})
                    G.add_node(node_id, **{k: str(v)[:50] for k, v in properties.items()})
            
            # Add edges
            for rel in kg['relationships']:
                if 'source' in rel and 'target' in rel:
                    source_id = rel['source']
                    target_id = rel['target']
                    rel_type = rel.get('type', 'unknown')
                    properties = rel.get('properties', {})
                    G.add_edge(source_id, target_id, type=rel_type, 
                               **{k: str(v)[:50] for k, v in properties.items()})
    
    return G

def analyze_graph_metrics(G):
    """Analyze graph metrics using NetworkX."""
    print("\nGraph Metrics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        # Calculate degree statistics
        degrees = [d for _, d in G.degree()]
        avg_degree = sum(degrees) / len(degrees)
        print(f"Average degree: {avg_degree:.2f}")
        
        # Find top nodes by degree
        node_degrees = dict(G.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 nodes by degree:")
        for node_id, degree in top_nodes:
            print(f"  - Node {node_id}: {degree} connections")
        
        # Connected components
        conn_components = list(nx.connected_components(G))
        print(f"\nNumber of connected components: {len(conn_components)}")
        print(f"Largest component size: {len(max(conn_components, key=len))}")
        
        # Try calculating more metrics if the graph is not too large
        if G.number_of_nodes() < 1000:
            try:
                # Centrality metrics for a sample of nodes
                print("\nCentrality metrics (sample):")
                betweenness = nx.betweenness_centrality(G, k=min(5, G.number_of_nodes()))
                top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
                print("Top nodes by betweenness centrality:")
                for node_id, centrality in top_between:
                    print(f"  - Node {node_id}: {centrality:.4f}")
            except Exception as e:
                print(f"Skipping some metrics due to: {e}")

def visualize_graph(G, output_path=None):
    """Visualize the graph using NetworkX and matplotlib."""
    plt.figure(figsize=(12, 10))
    
    # Limit to a reasonable subset for visualization if too large
    if G.number_of_nodes() > 50:
        # Get the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_cc).copy()
        if len(largest_cc) > 50:
            # Further sample to top 50 nodes by degree
            node_degrees = dict(H.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:50]
            top_node_ids = [node_id for node_id, _ in top_nodes]
            H = H.subgraph(top_node_ids).copy()
        print(f"\nVisualization limited to {H.number_of_nodes()} nodes (subset of original graph)")
    else:
        H = G
    
    # Create layout
    try:
        pos = nx.spring_layout(H, seed=42)
        
        # Get edge types for coloring
        edge_types = nx.get_edge_attributes(H, 'type')
        unique_types = set(edge_types.values())
        
        # Fix: Use a default colormap that is definitely available
        cmap = plt.cm.get_cmap('tab10', 10) if hasattr(plt.cm, 'tab10') else plt.cm.get_cmap('viridis', 10)
        edge_colors = {t: cmap(i/10) for i, t in enumerate(unique_types)}
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(H, pos, node_size=100, alpha=0.8)
        
        # Draw edges with different colors by type
        for edge_type in unique_types:
            edge_list = [e for e in H.edges() if edge_types.get(e) == edge_type]
            nx.draw_networkx_edges(H, pos, edgelist=edge_list, alpha=0.5, 
                                  width=1, edge_color=[edge_colors[edge_type]])
        
        # Add node labels (limited to save space)
        if H.number_of_nodes() <= 20:
            nx.draw_networkx_labels(H, pos, font_size=8)
        
        # Add legend for edge types
        plt.legend([plt.Line2D([0], [0], color=edge_colors[t], linewidth=2) for t in unique_types],
                   list(unique_types), title="Relationship Types", loc="upper right")
        
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        
        # Save to file if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {output_path}")
        
        plt.show()
    except Exception as e:
        print(f"Error visualizing graph: {e}")

def extract_common_patterns(kg):
    """Extract common patterns from the knowledge graph."""
    if not (hasattr(kg, 'nodes') and hasattr(kg, 'relationships')):
        print("Cannot extract patterns from raw JSON data")
        return
    
    print("\nExtracting Common Patterns:")
    
    # Entity extraction analysis
    entities = []
    for node in kg.nodes:
        if 'entities' in node.properties:
            node_entities = node.properties['entities']
            if isinstance(node_entities, list):
                entities.extend(node_entities)
            elif isinstance(node_entities, dict):
                for entity_type, values in node_entities.items():
                    if isinstance(values, list):
                        entities.extend([(entity_type, e) for e in values])
    
    if entities:
        entities_counter = Counter(entities)
        print("\nMost common entities:")
        for entity, count in entities_counter.most_common(10):
            print(f"  - {entity}: {count} occurrences")
    
    # Relationship pattern analysis
    if kg.relationships:
        rel_patterns = []
        for rel in kg.relationships:
            if hasattr(rel, 'source') and hasattr(rel, 'target'):
                pattern = rel.type
                rel_patterns.append(pattern)
        
        rel_pattern_counter = Counter(rel_patterns)
        print("\nMost common relationship patterns:")
        for pattern, count in rel_pattern_counter.most_common():
            print(f"  - {pattern}: {count} occurrences")

def create_interactive_visualization(G, output_path):
    """Create an interactive visualization using Pyvis."""
    if not PYVIS_AVAILABLE:
        print("Skipping interactive visualization: pyvis package not available")
        return None
        
    # Create a new graph with string node IDs
    H = nx.Graph()
    
    # Add nodes with string IDs
    for node_id in G.nodes():
        # Convert node_id to string if it's not already
        str_id = str(node_id)
        # Copy node attributes
        H.add_node(str_id, **G.nodes[node_id])
    
    # Add edges with string IDs
    for u, v, data in G.edges(data=True):
        H.add_edge(str(u), str(v), **data)
    
    # Create a Pyvis network
    net = Network(notebook=False, height="750px", width="100%", bgcolor="#222222", font_color="white")
    
    # Configure physics and visualization options
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250)
    net.toggle_hide_edges_on_drag(True)
    net.toggle_physics(True)
    net.show_buttons(filter_=['physics'])
    
    # Convert NetworkX graph to Pyvis
    net.from_nx(H)
    
    # Add custom tooltips with node properties
    for node in net.nodes:
        # Get node data
        node_data = {}
        for key, value in node.items():
            if key not in ['id', 'x', 'y', 'color', 'size', 'label', 'shape', 'font', 'title']:
                # Skip embeddings and very long values
                if 'embedding' not in key.lower() and (not isinstance(value, str) or len(value) < 200):
                    node_data[key] = value

        # Format the tooltip HTML with better styling
        tooltip = "<div style='max-width: 400px; padding: 10px; background-color: #333; color: #fff; border-radius: 5px;'>"
        
        # Add title if available
        if 'title' in node_data:
            tooltip += f"<div style='font-weight: bold; font-size: 14px; margin-bottom: 8px;'>{node_data.get('title', 'No Title')}</div>"
        
        # Add summary if available
        if 'summary' in node_data:
            tooltip += f"<div style='margin-bottom: 10px; font-style: italic;'>{node_data.get('summary', '')}</div>"
            
        # Add horizontal divider
        tooltip += "<hr style='border-color: #555; margin: 8px 0;'>"
        
        # Add other properties in a formatted list
        tooltip += "<table style='width: 100%; border-collapse: collapse;'>"
        for k, v in node_data.items():
            if k not in ['title', 'summary']:
                # Format the value - truncate if it's too long
                if isinstance(v, str) and len(v) > 100:
                    v = v[:100] + "..."
                tooltip += f"<tr><td style='padding: 3px; font-weight: bold; vertical-align: top;'>{k}</td><td style='padding: 3px;'>{v}</td></tr>"
        tooltip += "</table></div>"
        
        # Set the tooltip
        node['title'] = tooltip
        
        # Add label based on summary or title for better readability
        if 'summary' in node_data and len(node_data['summary']) > 10:
            short_summary = node_data['summary'][:30] + "..." if len(node_data['summary']) > 30 else node_data['summary']
            node['label'] = short_summary
        elif 'title' in node_data and node_data['title']:
            node['label'] = node_data['title'][:30] + "..." if len(node_data['title']) > 30 else node_data['title']
    
    # Save the interactive visualization
    net.save_graph(output_path)
    print(f"Interactive visualization saved to {output_path}")
    return output_path

def main():
    # Replace with setup_common_args from kg_utils
    parser = setup_common_args('Analyze RAGAS Knowledge Graph')
    parser.add_argument('--output-dir', type=str, default=os.path.join(PROCESSED_DIR, 'analysis'),
                        help=f'Directory to save analysis outputs (default: {os.path.join(PROCESSED_DIR, "analysis")})')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load knowledge graph
    kg = load_knowledge_graph(args.input)
    if kg is None:
        print("Failed to load knowledge graph. Exiting.")
        return
    
    # Analyze KG structure
    analyze_kg_structure(kg)
    
    # Convert to NetworkX graph for further analysis
    G = create_networkx_graph(kg)
    
    # Analyze graph metrics
    analyze_graph_metrics(G)
    
    # Extract common patterns
    extract_common_patterns(kg)
    
    # Visualize graph
    viz_path = output_dir / "knowledge_graph.png"
    visualize_graph(G, str(viz_path))
    
    # Create interactive visualization if pyvis is available
    if PYVIS_AVAILABLE:
        interactive_viz_path = output_dir / "interactive_graph.html"
        create_interactive_visualization(G, str(interactive_viz_path))
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}.")

if __name__ == "__main__":
    main() 