#!/usr/bin/env python
"""
RAGAS Knowledge Graph Visualization Wrapper

This script is a lightweight wrapper around the more comprehensive analyze_kg.py script,
providing a simple interface for visualizing knowledge graphs.
"""

import sys
import os
import argparse
from dotenv import load_dotenv
from kg_utils import load_kg_json, KG_OUTPUT_PATH, PROCESSED_DIR

# Ensure analyze_kg.py is available
try:
    from analyze_kg import create_networkx_graph, visualize_graph, create_interactive_visualization
except ImportError:
    print("Error: This script requires analyze_kg.py to be in the same directory.")
    sys.exit(1)

# Load environment variables
load_dotenv()

def visualize_kg(input_file, output_file=None, interactive=True):
    """
    Visualize a RAGAS knowledge graph.
    
    Args:
        input_file (str): Path to the knowledge graph JSON file
        output_file (str, optional): Output path for the visualization
        interactive (bool): Whether to create an interactive visualization
    """
    # Set default output filenames
    if output_file is None:
        output_dir = os.path.join(PROCESSED_DIR, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        if interactive:
            output_file = os.path.join(output_dir, f"{base_name}_visualization.html")
        else:
            output_file = os.path.join(output_dir, f"{base_name}_visualization.png")
    
    # Load the knowledge graph
    print(f"Loading knowledge graph from {input_file}...")
    data = load_kg_json(input_file)
    
    # Print basic statistics
    num_nodes = len(data.get('nodes', []))
    num_relationships = len(data.get('relationships', []))
    print(f"Knowledge Graph Statistics:")
    print(f"- Number of nodes: {num_nodes}")
    print(f"- Number of relationships: {num_relationships}")
    
    # Convert to NetworkX graph
    G = create_networkx_graph(data)
    
    # Create visualization
    if interactive:
        result_path = create_interactive_visualization(G, output_file)
        if result_path:
            print(f"Interactive visualization saved to {os.path.abspath(result_path)}")
        else:
            print("Interactive visualization failed. Falling back to static visualization.")
            visualize_graph(G, output_file.replace(".html", ".png"))
    else:
        visualize_graph(G, output_file)
        print(f"Static visualization saved to {os.path.abspath(output_file)}")
    
    print("\nFor more detailed analysis, use: python analyze_kg.py --kg-path", input_file)

def main():
    parser = argparse.ArgumentParser(description="Visualize a RAGAS knowledge graph")
    parser.add_argument('--input', '-i', type=str, default=KG_OUTPUT_PATH,
                      help=f'Path to the knowledge graph JSON file (default: {KG_OUTPUT_PATH})')
    parser.add_argument('--output', '-o', type=str, default=None,
                      help='Path to save the visualization output')
    parser.add_argument('--static', action='store_true',
                      help='Generate a static PNG image instead of an interactive HTML visualization')
    
    args = parser.parse_args()
    
    visualize_kg(args.input, args.output, not args.static)

if __name__ == "__main__":
    main()
