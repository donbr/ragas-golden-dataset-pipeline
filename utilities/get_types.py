#!/usr/bin/env python3
"""
Script to extract and print all unique node and relationship types from a RAGAS knowledge graph.
"""

import sys
import os
from kg_utils import load_kg_json, get_kg_stats

def get_unique_types(json_file):
    """
    Extract and print all unique 'type' values in the knowledge graph.
    
    Args:
        json_file: Path to the knowledge graph JSON file
    """
    # Load the knowledge graph
    data = load_kg_json(json_file)
    
    # Get statistics including node types
    stats = get_kg_stats(data)
    
    # Print unique node types
    if 'node_types' in stats:
        print("\nUnique node types:")
        for node_type, count in stats['node_types'].items():
            print(f"  - {node_type}: {count} nodes")
    else:
        print("\nNo node types found in the statistics.")
        
        # Manual approach as fallback
        node_types = {}
        for node in data.get('nodes', []):
            if 'type' in node:
                node_type = node['type']
                node_types[node_type] = node_types.get(node_type, 0) + 1
        
        if node_types:
            print("\nManually extracted unique node types:")
            for node_type, count in node_types.items():
                print(f"  - {node_type}: {count} nodes")
        else:
            print("No 'type' field found in the nodes.")
    
    # Print unique relationship types
    if 'relationship_types' in stats:
        print("\nUnique relationship types:")
        for rel_type, count in stats['relationship_types'].items():
            print(f"  - {rel_type}: {count} relationships")
    else:
        print("\nNo relationship types found in the statistics.")
        
        # Manual approach as fallback
        rel_types = {}
        for rel in data.get('relationships', []):
            if 'type' in rel:
                rel_type = rel['type']
                rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
        
        if rel_types:
            print("\nManually extracted unique relationship types:")
            for rel_type, count in rel_types.items():
                print(f"  - {rel_type}: {count} relationships")
        else:
            print("No 'type' field found in the relationships.")

    # Additional check for types in properties
    print("\nChecking for 'type' fields in node properties...")
    property_types = {}
    for node in data.get('nodes', []):
        if 'properties' in node and isinstance(node['properties'], dict):
            for prop_key, prop_value in node['properties'].items():
                if prop_key == 'type' and isinstance(prop_value, str):
                    property_types[prop_value] = property_types.get(prop_value, 0) + 1
    
    if property_types:
        print("\nUnique 'type' values in node properties:")
        for prop_type, count in property_types.items():
            print(f"  - {prop_type}: {count} occurrences")
    else:
        print("No 'type' field found in node properties.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <knowledge_graph_json_file>")
        print(f"Example: {sys.argv[0]} output/kg_no_embeddings.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)
    
    get_unique_types(json_file) 