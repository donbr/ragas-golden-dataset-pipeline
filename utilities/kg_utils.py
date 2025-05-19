"""
Common utility functions for RAGAS knowledge graph operations.

This module provides shared functionality used across multiple scripts in the utilities directory,
reducing code duplication and improving maintainability.
"""

import json
import os
import sys
from typing import Dict, Any, List, Optional, Tuple, Union
import argparse

def load_kg_json(input_file: str) -> Dict[str, Any]:
    """
    Load a knowledge graph from a JSON file with proper error handling.
    
    Args:
        input_file: Path to the JSON file
        
    Returns:
        Dict containing the parsed JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        print(f"Loading JSON from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}': {e}")
        raise
    except Exception as e:
        print(f"Error loading file: {e}")
        raise

def save_kg_json(data: Dict[str, Any], output_file: str) -> None:
    """
    Save knowledge graph data to a JSON file.
    
    Args:
        data: Dict containing the knowledge graph data
        output_file: Path where the JSON file will be saved
    """
    try:
        print(f"Saving JSON to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        raise

def find_embedding_fields(obj: Any, path: str = "") -> List[str]:
    """
    Recursively find all fields containing 'embedding' in their name.
    
    Args:
        obj: The object to search (dict, list, or other value)
        path: Current path in the object (for recursive calls)
        
    Returns:
        List of paths to fields containing 'embedding'
    """
    embedding_fields = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if 'embedding' in key.lower():
                embedding_fields.append(new_path)
            embedding_fields.extend(find_embedding_fields(value, new_path))
    elif isinstance(obj, list) and len(obj) > 0:
        # Only check the first item in the list to avoid excessive output
        embedding_fields.extend(find_embedding_fields(obj[0], f"{path}[0]"))
    
    return embedding_fields

def remove_embedding_fields(data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Remove all fields containing 'embedding' in their name from a knowledge graph.
    
    Args:
        data: Dict containing the knowledge graph data
        
    Returns:
        Tuple containing:
        - Modified data with embedding fields removed
        - Count of removed fields
    """
    total_fields_removed = 0
    
    # Process nodes
    if 'nodes' in data:
        for node in data['nodes']:
            if 'properties' in node:
                # Find keys containing 'embedding' in the properties
                keys_to_remove = [key for key in node['properties'].keys() if 'embedding' in key.lower()]
                
                for key in keys_to_remove:
                    del node['properties'][key]
                    total_fields_removed += 1
    
    # Process relationships (if needed)
    if 'relationships' in data:
        for rel in data['relationships']:
            if 'properties' in rel:
                # Find keys containing 'embedding' in the properties
                keys_to_remove = [key for key in rel['properties'].keys() if 'embedding' in key.lower()]
                
                for key in keys_to_remove:
                    del rel['properties'][key]
                    total_fields_removed += 1
    
    return data, total_fields_removed

def get_kg_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get basic statistics about a knowledge graph.
    
    Args:
        data: Dict containing the knowledge graph data
        
    Returns:
        Dict containing various statistics about the graph
    """
    stats = {
        'num_nodes': len(data.get('nodes', [])),
        'num_relationships': len(data.get('relationships', [])),
    }
    
    # Calculate node-to-relationship ratio if possible
    if stats['num_relationships'] > 0:
        stats['node_to_rel_ratio'] = stats['num_nodes'] / stats['num_relationships']
    else:
        stats['node_to_rel_ratio'] = float('inf')
    
    # Count node types if available
    node_types = {}
    for node in data.get('nodes', []):
        if 'type' in node:
            node_type = node['type']
            node_types[node_type] = node_types.get(node_type, 0) + 1
    
    if node_types:
        stats['node_types'] = node_types
    
    # Count relationship types
    rel_types = {}
    for rel in data.get('relationships', []):
        if 'type' in rel:
            rel_type = rel['type']
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
    
    if rel_types:
        stats['relationship_types'] = rel_types
    
    return stats

def setup_common_args(description: str) -> argparse.ArgumentParser:
    """
    Set up common command-line arguments used across multiple scripts.
    
    Args:
        description: Description of the script
        
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', '-i', type=str, default='output/kg.json',
                      help='Path to the input knowledge graph JSON file')
    parser.add_argument('--output', '-o', type=str, 
                      help='Path to save the output file (default: based on input filename)')
    return parser

def get_default_output_filename(input_file: str, suffix: str) -> str:
    """
    Generate a default output filename based on the input filename.
    
    Args:
        input_file: Original input filename
        suffix: Suffix to add before the file extension
        
    Returns:
        Generated output filename
    """
    parts = input_file.rsplit('.', 1)
    return f"{parts[0]}_{suffix}.{parts[1]}" if len(parts) > 1 else f"{input_file}_{suffix}" 