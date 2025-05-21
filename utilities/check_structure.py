import sys
from kg_utils import load_kg_json, find_embedding_fields, setup_common_args, KG_OUTPUT_PATH

def check_structure(json_file):
    """
    Analyze the structure of a knowledge graph JSON file.
    
    Args:
        json_file: Path to the knowledge graph JSON file
    """
    # Load the knowledge graph
    data = load_kg_json(json_file)
    
    # Print top-level keys
    print(f"Top-level keys: {list(data.keys())}")
    
    # Check nodes
    if 'nodes' in data:
        if data['nodes']:
            node = data['nodes'][0]  # Take first node
            print(f"\nSample node keys: {list(node.keys())}")
            
            # Check properties
            if 'properties' in node:
                print(f"Node properties keys: {list(node['properties'].keys())}")
                
                # Check summary
                if 'summary' in node['properties']:
                    summary = node['properties']['summary']
                    if isinstance(summary, dict):
                        print(f"Summary is a dictionary with keys: {list(summary.keys())}")
                        
                        # Check for embedding fields
                        for key in summary.keys():
                            if 'embedding' in key.lower():
                                print(f"Found embedding field: {key}")
                    else:
                        print(f"Summary is not a dictionary, type: {type(summary)}")
        else:
            print("\nNo nodes found in the knowledge graph.")
    
    # Look for any keys containing 'embedding' anywhere in the structure
    print("\nSearching for embedding fields...")
    embedding_fields = find_embedding_fields(data)
    
    if embedding_fields:
        print(f"Found {len(embedding_fields)} embedding fields:")
        for field in embedding_fields:
            print(f"  - {field}")
    else:
        print("No embedding fields found.")

def main():
    parser = setup_common_args("Analyze the structure of a knowledge graph JSON file")
    args = parser.parse_args()
    
    try:
        check_structure(args.input)
    except Exception as e:
        print(f"Error analyzing file structure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 