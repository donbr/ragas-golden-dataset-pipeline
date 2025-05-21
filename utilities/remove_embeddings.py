import sys
import os
from kg_utils import load_kg_json, save_kg_json, remove_embedding_fields, setup_common_args, get_default_output_filename, KG_OUTPUT_PATH, PROCESSED_DIR

def remove_summary_embeddings(input_file, output_file=None):
    """
    Removes all fields containing 'embedding' in their name from a JSON file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str, optional): Path to save the modified JSON.
                                    If None, will use '{input_file}_no_embeddings.json'
    """
    if output_file is None:
        output_file = get_default_output_filename(input_file, "no_embeddings")
    
    # Load the JSON data
    data = load_kg_json(input_file)
    
    print("Removing embedding fields...")
    
    # Remove embedding fields and get count
    modified_data, total_fields_removed = remove_embedding_fields(data)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Save the modified data
    save_kg_json(modified_data, output_file)
    
    print(f"Done! Removed {total_fields_removed} embedding fields.")
    return total_fields_removed

def main():
    parser = setup_common_args("Remove embedding fields from knowledge graph JSON files")
    args = parser.parse_args()
    
    try:
        total_removed = remove_summary_embeddings(args.input, args.output)
        print(f"Successfully processed file. Removed {total_removed} embedding fields.")
        
        # Determine the output file for the message
        output_file = args.output or get_default_output_filename(args.input, "no_embeddings")
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 