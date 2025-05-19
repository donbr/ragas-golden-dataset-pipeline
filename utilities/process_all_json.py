import os
import sys
import glob
import argparse
from kg_utils import get_default_output_filename, load_kg_json, save_kg_json, remove_embedding_fields

def process_directory(directory_path, pattern="*.json"):
    """
    Process all JSON files in a directory that match the pattern.
    
    Args:
        directory_path (str): Directory containing JSON files
        pattern (str): Glob pattern to match files (default: "*.json")
    """
    # Find all JSON files
    json_files = glob.glob(os.path.join(directory_path, pattern))
    
    if not json_files:
        print(f"No JSON files found in {directory_path} matching pattern {pattern}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    total_processed = 0
    total_fields_removed = 0
    
    # Process each file
    for json_file in json_files:
        # Skip files that already have "_no_embeddings" in the name
        if "_no_embeddings" in json_file:
            print(f"Skipping {json_file} (already processed)")
            continue
        
        print(f"\nProcessing file: {json_file}")
        
        try:
            # Load the file
            data = load_kg_json(json_file)
            
            # Remove embedding fields
            modified_data, fields_removed = remove_embedding_fields(data)
            
            # Create output filename
            output_file = get_default_output_filename(json_file, "no_embeddings")
            
            # Save modified data
            save_kg_json(modified_data, output_file)
            
            total_processed += 1
            total_fields_removed += fields_removed
            
            # Calculate file size reduction
            original_size = os.path.getsize(json_file)
            if os.path.exists(output_file):
                new_size = os.path.getsize(output_file)
                reduction = original_size - new_size
                percentage = (reduction / original_size) * 100
                print(f"File size: {original_size/1024/1024:.2f} MB → {new_size/1024/1024:.2f} MB")
                print(f"Reduction: {reduction/1024/1024:.2f} MB ({percentage:.2f}%)")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Files processed: {total_processed}")
    print(f"  Embedding fields removed: {total_fields_removed}")

def main():
    parser = argparse.ArgumentParser(description="Process all JSON files in a directory to remove embedding fields")
    parser.add_argument('directory', type=str, help="Directory containing JSON files to process")
    parser.add_argument('--pattern', type=str, default="*.json", help="Glob pattern for files (default: *.json)")
    
    args = parser.parse_args()
    
    process_directory(args.directory, args.pattern)

if __name__ == "__main__":
    main() 