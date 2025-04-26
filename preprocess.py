import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from tqdm import tqdm
import time
import gc  # For garbage collection
import sys
import psutil  # For memory tracking

def check_file_access():
    """Check if we can access the data file"""
    file_path = "911_Calls_for_Service.csv"
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")
        
        # Check if file is readable
        with open(file_path, 'r') as f:
            # Read first line to verify access
            f.readline()
        
        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"Found data file: {file_path}")
        print(f"File size: {file_size:.2f} MB")
        return True
        
    except PermissionError:
        print(f"ERROR: No permission to access {file_path}")
        print("Please check file permissions and make sure no other program is using the file")
        return False
    except FileNotFoundError as e:
        print(f"ERROR: {str(e)}")
        print("Please make sure the file is in the correct location")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error accessing file: {str(e)}")
        return False

def load_data(target_rows=100000, chunksize=10000):
    """Load specific number of rows in chunks"""
    print("Loading and processing 911 calls data in chunks...")
    
    # Read only essential columns for analysis
    useful_columns = [
        'recordId',
        'callDateTime',
        'priority',
        'district',
        'description',
        'incidentLocation',
        'location',
        'Neighborhood',
        'PoliceDistrict',
        'ZIPCode'
    ]
    
    try:
        file_path = "911_Calls_for_Service.csv"
        
        # First check if we can access the file
        if not check_file_access():
            raise PermissionError("Cannot access data file")
        
        # Create a chunked iterator
        chunks = pd.read_csv(
            file_path,
            sep='\t',
            usecols=[col.strip() for col in useful_columns],
            parse_dates=['callDateTime'],
            chunksize=chunksize,
            on_bad_lines='warn',  # Don't fail on bad lines
            nrows=target_rows  # Limit to target number of rows
        )
        
        return chunks, target_rows
        
    except pd.errors.EmptyDataError:
        print("ERROR: The CSV file is empty")
        raise
    except pd.errors.ParserError as e:
        print(f"ERROR: Could not parse CSV file: {str(e)}")
        print("Please check if the file is properly formatted")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load data: {str(e)}")
        raise

def process_chunk(chunk):
    """Process a chunk of rows"""
    texts = []
    metadata = []
    
    # Pre-process the chunk data
    chunk = chunk.fillna('')  # Fill NaN values
    
    for _, row in chunk.iterrows():
        try:
            text = f"Emergency Call: "
            text += f"Type: {str(row['description']).strip()}. "
            text += f"Priority: {str(row['priority']).strip()}. "
            text += f"Location: {str(row['incidentLocation']).strip()}, {str(row['location']).strip()}. "
            if row['Neighborhood']:
                text += f"Neighborhood: {str(row['Neighborhood']).strip()}. "
            text += f"District: {str(row['district']).strip()}. "
            text += f"Police District: {str(row['PoliceDistrict']).strip()}. "
            if row['ZIPCode']:
                text += f"ZIP: {str(row['ZIPCode']).strip()}. "
            text += f"Time: {row['callDateTime']}. "
            
            meta = {
                "id": row['recordId'],
                "type": str(row['description']).strip(),
                "location": str(row['location']).strip(),
                "district": str(row['district']).strip(),
                "police_district": str(row['PoliceDistrict']).strip(),
                "timestamp": str(row['callDateTime'])
            }
            
            texts.append(text)
            metadata.append(meta)
            
        except Exception as e:
            print(f"Warning: Skipping row due to error: {e}")
            continue
            
    return texts, metadata

def preprocess_data_for_rag(chunks, total_rows):
    """Process data in chunks to manage memory"""
    print("Preprocessing data...")
    start_time = time.time()
    
    all_texts = []
    all_metadata = []
    processed_rows = 0
    
    # Process chunks with progress bar
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        for chunk in chunks:
            texts, metadata = process_chunk(chunk)
            all_texts.extend(texts)
            all_metadata.extend(metadata)
            
            processed_rows += len(chunk)
            pbar.update(len(chunk))
            
            # Force garbage collection every 5 chunks
            if processed_rows % 50000 == 0:
                gc.collect()
    
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    print(f"Total processed entries: {len(all_texts):,}")
    return all_texts, all_metadata

def create_vector_store(texts):
    print("Creating vector store...")
    start_time = time.time()
    
    # Use a smaller, faster model for embeddings
    embedder = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    
    # Process embeddings in larger batches for 100k dataset
    batch_size = 128  # Increased batch size
    embeddings = []
    
    # Calculate total batches for progress bar
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        try:
            batch_embeddings = embedder.encode(
                batch_texts,
                show_progress_bar=False,
                device='cuda' if embedder.device.type == 'cuda' else 'cpu'
            )
            embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"Error encoding batch {i//batch_size}: {e}")
            continue
        
        # Force garbage collection every 20 batches
        if i % (batch_size * 20) == 0:
            gc.collect()
    
    # Combine all embeddings
    print("Combining embeddings...")
    embeddings = np.vstack(embeddings)
    
    # Create FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors in larger batches
    batch_size = 20000  # Increased for 100k dataset
    for i in range(0, len(embeddings), batch_size):
        index.add(embeddings[i:i + batch_size].astype(np.float32))
        gc.collect()
    
    print(f"Vector store created in {time.time() - start_time:.2f} seconds")
    return index, embedder, embeddings

def ensure_output_dir():
    """Ensure output directory exists and is writable"""
    output_dir = "output"
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Test if directory is writable
        test_file = os.path.join(output_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        
        return output_dir
    except Exception as e:
        print(f"ERROR: Cannot create or write to output directory: {str(e)}")
        raise

def save_vector_store(index, embedder, embeddings, texts, metadata):
    print("Saving vector store and metadata...")
    start_time = time.time()
    
    # Ensure output directory exists
    output_dir = ensure_output_dir()
    
    try:
        # Save FAISS index
        print("Saving FAISS index...")
        index_path = os.path.join(output_dir, "vector_store.index")
        faiss.write_index(index, index_path)
        
        # Save metadata and texts
        print("Saving metadata and texts...")
        data_path = os.path.join(output_dir, "rag_data.pkl")
        with open(data_path, "wb") as f:
            pickle.dump({
                "embedder": embedder,
                "texts": texts,
                "metadata": metadata
            }, f, protocol=4)
        
        # Save embeddings in chunks
        print("Saving embeddings...")
        chunk_size = 25000  # Save embeddings in chunks
        for i in range(0, len(embeddings), chunk_size):
            chunk_path = os.path.join(output_dir, f"embeddings_chunk_{i//chunk_size}.npy")
            np.save(chunk_path, embeddings[i:i + chunk_size])
        
        print(f"Save completed in {time.time() - start_time:.2f} seconds")
        print(f"Files saved in: {output_dir}/")
        
    except Exception as e:
        print(f"ERROR: Failed to save data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        total_start_time = time.time()
        print("Starting preprocessing pipeline...")
        print("Target: Processing 100,000 rows")
        
        # Load data in chunks
        chunks, total_rows = load_data()
        
        # Process all chunks
        texts, metadata = preprocess_data_for_rag(chunks, total_rows)
        
        # Create vector store
        index, embedder, embeddings = create_vector_store(texts)
        
        # Save everything
        save_vector_store(index, embedder, embeddings, texts, metadata)
        
        total_time = time.time() - total_start_time
        print(f"Total preprocessing completed in {total_time:.2f} seconds!")
        
        # Get memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Total memory usage: {memory_mb:.2f} MB")
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"- Total rows processed: {len(texts):,}")
        print(f"- Total embeddings created: {len(embeddings):,}")
        print(f"- Embedding dimensions: {embeddings.shape[1]}")
        print(f"- Average processing speed: {len(texts)/total_time:.2f} rows/second")
        print(f"- Output files saved in: {os.path.abspath('output')}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Processing failed: {str(e)}")
        sys.exit(1)
