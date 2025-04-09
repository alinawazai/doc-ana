import faiss
import numpy as np
import os

# Function to load the FAISS index from a file
def load_faiss_index(faiss_file_path):
    try:
        faiss_index = faiss.read_index(faiss_file_path)
        print("FAISS index loaded successfully.")
        return faiss_index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

# Function to test FAISS retrieval
def test_faiss_retrieval(faiss_index, query_vector, k=5):
    try:
        # Perform a search with the query vector
        distances, indices = faiss_index.search(np.array([query_vector]).astype('float32'), k)
        print(f"Distances: {distances}")
        print(f"Indices: {indices}")
        return distances, indices
    except Exception as e:
        print(f"Error performing FAISS search: {e}")
        return None, None

# Test FAISS Index
if __name__ == "__main__":
    faiss_file_path = "vector_db.faiss"  # Replace with your FAISS file path

    # Check if the file exists
    if not os.path.exists(faiss_file_path):
        print(f"FAISS index file does not exist: {faiss_file_path}")
    else:
        # Load the FAISS index
        faiss_index = load_faiss_index(faiss_file_path)
        
        if faiss_index:
            # Create a dummy query vector (use an actual query vector that matches your index's dimensionality)
            query_vector = np.random.random(128).tolist()  # Assuming your index has 128 dimensions

            # Test retrieval
            distances, indices = test_faiss_retrieval(faiss_index, query_vector)

            # Check if results were found
            if distances is not None and indices is not None:
                print("Search completed. Results:")
                print(f"Distances: {distances}")
                print(f"Indices: {indices}")
            else:
                print("No results found or error during search.")
        else:
            print("Failed to load FAISS index.")
