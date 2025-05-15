import fitz  # PyMuPDF
import pymongo
import pytesseract  # OCR for image-based text extraction
from PIL import Image
import io  # Required for image handling
from sentence_transformers import SentenceTransformer

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "medical_data"
COLLECTION_NAME = "try_2"

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Efficient small model

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file while handling text and image-based pages."""
    extracted_data = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")  # Extract text normally
            
            if not text.strip():  # If no text, skip this page
                print(f"Skipping Page {page_num + 1}: No text found.")
                continue  # Move to the next page
            
            extracted_data.append({
                "page": page_num + 1,
                "text": text
            })
        
        print(f"Extracted text from {len(extracted_data)} pages out of {len(doc)} total pages.")
        return extracted_data
    
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

def vectorize_text(extracted_data):
    """Generates embeddings for each extracted page."""
    try:
        vectorized_data = []
        
        for entry in extracted_data:
            text_content = entry["text"].strip()
            if text_content:
                entry["embedding"] = embedding_model.encode(text_content).tolist()
                vectorized_data.append(entry)
                # print(f"Page {entry['page']} vectorized successfully.")
        
        print(f"Vectorized {len(vectorized_data)} pages successfully.")
        return vectorized_data
    
    except Exception as e:
        print(f"Error during vectorization: {e}")
        return []

def store_in_mongodb(data):
    """Stores extracted text and vectors in MongoDB."""
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        if data:
            print(f"Inserting {len(data)} pages into MongoDB...")
            collection.insert_many(data)
            print("Data stored successfully in MongoDB.")
        else:
            print("No valid data found for MongoDB insertion.")
    
    except Exception as e:
        print(f"Error storing in MongoDB: {e}")

if __name__ == "__main__":
    pdf_path = "sample.pdf"  # Change to your PDF file path

    # Step 1: Extract text while skipping empty pages
    extracted_text_data = extract_text_from_pdf(pdf_path)

    # Step 2: Vectorize text
    if extracted_text_data:
        vectorized_data = vectorize_text(extracted_text_data)

        # Step 3: Store in MongoDB
        store_in_mongodb(vectorized_data)
    else:
        print("No text extracted. Skipping vectorization and storage.")



#       BOTH ABOVE CODE RUN SUCCESSFULLY


