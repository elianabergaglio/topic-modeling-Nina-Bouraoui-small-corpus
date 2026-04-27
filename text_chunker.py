
import os
import re
import glob
import pandas as pd

from utils import extract_metadata_from_filename

class TextChunker:
    """
    Class for processing text files into sentence-level chunks with metadata.
    """
    
    def __init__(self, min_chunk_size: int, method: str = "default", min_tokens: int = 3, max_chunk_size: int = 400, openai_key=None):
        """
        Initialize the TextChunker.
        
        Args:
            min_chunk_size: Minimum size for text chunks (default: 100 characters)
            method: Chunking method - "default" for punctuation-based 
            min_tokens: Minimum number of tokens per chunk, chunks with fewer tokens are discarded (default: 3)
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.method = method
        self.min_tokens = min_tokens
        self.openai_key = openai_key
        self.chunks_data = []

    def chunk_text_by_punctuation(self, text: str) -> list[str]:
        # Split on ., ?, ;, » while keeping the delimiter
        parts = re.split(r'([.?;!»])', text)
        
        sentences = []
        current = ""

        for part in parts:
            current += part
            if part in ".?;!»":
                if not sentences:
                    sentences.append(current.strip())
                elif len(sentences[-1]) < self.min_chunk_size:
                    sentences[-1] += " " + current.strip()
                else:
                    sentences.append(current.strip())
                current = ""

        # Handle any trailing text without punctuation
        if current.strip():
            if sentences and len(sentences[-1]) < self.min_chunk_size:
                sentences[-1] += " " + current.strip()
            else:
                sentences.append(current.strip())

        return sentences


    def chunk_text_by_dot(self, text: str) -> list[str]:
        dot_split = text.split(".")
        sentences = []
        for i, chunk in enumerate(dot_split):
            if len(sentences) == 0:
                sentences.append(chunk)
            else:
                if len(sentences[-1]) < self.min_chunk_size:
                    sentences[-1] += "."+chunk
                else:
                    sentences.append(chunk)
        sentences = [s[1:] if s.startswith(" ") else s for s in sentences]    
        return sentences
    

    
    def process_file(self, file_path: str) -> list[dict]:
        """
        Process a single text file into chunks.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of chunk dictionaries
        """
        filename = os.path.basename(file_path)
        work_name, year = extract_metadata_from_filename(filename)
        

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip().replace('\n', ' ').replace('\r', ' ')
        
        if not text:
            print(f"Warning: Empty file {filename}")
            return []
        
        # Create chunks using the selected method
        if self.method == "default":
            chunks = self.chunk_text_by_punctuation(text)
        elif self.method == "dot":
            chunks = self.chunk_text_by_dot(text)        
        else:
            raise Exception(f"Method {self.method} not supported")

        file_chunks = []
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():  # Skip empty chunks
                num_tokens = len(chunk_text.split()) # white-space separation
                if num_tokens >= self.min_tokens:  # Only keep chunks with minimum token count
                    chunk_data = {
                        'chunk_id': f"{work_name}_{year}_chunk_{i+1:04d}",
                        'chunk_text': chunk_text.strip(),
                        'num_tokens': num_tokens,
                        'num_characters': len(chunk_text),
                        'year': year,
                        'work_name': work_name
                    }
                    file_chunks.append(chunk_data)
        
        return file_chunks
        
    
    def process_directory(self, input_dir: str) -> pd.DataFrame:
        """
        Process all .txt files in a directory.
        
        Args:
            input_dir: Directory containing text files
            
        Returns:
            pandas DataFrame with chunk data
        """
        if not os.path.isdir(input_dir):
            raise ValueError(f"Directory {input_dir} does not exist")
        
        # Find all .txt files
        txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
        
        if not txt_files:
            print(f"No .txt files found in {input_dir}")
            return pd.DataFrame()
        
        print(f"Found {len(txt_files)} text files to process...")
        
        all_chunks = []
        processed_files = 0
        
        for file_path in txt_files:
            filename = os.path.basename(file_path)
            print(f"Processing: {filename}")
            
            file_chunks = self.process_file(file_path)
            all_chunks.extend(file_chunks)
            
            if file_chunks:
                processed_files += 1
                print(f"  Created {len(file_chunks)} chunks")
        
        print(f"\nCompleted! Processed {processed_files} files, created {len(all_chunks)} total chunks.")
        
        # Create DataFrame
        if all_chunks:
            df = pd.DataFrame(all_chunks)
            # Reorder columns to match requirements
            df = df[['chunk_id', 'chunk_text', 'num_tokens', 'num_characters', 'year', 'work_name']]
            return df
        else:
            return pd.DataFrame()
