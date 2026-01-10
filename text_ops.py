from typing import List, Dict
from contracts import Chunk,DocumentText
import re


def make_chunks(chunks: List[Dict[str,str]], source:str) -> List[Chunk]:
    final_chunks =[]
    for idx ,chunk in enumerate(chunks, start = 1):
        final_chunks.append(Chunk(
            text=chunk["chunk"],
            file_name=chunk["file_name"],
            chunk_id= f"{chunk['file_name']}::chunk_{idx}"  ,
            source=source
        ))

    return final_chunks    
         
        


def text_to_sentence(docText : DocumentText) -> Dict[str, List[str]]:
    text = docText.text.strip()
    if not text:
        return {}
    sentences = re.split(r'(?<=[.!?])\s+', text)
    #print(sentences, "and", [s.strip() for s in sentences if s.strip()])
    return {docText.file_name :[s.strip() for s in sentences if s.strip()]}


def chunk_sentences(metaSentences: Dict[str,List[str]] , file_name: str, chunk_size: int = 500 , overlapping_sentence: int = 1) -> List[Dict[str,str]]:
    chunks =[]
    
    #for file_name , sentences in metaSentences.items():
    current_chunk = []
    current_length = 0
    Sentences = next(iter(metaSentences.values()))
    for sentence in Sentences:
        sentence_length = len(sentence)
        if current_chunk and (current_length + sentence_length) > chunk_size:
            chunk = " ".join(current_chunk)
            chunks.append({"chunk":chunk, "file_name":file_name})
            if overlapping_sentence > 0:
                current_chunk = current_chunk[-overlapping_sentence:]
                current_length = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length = current_length + sentence_length

    if current_chunk:
        chunks.append({"chunk":" ".join(current_chunk), "file_name":file_name})

    return chunks


