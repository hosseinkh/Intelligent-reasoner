from contracts import DocumentText,Chunk
from text_ops import text_to_sentence, chunk_sentences, make_chunks

def test_text_to_sentence_empty_sentence():
    doc = DocumentText(file_name= ".docx", text = "     ")
    assert text_to_sentence(doc) == {}


def test_text_to_sentence_simple_sentence():
    doc = DocumentText(file_name = ".docx" , text = "Hello!   This is Hossein.")
    assert text_to_sentence(doc) == {".docx": ["Hello!", "This is Hossein."]}

def test_chunk_sentences_empty_list():
    arg = {"test.docx": []}
    assert chunk_sentences(arg, "test.docx") == []

def test_chunk_sentences_normal_case():
    arg = {"test.docx":["Hello!", "It's me.", "How can I help you?"]}
    
    chunk =  chunk_sentences(arg, "test.docx") 
    chunk_phrase = next(iter(chunk[0].values()))
    assert chunk_phrase == "Hello! It's me. How can I help you?"

def test_make_chunks_empty_chunk_list():
    arg = []
    assert make_chunks(arg, "MedDoc") == []

def test_make_chunks_normal_case():
    arg = [{"chunk":"Hello, it is me.", "file_name":"test1.docx"},{"chunk": "How can I help you?", "file_name": "test2.docx"}]
    assert make_chunks(arg, "MedDoc") == [Chunk(
            text= "Hello, it is me.",
            file_name="test1.docx",
            chunk_id=  "test1.docx::chunk_1",
            source="MedDoc"
        ),Chunk(
            text="How can I help you?",
            file_name="test2.docx",
            chunk_id= "test2.docx::chunk_2",
            source="MedDoc"
        )]

