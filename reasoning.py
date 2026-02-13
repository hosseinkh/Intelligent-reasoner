
from typing import Dict, List, Optional,Any
from rag_store import similar
from contracts import Hit,LLMAnswer
#import ollama
import vertexai
from vertexai.generative_models import GenerativeModel
import json
from api.config import MODEL,MAX_TRIES,LLM_TIMEOUT_S
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import httpx
from pydantic import ValidationError
import grpc
import logging




def rag_search(question: str, k: int = 3, where: Optional[Dict[str, Any]] = None):
    hits = similar(question, k , where)
    #print(f"the identified hits are {hits}")
    return hits

def make_blocks(hits: List[Hit]):
    blocks = []
    for hit in hits:
        blocks.append(f"""source : {hit.metadata['file_name']}| chunk_id: {hit.id}| | text: {hit.document} """)
    #print(f"The identified blocks are {blocks}")
    return "\n".join(blocks)

def make_prompt(question: str, block: str)-> str:
    return f"""You are an expert in medication,
I want you to answer the question {question}, only based on the following content: {block}. Do not add any information
that is not inferred from the block. If you did not find the solution in the block, you must answer I do not know.
If you find the answer make a citation to the part of the block that the answer is there (by showing its id),
and cite its source.

You MUST answer ONLY with a valid JSON object, with this exact structure:
{{
  "cause": "<one of: Manufacturing, DemandSpike, Quality, Logistics, Unknown>",
  "confidence": <a number between 0 and 1>
  "source" : "<the relevant file name indicated in the content>"
}}

Do NOT add any explanation, text, or formatting outside the JSON.

"""
"""
def call_llm(prompt: str):
    def _call():
        response = ollama.chat(
            #model = "qwen2.5:1.5b-instruct",
            model = MODEL,
            messages = [
                {
                "role": "system",
                "content": (
                    "You are an expert in drug shortages. "
                    "You must answer only using the provided CONTEXT and say 'Unknown' if unsure."
                )
                },
                {
                "role": "user", "content": prompt
                }
            ],
        # timeout = LLM_TIMEOUT_S,
        )
        return response["message"]["content"]

    with ThreadPoolExecutor(max_workers = 1) as executor:
        future = executor.submit(_call)
        try:
            return future.result(timeout=LLM_TIMEOUT_S)
        except FuturesTimeout:
            raise TimeoutError(f"LLM call exceeded {LLM_TIMEOUT_S}s")    
    
"""
logger = logging.getLogger(__name__)
def call_llm(prompt:str):
    try:
        vertexai.init(
            project = "medshortage-agent-v2",
            location = "europe-west4"
        )
        model = GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)   
        return response.text 
    except grpc.RpcError as e:
        logger.exception("Vertex gPRC error: code=%s details=%s", e.code(),e.details())
        raise
    except Exception as e:
        logger.exception("Unexpected error calling vertex %r", e)
        raise

def call_llm_with_validation(prompt: str, call_llm, max_retries: int = 2) -> LLMAnswer:
    raw = call_llm(prompt)

    for attempt in range(max_retries + 1):
        try:
            data = json.loads(raw)
            return LLMAnswer.model_validate(data)
        except TimeoutError:
            raise 
        except json.JSONDecodeError:
            repair = (
                "Your output was not valid JSON. "
                "Return ONLY a JSON object matching the schema.\n\n"
                "SCHEMA:\n"
                '{"cause":"<one of: Manufacturing, DemandSpike, Quality, Logistics, Unknown>",'
                '"confidence":<0..1>,"source":"<file name>"}\n\n'
                "BAD OUTPUT:\n" + raw
            )
            raw = call_llm(repair)

        except ValidationError as e:
            repair = (
                "Your JSON did not match the schema (wrong fields/types/values). "
                "Return ONLY corrected JSON matching the schema.\n\n"
                "ERRORS:\n" + str(e) + "\n\n"
                "BAD JSON:\n" + raw
            )
            raw = call_llm(repair)

    raise ValueError("Failed to get valid JSON after retries")
  

def answer_question(question: str, k:int = 3) -> LLMAnswer:
    rag_similarities = rag_search(question , k)
    block = make_blocks(rag_similarities)
    prompt = make_prompt(question, block)
    #answer = call_llm(prompt)
    answer = call_llm_with_validation(prompt, call_llm, max_retries = MAX_TRIES)
    print(answer)
    return answer

def main():
    answer_question("Quelle est la raison d'une tension d’approvisionnement ")

if __name__ == "__main__":
    main()

