
import os
from typing import Dict, List, Optional,Any, Tuple
from rag_store import similar
from contracts import Hit,LLMAnswer
import ollama
import vertexai
from vertexai.generative_models import GenerativeModel
import json
from api.config import (MODEL,
                        MAX_TRIES,
                        LLM_TIMEOUT_S,
                        GOOGLE_CLOUD_PROJECT,
                        VERTEX_LOCATION,
                        VERTEX_MODEL,
                        LLM_PROVIDER,
                        RAG_TOP_K,
                        NUM_CHAR_OUT,
                        TEMPRATURE,
                        CACHE_TTL_S)
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import httpx
from pydantic import ValidationError
import grpc
import logging
import time
import hashlib

from api import metrics


def rag_search(question: str, k: int = RAG_TOP_K, where: Optional[Dict[str, Any]] = None, logger  = None):
    logger = logger or logging.getLogger("intelligent-reasoner")
    t0 = time.perf_counter()
    hits = similar(question, k , where)
    dt_ms = int((time.perf_counter()-t0)*1000)
    hit_counts = len(hits) if hits is not None else 0
    logger.info("rag_search", extra={
        "event": "rag_search",
        "k" : k,
        "where": where,
        "hits_count": hit_counts,
        "duration_ms": dt_ms,
    })
    #print(f"the identified hits are {hits}")
    metrics.RAG_SEARCH_TOTAL.inc()
    metrics.RAG_SEARCH_LATENCY_S.observe(dt_ms / 1000)
    return hits

def make_blocks(hits: List[Hit], logger = None):
    logger = logger or logging.getLogger("intelligent-reasoner")
    blocks = []
    for hit in hits:
        blocks.append(f"""source : {hit.metadata['file_name']}| chunk_id: {hit.id}| | text: {hit.document} """)
    #print(f"The identified blocks are {blocks}")
    result = "\n".join(blocks)
    if len(hits) == 0:
        logger.info("make_blocks: no block to form", extra={
            "number of blocks": len(hits)
        })
    if len(result)> 2000:
        logger.warning("make_blocks: large context build", extra={
            "context_characters":len(result)
        })
    return result

def make_prompt(question: str, block: str, logger =None)-> str:
    logger = logger or logging.getLogger("intelligent-reasoner")
    prompt =  f"""
Question: {question}
Content: {block}. 
Answer using ONLY the content above, if the answer is not present, set cause = "Unknown".

Return ONLY a valid JSON object with this structure:
{{
  "cause": "<one of: Manufacturing, DemandSpike, Quality, Logistics, Unknown>",
  "confidence": <a number between 0 and 1>
  "source" : "<the relevant file name indicated in the content>"
}}
Do NOT add any explanation, text, or formatting outside the JSON.
Do NOT include markdown.
Do NOT include explanation.
DO NOY include backticks."""
    prompt_length = len(prompt)
    logger.debug("make_prompt", extra = {
        "prompt_char_count": prompt_length
    })
    if prompt_length > 4000:
        logger.warning("make_prompt: number of characters exceed", extra = {
            "prompt_char_count": prompt_length
        })


    return prompt



def _call_ollama(prompt: str, logger = None):
    logger = logger or logging.getLogger("intelligent-reasoner")
    start = time.perf_counter()
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
            options = {
                "num_predict": NUM_CHAR_OUT, "temperature":TEMPRATURE
            },
        # timeout = LLM_TIMEOUT_S,
        )
        return response["message"]["content"]

    with ThreadPoolExecutor(max_workers = 1) as executor:
        future = executor.submit(_call)
        try:
            out = future.result(timeout=LLM_TIMEOUT_S)
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            prompt_tokens = len(encoding.encode(prompt))
            completion_tokens = len(encoding.encode(out))
            total_tokens = prompt_tokens + completion_tokens
            ms = int((time.perf_counter()-start)*1000)
            logger.info("llm.call.success",extra = {
                "llm_provider":"local",
                "model" : MODEL,
                "latency_ms" : ms,
                "response_char_count":len(out)
            })
            logger.info("llm.token_usage", extra ={
                "model": MODEL,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            })
            metrics.LLM_TOKEN_TOTAL.labels(
                provider = LLM_PROVIDER,
                type = "prompt"
            ).inc(prompt_tokens)
            metrics.LLM_TOKEN_TOTAL.labels(
                provider = LLM_PROVIDER,
                type = "completion"
            ).inc(completion_tokens)
            metrics.LLM_TOKEN_TOTAL.labels(
                provider = LLM_PROVIDER,
                type = "total"
            ).inc(total_tokens) 
            metrics.LLM_LATENCY_SECONDS.labels(
                provider = LLM_PROVIDER
            ).observe(ms/1000)
                       
            return out
        except FuturesTimeout:
            ms = int((time.perf_counter() - start) * 1000)
            logger.exception("llm.call.error", extra={
                "llm_provider": "local",
                "model": MODEL,
                "latency_ms": ms
            })
            raise TimeoutError(f"LLM call exceeded {LLM_TIMEOUT_S}s") 
            
def _call_vertex(prompt: str, logger=None) -> str:
    logger = logger or logging.getLogger("intelligent-reasoner")
    start = time.perf_counter()
    model = None

    try:
        vertexai.init(
            project=GOOGLE_CLOUD_PROJECT,
            location=VERTEX_LOCATION,
        )

        model = GenerativeModel(VERTEX_MODEL)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": TEMPRATURE,
                "max_output_tokens": NUM_CHAR_OUT,
                "response_mime_type":"application/json"
            },
        )

        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count
        completion_tokens = usage.candidates_token_count
        total_tokens = usage.total_token_count

        if not response.candidates:
            raise ValueError("Vertex returned no candidates")
        
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
            
        try:
            out = response.text
        except Exception:
            raise ValueError(f"Vertex returned no readable text. finish_reason={finish_reason}")
        ms = int((time.perf_counter() - start) * 1000)

        logger.info(
            "llm.call.success",
            extra={
                "llm_provider": "vertex",
                "model": VERTEX_MODEL,
                "latency_ms": ms,
                "response_char_count": len(out),
            },
        )

        logger.info(
            "llm.token_usage",
            extra={
                "provider": "vertex",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )

        metrics.LLM_TOKEN_TOTAL.labels(
            provider="vertex",
            type="prompt",
        ).inc(prompt_tokens)

        metrics.LLM_TOKEN_TOTAL.labels(
            provider="vertex",
            type="completion",
        ).inc(completion_tokens)

        metrics.LLM_TOKEN_TOTAL.labels(
            provider="vertex",
            type="total",
        ).inc(total_tokens)

        metrics.LLM_LATENCY_SECONDS.labels(
            provider="vertex"
        ).observe(ms / 1000)

        return out

    except Exception:
        ms = int((time.perf_counter() - start) * 1000)
        logger.exception(
            "llm.call.error",
            extra={
                "llm_provider": "vertex",
                "model": VERTEX_MODEL,
                "latency_ms": ms,
            },
        )
        raise

def call_llm(prompt:str, logger = None)->str:
    logger = logger or logging.getLogger("intelligent-reasoner")
    logger.info("call_llm", extra = {
        "LLM_PROVIDER": LLM_PROVIDER
    })
   
    
    if LLM_PROVIDER =="vertex":
        metrics.LLM_CALLS_TOTAL.labels(
        provider = "vertex"
        ).inc()
        return _call_vertex(prompt, logger = logger)
    elif LLM_PROVIDER =="local":
        metrics.LLM_CALLS_TOTAL.labels(
        provider = "local"
        ).inc()
        return _call_ollama(prompt, logger = logger)
    
    else:
        logger.error("call_llm: invalide_provider", extra = {
            "LLM_PROVIDER":LLM_PROVIDER
        })
        raise ValueError(f"Uknown LLM_PROVIDER {LLM_PROVIDER}")    

_LLM_VALID_CACHE: Dict[str,Tuple[float,str]] = {}
def _cach_key(prompt:str, provider:str,model:str, temprature:float,max_tokens:int)->str:
    playload = f"{provider}|{model}|temp={temprature}|max_tokens={max_tokens}|{prompt}"
    return hashlib.sha256(playload.encode("utf-8")).hexdigest()

def _cache_get(key:str)->str|None:
    item = _LLM_VALID_CACHE.get(key)
    if not item:
        return None
    expires_at, raw = item
    if time.time()>=expires_at:
        _LLM_VALID_CACHE.pop(key,None)
    return raw

def _cach_set(key:str,raw:str, ttl_s:int = CACHE_TTL_S)->None:
    expires_at = time.time()+ttl_s
    _LLM_VALID_CACHE[key]=(expires_at,raw)

def call_llm_with_validation(prompt: str, call_llm, max_retries: int = 2, logger = None) -> LLMAnswer:
    logger = logger or logging.getLogger("intelligent-reasoner")
    start = time.perf_counter()
    key = _cach_key(prompt, LLM_PROVIDER,MODEL,TEMPRATURE,NUM_CHAR_OUT)
    cached = _cache_get(key)
    if cached is not None:
        logger.info("llm.cache_hit", extra = {
            "chache":"valid", "key":key
        })
        raw = cached
        return LLMAnswer.model_validate(json.loads(raw))
    
    logger.info("llm.cache_miss", extra= {"cache":"valid"})    
    raw = call_llm(prompt, logger = logger)
        
    len_prompt = len(prompt)
    logger.info('call_llm_with_validation.start',extra={
        "max_retries":max_retries,
        "prompt_chars":len_prompt
    })
    last_err_type = None
    for attempt in range(max_retries + 1):
        try:
            raw = raw.strip()
            if raw.startswith("```"):
              parts = raw.split("`")
              if len(parts)>=2:
                raw = patrs[1].strip()
                if raw.startswith("json"):
                  raw = raw[4:].strip()
              
            logger.info("llm.raw_output", extra={"raw":raw})
            data = json.loads(raw)
            obj = LLMAnswer.model_validate(data)
            ms = int((time.perf_counter()-start)*1000)

            logger.info("llm_with_validaiton.success", extra={
                "attempt": attempt,
                "latency_ms":ms,
                "raw_chars": len(raw)
            })
            _cach_set(key,raw)
            return obj

        except TimeoutError:
            last_err_type = "timeout"
            if logger:
                ms = int((time.perf_counter() - start) * 1000)
                logger.error(
                    "llm_with_validation_timeout",
                    extra={"attempt": attempt, "duration_ms": ms}
                )
            metrics.LLM_VALIDATION_FAIL_TOTAL.labels(reason = "timeout").inc()
            raise 
        except json.JSONDecodeError as e:
            last_err_type = "json_decode"
            if logger:
                logger.warning(
                    "llm_with_validation_bad_json",
                    extra={"attempt": attempt, "raw_chars": len(raw), "error": str(e)[:200]}
                )
            repair = (
                "Your output was not valid JSON. "
                "Return ONLY a JSON object matching the schema.\n\n"
                "SCHEMA:\n"
                '{"cause":"<one of: Manufacturing, DemandSpike, Quality, Logistics, Unknown>",'
                '"confidence":<0..1>,"source":"<file name>"}\n\n'
                "BAD OUTPUT:\n" + raw
            )
            metrics.LLM_VALIDATION_FAIL_TOTAL.labels(reason = "JSON Decoder Error").inc()
            raw = call_llm(repair, logger = logger)

        except ValidationError as e:
            last_err_type = "schema_validation"
            if logger:
                logger.warning(
                    "llm_with_validation_schema_error",
                    extra={"attempt": attempt, "raw_chars": len(raw), "error": str(e)[:200]}
                )
            repair = (
                "Your JSON did not match the schema (wrong fields/types/values). "
                "Return ONLY corrected JSON matching the schema.\n\n"
                "ERRORS:\n" + str(e) + "\n\n"
                "BAD JSON:\n" + raw
            )
            metrics.LLM_VALIDATION_FAIL_TOTAL.labels(reason = "JSON Validation Error").inc()

            raw = call_llm(repair, logger = logger)

    if logger:
        ms = int((time.perf_counter() - start) * 1000)
        logger.error(
            "llm_with_validation_failed",
            extra={"duration_ms": ms, "max_retries": max_retries, "last_error_type": last_err_type}
        )

    raise ValueError("Failed to get valid JSON after retries")
  

def answer_question(question: str, k:int = 3, logger = None) -> LLMAnswer:
    logger = logger or logging.getLogger("intelligent-reasoner")
    logger. info("reasoning started...")
    rag_similarities = rag_search(question , k,logger = logger)
    start = time.perf_counter()
    block = make_blocks(rag_similarities, logger = logger)
    prompt = make_prompt(question, block,logger = logger)
    duration_prompt_making = ((time.perf_counter()-start)*1000)
    #answer = call_llm(prompt)
    answer = call_llm_with_validation(prompt, call_llm, max_retries = MAX_TRIES, logger=logger)
    metrics.PROMPT_MAKING_LATENCY_S.observe(duration_prompt_making)
    return answer

def main():
    answer_question("Quelle est la raison d'une tension d’approvisionnement ")

if __name__ == "__main__":
    main()

