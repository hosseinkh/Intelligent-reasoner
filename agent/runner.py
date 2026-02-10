#from api.logging_config import logger
from contracts import TraceCall,ToolName, ToolCall, ToolChoice, Hit
from typing import List, Optional,Tuple
import uuid
import time
from api.config import RAG_TOP_K, RAG_ENABLED

RAG_KEYWORDS = [
    "selon",
    "source",
    "document",
    "ansm",
    "bdpm",
    "référence",
    "preuve",
    "rapport"
]

def agent_run(query:str , logger = None) -> TraceCall:
    total_start = time.time()
    logger.info("agent_run started...")
    run_id = str(uuid.uuid4())
    tool_calls =[]
    tool_name, router_trigger  = router(query, logger)
    if tool_name == 'llm_answer':
        hits = None
        start = time.time()
        sources = None
        avg_score = None

        retrieval_stats = {
            "k" : 0,
            "avg_score": avg_score,
        }
        answer = llm_answer(query)
        end = time.time()
        status = 'success'
        ms = int((end - start) * 1000)
        tool_call = ToolCall(
        name = tool_name,
        input_text = query,
        output_summary = answer,
        status = status,
        ms = ms,
        )
        tool_calls.append(tool_call)

    elif tool_name == 'rag_search':
        start = time.time()
        hits = rag_search(query)
        sources = list({hit.metadata['file_name'] for hit in hits})
        scores = [hit.score for hit in hits]
        avg_score = sum(scores) / len(scores) if hits else None

        retrieval_stats = {
            "k" : len(hits),
            "avg_score": avg_score,
        }
        end = time.time()
        ms1 = int((end - start) * 1000)
        start = time.time()
        answer = llm_answer(query, hits)
        end = time.time()
        ms2 = int((end - start) * 1000)
        status = 'success'
        tool_name1 = 'rag_search'
        tool_name2 = 'llm_answer'
        doc = " ".join([hit.document for hit in hits])
        tool_call = [ToolCall(
        name = tool_name1,
        input_text = query,
        output_summary = doc,
        status = status,
        ms = ms1,
        ), ToolCall(
        name = tool_name2,
        input_text = query,
        output_summary = answer,
        status = status,
        ms = ms2,
        )]

        tool_calls.extend(tool_call)
    else:
        tool_name = None
        status = 'fail'
        answer = "No tool matched"
    
    final_answer = answer
    total_end = time.time()
    total_ms = int(total_end-total_start) * 1000
    
    return TraceCall(
        run_id = run_id,
        query = query,
        router_policy = tool_name,
        router_trigger = router_trigger,
        rag_enabled = RAG_ENABLED,
        tool_calls = tool_calls,
        status = status,
        final_answer = final_answer,
        sources = sources, 
        retrieval_stats = retrieval_stats,
        total_ms = total_ms
        )

"""
def router(query:str, logger = None) -> ToolChoice:
    if RAG_ENABLED:
        q_lower = query.lower()
        if any( w in q_lower for w in RAG_KEYWORDS):
            logger.info("the router policy is rag_search")
            return "rag_search"
        else:
            logger.info("the router policy is llm_search")
            return "llm_answer"
    else:
        logger.info("!!! Rag is not enabled by the user !!!")
        return "llm_answer"

    ####
    if query == 'help':
        return 'llm_answer'
    elif query == 'rag_help':
        return 'rag_search'    
    else:
        return None
    """
def router(query:str, logger = None)-> Tuple[str, Optional[str]]:
    if not RAG_ENABLED:
        if logger:
            logger.info("router: rag disabled -> llm answer")
        return "logger_llm", None    

    q_lower = query.lower()
    trigger = next((w for w in RAG_KEYWORDS if w in q_lower), None)

    if trigger is not None:
        if logger:
            logger.info(f"router: rag_searcg(trigger={trigger})")
        return "rag_search", trigger

    if logger:
        logger.info("router: llm_answer (no keyword match)")
    return "llm_answer" , None            

def llm_answer(query:str, hits: Optional[List[Hit]] = None) -> str:
    from reasoning import call_llm_with_validation, make_blocks,make_prompt
    from reasoning import call_llm as v1_call_llm
    if hits is None:
        prompt = make_prompt(query, block="")
        response = call_llm_with_validation(prompt,v1_call_llm)
    else:
        blocks = make_blocks(hits)
        prompt = make_prompt(query,blocks)
        response = call_llm_with_validation(prompt,v1_call_llm)
    
    return response.model_dump_json()

    
    """
    if hits is None:
        output_summary = "here is what I can help you with ..."
    else:    
        doc = " ".join([hit.document for hit in hits])
        output_summary = f'{doc}: here is what I can help you with ....'

    return output_summary
    """

def rag_search(query:str)->List[Hit]:
    from reasoning import rag_search as v1_rag_search

    hits = v1_rag_search(query , k = RAG_TOP_K)
    return hits
    """
    return [
        Hit(
            id = str(1),
            score = 0.8,
            document = "Document 1",
            metadata = {}
        )
    ]
    """

  

