#from api.logging_config import logger
from contracts import TraceCall,ToolName, ToolCall, ToolChoice, Hit, RouterPolicy
from typing import List, Optional,Tuple
import uuid
import time
from api.config import RAG_TOP_K, RAG_ENABLED
from api import metrics
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

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

ROUTER_LLM_MODEL = "qwen2.5:1.5b-instruct"
ROUTER_LLM_TEMP = 0.0
ROUTER_CONF_THRESHOLD = 0.70

def build_router_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a routing classifier for a RAG application.\n"
         "Decide whether to answer directly (llm_answer) or retrieve documents first (rag_search).\n"
         "\n"
         "Use rag_search when the user asks for sources/references/proofs/citations or mentions ANSM/BDPM/documents.\n"
         "Use llm_answer for general explanations, brainstorming, writing, or coding help.\n"
         "\n"
         "Return output strictly matching the RouterPolicy schema."),
        ("human", "Query: {query}")
    ])

    llm = ChatOllama(model=ROUTER_LLM_MODEL, temperature=ROUTER_LLM_TEMP)
    return prompt | llm.with_structured_output(RouterPolicy)

ROUTER_CHAIN = build_router_chain()

def llm_router_decide(query:str, logger = None) -> RouterPolicy:
    decision : RouterPolicy = ROUTER_CHAIN.invoke({"query":query})
    if logger:
        logger.info(
            f"llm_router_decide: tool={decision.tool} "
            f"confidence={decision.confidence} trigger={decision.trigger}"
        )
    return decision

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
        answer = llm_answer(query = query, logger = logger)
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
        hits = rag_search(query = query, logger=logger)
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
        answer = llm_answer(query = query, hits = hits,logger = logger)
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
    total_ms = int((total_end-total_start) * 1000)
    
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


def router(query:str, logger = None)-> Tuple[str, Optional[str]]:
    if not RAG_ENABLED:
        if logger:
            logger.info("router: rag disabled -> llm answer")
        return "llm_answer", None    

    q_lower = query.lower()
    trigger = next((w for w in RAG_KEYWORDS if w in q_lower), None)

    if trigger is not None:
        if logger:
            logger.info(f"router: rag_searcg(trigger={trigger})")
        return "rag_search", trigger
    
    try:
        decision = llm_router_decide(query= query, logger = logger)
        if decision.confidence > ROUTER_CONF_THRESHOLD:
            return decision.tool, decision.trigger
        else:
            if logger:
                logger.info(
                    f"router: low confidence ({decision.confidence} < {ROUTER_CONF_THRESHOLD}) "
                    "-> fallback llm_answer"
                )
            return "llm_answer", None
    except Exception as e:
        
        if logger:
            logger.exception(f"router: LLM router failed -> fallback llm_answer: {e}")
        return "llm_answer", None           

def llm_answer(query:str, hits: Optional[List[Hit]] = None, logger = None) -> str:
    
    logger.info(f"llm_answer: hits_type={type(hits)} and logger type = {type(logger)}")
    from reasoning import call_llm_with_validation, make_blocks,make_prompt
    from reasoning import call_llm as v1_call_llm
    if hits is None:
        start = time.perf_counter()
        prompt = make_prompt(question=query, block = "", logger=logger)
        duration_prompt_make = ((time.perf_counter()-start)*1000)
        metrics.PROMPT_MAKING_LATENCY_S.observe(duration_prompt_make/1000)
        response = call_llm_with_validation(prompt,v1_call_llm, logger = logger)
    else:
        start = time.perf_counter()
        blocks = make_blocks(hits = hits, logger = logger)
        prompt = make_prompt(question=query,block = blocks, logger= logger)
        duration_prompt_make = ((time.perf_counter()-start)*1000)
        metrics.PROMPT_MAKING_LATENCY_S.observe(duration_prompt_make/1000)
        response = call_llm_with_validation(prompt,v1_call_llm,logger = logger)
    
    return response.model_dump_json()

    
    """
    if hits is None:
        output_summary = "here is what I can help you with ..."
    else:    
        doc = " ".join([hit.document for hit in hits])
        output_summary = f'{doc}: here is what I can help you with ....'

    return output_summary
    """

def rag_search(query:str , logger = None)->List[Hit]:
    from reasoning import rag_search as v1_rag_search

    hits = v1_rag_search(question=query , k=RAG_TOP_K , logger=logger)
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

  

