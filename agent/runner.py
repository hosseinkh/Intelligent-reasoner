from contracts import TraceCall,ToolName, ToolCall, ToolChoice, Hit
from typing import List, Optional
import uuid
import time

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

def agent_run(query:str) -> TraceCall:
    run_id = str(uuid.uuid4())
    tool_calls =[]
    tool_name  = router(query)
    if tool_name == 'llm_answer':
        start = time.time()
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
    return TraceCall(
        run_id = run_id,
        query = query,
        tool_calls = tool_calls,
        status = status,
        final_answer = final_answer,
        )

def router(query:str) -> ToolChoice:
    q_lower = query.lower()
    if any( w in q_lower for w in RAG_KEYWORDS):
        return "rag_search"
    else:
        return "llm_answer"
    """
    if query == 'help':
        return 'llm_answer'
    elif query == 'rag_help':
        return 'rag_search'    
    else:
        return None
    """
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

    hits = v1_rag_search(query , k = 3)
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

  

