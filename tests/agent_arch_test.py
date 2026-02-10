from agent import runner
from api import config
#from agent.runner import agent_run, router
from contracts import TraceCall,ToolCall
import importlib
"""
def test_run_agent():
    test_query = "rag_help"
    TC = agent_run(test_query)
    assert TC.query == "rag_help"
    assert TC.tool_calls[0].name =="rag_search"
    assert TC.tool_calls[1].name =="llm_answer"
    assert TC.tool_calls[0].output_summary == "Document 1"
    assert TC.tool_calls[1].output_summary == "Document 1: here is what I can help you with ...."
    assert TC.tool_calls[0].ms >= 0
    assert TC.tool_calls[1].ms >= 0
    assert len(TC.tool_calls) == 2
"""
def test_router_triggers_rag_on_keywords():
    test_query1 = "Selon ANSM, c'est quoi la médicament le plus utilisé?"
    router_policy1 = runner.router(test_query1)
    assert router_policy1 == "rag_search"
    test_query2 = "c'est quoi la médicament le plus utilisé?"
    router_policy2 = runner.router(test_query2)
    assert router_policy2 == "llm_answer"
    




def test_agent_run_rag_path_has_two_tool_calls():
    test_query1 = "Selon ANSM, c'est quoi la médicament le plus utilisé?"
    TC = runner.agent_run(test_query1)
    assert len(TC.tool_calls) == 2
    assert TC.tool_calls[0].name == 'rag_search'
    assert TC.tool_calls[1].name == 'llm_answer'
    assert TC.status == "success"


def test_agent_run_non_rag_path_has_one_tool_call():
    test_query2 = "c'est quoi la médicament le plus utilisé?"
    TC = runner.agent_run(test_query2)
    assert len(TC.tool_calls) == 1
    assert TC.tool_calls[0].name =='llm_answer'
    assert TC.status == "success"

def test_router_forces_llm_when_rag_disabled(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED" ,"false")
    importlib.reload(config)
    importlib.reload(runner)

    decision = runner.router('Selon ANSL, quelle est la pénurie?')
    assert decision =="llm_answer"

