import json
from typing import List, Dict
from models.schemas import Claim, Evidence, FactCheckResult
from api.ollama_client import ask_ollama
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

def check_facts(claims: List[Claim], evidence_list: List[Evidence]) -> List[FactCheckResult]:
    results = []
    # Simple mapping to quickly get claim text
    claim_map = {c.claim_id: c for c in claims}    
    # Only verifying factual claims
    for ev in evidence_list:
        claim = claim_map.get(ev.claim_id)
        if not claim or claim.claim_type.lower() != 'factual':
            continue            
        # 1. Web search using ddgs with a clean query
        clean_text = claim.claim_text.replace('"', '').replace("'", "")
        # Limit query to reasonable search keywords length
        words = clean_text.split()[:8]
        search_query = " ".join(words)
        search_results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(search_query, max_results=3):
                    search_results.append(r)
        except Exception as e:
            print(f"Search failed for {search_query}: {e}")
            
        # If no search results, we cannot verify factually based on external sources
        if not search_results:
            results.append(FactCheckResult(
                claim_id=claim.claim_id,
                status="unsupported",
                source_url=None,
                reasoning="No external sources found to verify the claim."
            ))
            continue         
        # 2. Ollama evaluates the source
        # We will concatenate the top search snippets
        search_context = "\n".join([f"Source ({r.get('href', 'unknown')}): {r.get('body', '')}" for r in search_results])
        prompt = f"""
        Evaluate whether the provided reference material supports, contradicts, or leaves questionable the given claim and supporting evidence.
        Return the result ONLY as a JSON object with:
        - "status": One of "supported", "unsupported", "questionable", "contradicted"
        - "reasoning": A concise explanation of the evaluation
        Claim: "{claim.claim_text}"
        Evidence: "{ev.evidence_text}"
        
        Reference Material:
        {search_context}
        """
        response_text = ask_ollama(prompt, json_format=True)
        try:
            data = json.loads(response_text)
            status = data.get('status', 'questionable').lower()
            if status not in ["supported", "unsupported", "questionable", "contradicted"]:
                status = "questionable"
                
            results.append(FactCheckResult(
                claim_id=claim.claim_id,
                status=status,
                source_url=search_results[0].get('href'),
                reasoning=data.get('reasoning', '')
            ))
        except json.JSONDecodeError:
             results.append(FactCheckResult(
                claim_id=claim.claim_id,
                status="questionable",
                source_url=None,
                reasoning="Failed to parse AI evaluation of fact check."
            ))
    return results
