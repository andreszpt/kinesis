from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from hybrid_training_agent.clients.science_client import ScienceLiteratureClient


@tool
def search_scientific_literature(
    query: str,
    max_results: int = 5,
    prefer_meta_analyses_and_trials: bool = True,
) -> dict[str, Any]:
    """Search PubMed and Europe PMC for peer-reviewed exercise physiology and sports science literature.

    Use this tool to find scientific evidence on concurrent training, the interference effect,
    running economy, strength training for runners, HRV-guided training, or periodization.
    Returns titles, authors, publication years, journals, PMIDs, DOIs, and full abstracts.
    """
    client = ScienceLiteratureClient()
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        results = client.search_pubmed(
            query=query,
            max_results=max_results,
            prefer_reviews_and_trials=prefer_meta_analyses_and_trials,
        )
    except Exception as exc:
        errors.append(f"PubMed search error: {exc}")

    if len(results) < max_results:
        try:
            epmc = client.search_europe_pmc(query=query, max_results=max_results)
            existing_pmids = {r.get("pmid") for r in results if r.get("pmid")}
            for item in epmc:
                if item.get("pmid") not in existing_pmids:
                    results.append(item)
                    if len(results) >= max_results:
                        break
        except Exception as exc:
            errors.append(f"Europe PMC search error: {exc}")

    return {
        "query": query,
        "count": len(results),
        "papers": results,
        "warnings": errors,
    }


@tool
def fetch_pubmed_papers_by_pmid(pmids: list[str]) -> dict[str, Any]:
    """Fetch complete metadata, abstract, and DOI for specific PubMed IDs (PMIDs)."""
    client = ScienceLiteratureClient()
    try:
        papers = client.fetch_pubmed_by_pmids(pmids)
        return {"count": len(papers), "papers": papers}
    except Exception as exc:
        return {"count": 0, "papers": [], "error": str(exc)}


@tool
def get_curated_hybrid_training_evidence(topic: str = "all") -> dict[str, Any]:
    """Retrieve curated landmark meta-analyses, systematic reviews, and RCTs on hybrid running + strength training.

    Topics available:
    - 'concurrent_interference' (Schumann et al. 2022, Wilson et al. 2012, Robineau et al. 2016, Coffey & Hawley 2017)
    - 'running_economy_and_strength' (Blagrove et al. 2018, Rønnestad & Mujika 2014)
    - 'intensity_distribution_and_polarization' (Seiler 2010, Stöggl & Sperlich 2014)
    - 'hrv_and_load_monitoring' (Plews et al. 2013, Foster et al. 2001)
    - 'all' (returns all topics with PMIDs, DOIs, quantitative effect sizes, and programming implications)
    """
    client = ScienceLiteratureClient()
    return {
        "topic": topic,
        "evidence_by_category": client.get_curated_evidence(topic),
    }


SCIENCE_TOOLS = [
    search_scientific_literature,
    fetch_pubmed_papers_by_pmid,
    get_curated_hybrid_training_evidence,
]
