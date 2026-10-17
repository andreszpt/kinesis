from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import httpx

from hybrid_training_agent.config import Settings, get_settings
from hybrid_training_agent.models import ScientificCitation

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Foundational peer-reviewed meta-analyses, systematic reviews, and RCTs on Hybrid Running + Strength Training
CURATED_HYBRID_EVIDENCE: dict[str, list[ScientificCitation]] = {
    "concurrent_interference": [
        ScientificCitation(
            title="Compatibility of Concurrent Aerobic and Strength Training for Skeletal Muscle Size and Function: An Updated Systematic Review and Meta-Analysis",
            authors="Schumann M, Feuerbacher JF, Sünkeler M, Freitag N, Rønnestad BR, Doma K, Lundberg TR",
            year=2022,
            journal="Sports Medicine",
            pmid="34757594",
            doi="10.1007/s40279-021-01587-7",
            study_type="Systematic Review & Meta-Analysis (43 studies)",
            key_findings=(
                "Concurrent aerobic and strength training does NOT compromise maximal strength (SMD = -0.06, p=0.45) "
                "or muscle hypertrophy (SMD = -0.01, p=0.91) when total volume and recovery are managed; "
                "HOWEVER, explosive strength/rate of force development (RFD) IS significantly attenuated (SMD = -0.28, p=0.007), "
                "particularly when aerobic and strength sessions are performed within the same session (<3h apart)."
            ),
            practical_application=(
                "Separate explosive/heavy lower-body strength and running sessions by at least 6–24 hours when possible; "
                "perform power/RFD movements first when fresh."
            ),
        ),
        ScientificCitation(
            title="Concurrent training: a meta-analysis examining interference of aerobic and resistance exercises",
            authors="Wilson JM, Marin PJ, Rhea MR, Wilson SMC, Loenneke JP, Anderson JC",
            year=2012,
            journal="Journal of Strength and Conditioning Research",
            pmid="22002517",
            doi="10.1519/JSC.0b013e31823a3e2d",
            study_type="Meta-Analysis (21 studies, 422 effect sizes)",
            key_findings=(
                "Endurance running causes significantly greater lower-body strength and hypertrophy interference than cycling "
                "(likely due to eccentric muscle damage and ground reaction forces in running). Interference scales positively "
                "with endurance frequency (>3 days/wk) and duration (>30-40 min high intensity)."
            ),
            practical_application=(
                "In running + strength hybrid plans, minimize eccentric muscle damage prior to key running quality sessions, "
                "keep easy runs truly low-intensity (Zone 1-2), and avoid high-rep lower-body failure before long or interval runs."
            ),
        ),
        ScientificCitation(
            title="Does the time interval between strength and endurance exercises affect neuromuscular adaptations?",
            authors="Robineau J, Babault N, Piscione J, Lacome M, Bigard AX",
            year=2016,
            journal="Journal of Strength and Conditioning Research",
            pmid="25546444",
            doi="10.1519/JSC.0000000000000798",
            study_type="Randomized Controlled Trial",
            key_findings=(
                "Comparing 0h, 6h, and 24h recovery between strength and endurance sessions over 7 weeks: "
                "24h recovery produced maximal strength and aerobic gains; 6h recovery preserved most adaptations "
                "compared to 0h (back-to-back), which blunted both maximal voluntary contraction and VO2peak improvements."
            ),
            practical_application=(
                "Enforce a minimum 6-hour recovery window (AM/PM split) between same-day strength and running, "
                "and ideally 24 hours between heavy lower-body lifting and high-intensity interval running."
            ),
        ),
        ScientificCitation(
            title="Concurrent exercise training: do opposites distract?",
            authors="Coffey VG, Hawley JA",
            year=2017,
            journal="Journal of Physiology",
            pmid="27706542",
            doi="10.1113/JP272270",
            study_type="Mechanistic Review",
            key_findings=(
                "Glycogen depletion and high energetic stress activate AMPK and SIRT1, which can transiently inhibit mTORC1 "
                "myofibrillar protein synthesis signaling when resistance training is performed in an energy-depleted state."
            ),
            practical_application=(
                "Avoid heavy strength training immediately after glycogen-depleting long runs or VO2max intervals; "
                "refuel with carbohydrates (1.0-1.2 g/kg) and protein (0.3-0.4 g/kg) between double-day sessions."
            ),
        ),
    ],
    "running_economy_and_strength": [
        ScientificCitation(
            title="Effects of Strength Training on the Physiological Determinants of Middle- and Long-Distance Running Performance: A Systematic Review",
            authors="Blagrove RC, Howatson G, Hayes PR",
            year=2018,
            journal="Sports Medicine",
            pmid="29249083",
            doi="10.1007/s40279-017-0835-7",
            study_type="Systematic Review (24 RCTs)",
            key_findings=(
                "Adding 2–3 sessions/week of heavy resistance training (≥80% 1RM, low-to-moderate reps 3-6) and/or plyometrics "
                "to distance runners' programs improves running economy by 2% to 8%, vVO2max, and 1.5–10 km time-trial performance "
                "by 3% to 5%, with zero negative effect on VO2max or body mass."
            ),
            practical_application=(
                "Prioritize heavy neural strength (3–5 sets of 3–6 reps at 80–90% 1RM / 1–3 RIR) and reactive plyometrics "
                "rather than high-rep metabolic fatigue sets for lower-body sessions in runners."
            ),
        ),
        ScientificCitation(
            title="Optimizing strength training for running and cycling endurance performance: A review",
            authors="Rønnestad BR, Mujika I",
            year=2014,
            journal="Scandinavian Journal of Medicine & Science in Sports",
            pmid="23914932",
            doi="10.1111/sms.12104",
            study_type="Review",
            key_findings=(
                "Heavy strength training increases musculotendinous stiffness, delays type II muscle fiber recruitment during "
                "submaximal running, and reduces rate of force development requirements per stride. In-season maintenance "
                "requires only 1 session/week, while build phases benefit from 2 sessions/week."
            ),
            practical_application=(
                "Use 2 lower-body/full-body strength sessions/week during base/build blocks, and reduce to 1 maintenance "
                "heavy session/week during peak running volume or taper blocks."
            ),
        ),
    ],
    "intensity_distribution_and_polarization": [
        ScientificCitation(
            title="What is best practice for training intensity and duration distribution in endurance athletes?",
            authors="Seiler S",
            year=2010,
            journal="International Journal of Sports Physiology and Performance",
            pmid="20861519",
            doi="10.1123/ijspp.5.3.276",
            study_type="Review & Observational Analysis",
            key_findings=(
                "Elite and high-performing endurance athletes consistently converge on an ~80:20 session distribution "
                "(~80% of sessions below first ventilatory/lactate threshold VT1/LT1 in Zone 1-2, and ~20% at or above LT2 in Zone 4-5), "
                "minimizing 'grey zone' Zone 3 autonomic stress."
            ),
            practical_application=(
                "Keep easy and recovery runs strictly below LT1 (Zone 1-2 HR/Pace) so autonomic nervous system (HRV) recovery "
                "is preserved for both high-intensity running intervals and heavy strength sessions."
            ),
        ),
        ScientificCitation(
            title="Polarized training has greater impact on key endurance variables than threshold, high intensity, or high volume training",
            authors="Stöggl T, Sperlich B",
            year=2014,
            journal="Frontiers in Physiology",
            pmid="24550842",
            doi="10.3389/fphys.2014.00033",
            study_type="Randomized Controlled Trial (48 endurance athletes, 9 weeks)",
            key_findings=(
                "Polarized training led to the greatest improvements in VO2peak (+11.7%), time to exhaustion (+17.4%), "
                "and peak velocity/power (+5.1%) compared to threshold-heavy, HIIT-only, or high-volume-only groups."
            ),
            practical_application=(
                "Structure weekly running around 1–2 high-quality sessions (Threshold/VO2max) and 2–4 strictly aerobic Zone 1-2 runs."
            ),
        ),
    ],
    "hrv_and_load_monitoring": [
        ScientificCitation(
            title="Training adaptation and heart rate variability in elite endurance athletes: opening the door to effective monitoring",
            authors="Plews DJ, Laursen PB, Stanley J, Kilding AE, Buchheit M",
            year=2013,
            journal="Sports Medicine",
            pmid="23821469",
            doi="10.1007/s40279-013-0071-8",
            study_type="Review",
            key_findings=(
                "7-day rolling average of morning RMSSD (or lnRMSSD) and its coefficient of variation (CV) are superior to isolated single-day "
                "HRV readings for detecting functional overreaching and autonomic fatigue."
            ),
            practical_application=(
                "When 7-day HRV drops below baseline normal range or RHR elevates >4-5 bpm alongside high subjective fatigue, "
                "substitute high-CNS intervals/heavy squats with Zone 1 aerobic recovery or mobility."
            ),
        ),
        ScientificCitation(
            title="A new approach to monitoring exercise training",
            authors="Foster C, Florhaug JA, Franklin J, Gottschall L, Hrovatin LA, Parker S, Doleshal P, Dodge C",
            year=2001,
            journal="Journal of Strength and Conditioning Research",
            pmid="11219501",
            doi="10.1519/1533-4287(2001)015<0109:ANATME>2.0.CO;2",
            study_type="Validation Study",
            key_findings=(
                "Session RPE (sRPE = duration in minutes × RPE 1-10) is a valid, reliable method for quantifying internal training load "
                "across BOTH steady-state endurance, high-intensity intervals, and resistance training where heart-rate TRIMP underestimates load."
            ),
            practical_application=(
                "Because Suunto watches only record duration and HR during strength sessions (missing neuromuscular strain), "
                "use sRPE to calibrate resistance training load in intervals.icu."
            ),
        ),
    ],
}


class ScienceLiteratureClient:
    """Client for searching PubMed (NCBI Entrez) and Europe PMC for sports science evidence."""

    def __init__(self, settings: Settings | None = None, timeout: float = 20.0) -> None:
        self.settings = settings or get_settings()
        self.timeout = timeout

    def _pubmed_params(self, extra: dict[str, Any]) -> dict[str, Any]:
        params = {
            "tool": "hybrid-training-agent",
            "email": self.settings.ncbi_email,
            **extra,
        }
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    def search_pubmed(
        self,
        query: str,
        max_results: int = 5,
        prefer_reviews_and_trials: bool = True,
    ) -> list[dict[str, Any]]:
        """Search PubMed E-Utilities and return parsed paper metadata + abstracts."""
        effective_query = query.strip()
        if prefer_reviews_and_trials and "meta-analysis" not in effective_query.lower():
            effective_query = (
                f"({effective_query}) AND "
                "(Meta-Analysis[pt] OR Systematic Review[pt] OR Randomized Controlled Trial[pt] OR Review[pt])"
            )

        with httpx.Client(timeout=self.timeout) as client:
            esearch_resp = client.get(
                f"{PUBMED_BASE}/esearch.fcgi",
                params=self._pubmed_params(
                    {
                        "db": "pubmed",
                        "term": effective_query,
                        "retmax": max_results,
                        "retmode": "json",
                        "sort": "relevance",
                    }
                ),
            )
            esearch_resp.raise_for_status()
            id_list = (
                esearch_resp.json()
                .get("esearchresult", {})
                .get("idlist", [])
            )

            # Fallback to broader query if filter was too strict
            if not id_list and effective_query != query.strip():
                esearch_resp = client.get(
                    f"{PUBMED_BASE}/esearch.fcgi",
                    params=self._pubmed_params(
                        {
                            "db": "pubmed",
                            "term": query.strip(),
                            "retmax": max_results,
                            "retmode": "json",
                            "sort": "relevance",
                        }
                    ),
                )
                esearch_resp.raise_for_status()
                id_list = (
                    esearch_resp.json()
                    .get("esearchresult", {})
                    .get("idlist", [])
                )

        if not id_list:
            return []

        return self.fetch_pubmed_by_pmids(id_list)

    def fetch_pubmed_by_pmids(self, pmids: list[str]) -> list[dict[str, Any]]:
        """Fetch full titles, authors, journals, years, DOIs, and abstracts for a list of PMIDs."""
        if not pmids:
            return []

        with httpx.Client(timeout=self.timeout) as client:
            efetch_resp = client.get(
                f"{PUBMED_BASE}/efetch.fcgi",
                params=self._pubmed_params(
                    {
                        "db": "pubmed",
                        "id": ",".join(str(p).strip() for p in pmids),
                        "retmode": "xml",
                    }
                ),
            )
            efetch_resp.raise_for_status()

        return self._parse_pubmed_xml(efetch_resp.text)

    @staticmethod
    def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
        root = ET.fromstring(xml_text)
        articles: list[dict[str, Any]] = []

        for pubmed_article in root.findall(".//PubmedArticle"):
            pmid_el = pubmed_article.find(".//MedlineCitation/PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None

            title_el = pubmed_article.find(".//Article/ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else "Untitled"

            journal_el = pubmed_article.find(".//Article/Journal/Title")
            journal = journal_el.text.strip() if journal_el is not None and journal_el.text else ""

            year_el = pubmed_article.find(".//Article/Journal/JournalIssue/PubDate/Year")
            if year_el is None:
                year_el = pubmed_article.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate")
            year = year_el.text.strip()[:4] if year_el is not None and year_el.text else "Unknown"

            author_names: list[str] = []
            for author in pubmed_article.findall(".//Article/AuthorList/Author")[:5]:
                last = author.findtext("LastName", default="").strip()
                initials = author.findtext("Initials", default="").strip()
                if last:
                    author_names.append(f"{last} {initials}".strip())
            authors_str = ", ".join(author_names)
            if len(pubmed_article.findall(".//Article/AuthorList/Author")) > 5:
                authors_str += " et al."

            abstract_parts: list[str] = []
            for abs_text in pubmed_article.findall(".//Article/Abstract/AbstractText"):
                label = abs_text.attrib.get("Label")
                content = "".join(abs_text.itertext()).strip()
                if content:
                    abstract_parts.append(f"{label}: {content}" if label else content)
            abstract = " ".join(abstract_parts)

            doi: str | None = None
            for article_id in pubmed_article.findall(".//PubmedData/ArticleIdList/ArticleId"):
                if article_id.attrib.get("IdType") == "doi" and article_id.text:
                    doi = article_id.text.strip()
                    break

            pub_types = [
                pt.text.strip()
                for pt in pubmed_article.findall(".//Article/PublicationTypeList/PublicationType")
                if pt.text
            ]

            articles.append(
                {
                    "pmid": pmid,
                    "doi": doi,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                    "title": title,
                    "authors": authors_str or "Unknown",
                    "year": year,
                    "journal": journal,
                    "publication_types": pub_types,
                    "abstract": abstract or "No abstract available.",
                }
            )

        return articles

    def search_europe_pmc(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search Europe PMC REST API for open-access sports science literature and citation counts."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                EUROPE_PMC_BASE,
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": max_results,
                    "resultType": "core",
                },
            )
            resp.raise_for_status()
            results = resp.json().get("resultList", {}).get("result", [])

        papers: list[dict[str, Any]] = []
        for item in results:
            pmid = item.get("pmid")
            doi = item.get("doi")
            papers.append(
                {
                    "pmid": pmid,
                    "doi": doi,
                    "url": (
                        f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        if pmid
                        else (f"https://doi.org/{doi}" if doi else None)
                    ),
                    "title": item.get("title", "Untitled"),
                    "authors": item.get("authorString", "Unknown"),
                    "year": item.get("pubYear", "Unknown"),
                    "journal": item.get("journalTitle", ""),
                    "cited_by_count": item.get("citedByCount", 0),
                    "abstract": item.get("abstractText", "No abstract available."),
                }
            )
        return papers

    @staticmethod
    def get_curated_evidence(topic: str = "all") -> dict[str, list[dict[str, Any]]]:
        """Return curated peer-reviewed evidence by topic or all topics."""
        topic_key = topic.strip().lower()
        if topic_key in ("all", "", "*"):
            return {
                k: [c.model_dump() for c in v]
                for k, v in CURATED_HYBRID_EVIDENCE.items()
            }
        matched: dict[str, list[dict[str, Any]]] = {}
        for k, v in CURATED_HYBRID_EVIDENCE.items():
            if topic_key in k or any(
                topic_key in c.title.lower() or topic_key in c.key_findings.lower()
                for c in v
            ):
                matched[k] = [c.model_dump() for c in v]
        return matched or {
            k: [c.model_dump() for c in v]
            for k, v in CURATED_HYBRID_EVIDENCE.items()
        }
