#!/usr/bin/env python3
"""
Utility functions and data structures for article tagging.
"""
from typing import Dict, List, Any, Optional

# Keyword to topic mapping dictionary
KEYWORD_MAP = {
    "military": ["army", "drill", "exercise", "pla", "soldier", "military", "troops", "forces", "defense", "missile"],
    "naval": ["navy", "vessel", "ship", "fleet", "maritime", "submarine", "carrier", "warship", "frigate", "destroyer"],
    "aerospace": ["aircraft", "fighter", "bomber", "j-20", "jet", "aviation", "airspace", "runway", "radar", "plane"],
    "missile": ["missile", "icbm", "ballistic", "hypersonic", "launch", "test", "nuclear", "warhead", "rocket"],
    "coast guard": ["coast guard", "patrol", "territorial waters", "exclusive economic zone", "eez"],
    "intelligence": ["intelligence", "spy", "espionage", "surveillance", "classified", "secret", "leak"],
    "nuclear": ["nuclear", "atomic", "warhead", "deterrent", "proliferation", "reactor", "enrichment"],
    "diplomacy": ["diplomat", "embassy", "envoy", "foreign minister", "state department", "bilateral", "multilateral"],
    "alliances": ["alliance", "nato", "quad", "aukus", "partner", "coalition", "pact", "treaty"],
    "treaty": ["treaty", "agreement", "pact", "convention", "accord", "protocol", "memorandum"],
    "summit": ["summit", "meeting", "conference", "talks", "dialogue", "forum", "g7", "g20", "asean"],
    "sanctions": ["sanction", "embargo", "restriction", "blacklist", "ban", "export control"],
    "arms sale": ["arms sale", "weapon sale", "defense contract", "military aid", "security assistance"],
    "protest": ["protest", "demonstration", "riot", "unrest", "dissent", "opposition", "rally"],
    "riot": ["riot", "unrest", "clash", "violence", "police", "tear gas", "disperse"],
    "election": ["election", "vote", "ballot", "campaign", "polling", "candidate", "democracy"],
    "civil society": ["civil society", "ngo", "activist", "advocacy", "human rights", "freedom"],
    "governance": ["governance", "policy", "regulation", "reform", "administration", "government"],
    "law": ["law", "legislation", "bill", "court", "judiciary", "legal", "constitution"],
    "trade": ["trade", "export", "import", "tariff", "commerce", "market", "economy", "business"],
    "tariff": ["tariff", "duty", "tax", "customs", "protectionism", "trade barrier"],
    "investment": ["investment", "investor", "fund", "finance", "capital", "fdi", "stock"],
    "semiconductors": ["semiconductor", "chip", "microchip", "tsmc", "intel", "fab", "foundry"],
    "technology": ["technology", "tech", "innovation", "digital", "ai", "artificial intelligence", "quantum"],
    "infrastructure": ["infrastructure", "construction", "project", "development", "belt and road", "port", "railway"],
    "cyber": ["cyber", "hack", "breach", "malware", "ransomware", "network", "server", "attack"],
    "surveillance": ["surveillance", "monitor", "track", "facial recognition", "camera", "privacy"],
    "espionage": ["espionage", "spy", "intelligence", "agent", "classified", "secret", "leak"],
    "disinformation": ["disinformation", "misinformation", "fake news", "propaganda", "influence operation"],
    "AI": ["ai", "artificial intelligence", "machine learning", "algorithm", "neural network"],
    "hack": ["hack", "breach", "intrusion", "compromise", "vulnerability", "exploit", "attack"],
    "maritime": ["maritime", "sea", "ocean", "territorial waters", "exclusive economic zone", "eez"],
    "spratly": ["spratly", "paracel", "island", "reef", "atoll", "south china sea"],
    "fishing fleet": ["fishing", "trawler", "boat", "vessel", "maritime militia"],
    "air defense zone": ["adiz", "air defense", "identification zone", "airspace"],
    "incursion": ["incursion", "intrusion", "violation", "trespass", "breach", "cross"],
    "taiwan": ["taiwan", "taiwanese", "taipei", "cross-strait", "dpp", "kmt"],
    "south china sea": ["south china sea", "disputed waters", "nine-dash line", "paracel", "spratly"],
    "okinawa": ["okinawa", "ryukyu", "japanese island", "us base"],
    "philippines": ["philippines", "philippine", "manila", "duterte", "marcos"],
    "japan": ["japan", "japanese", "tokyo", "kishida", "self-defense force", "jsdf"],
    "guam": ["guam", "mariana", "us territory", "pacific island"],
    "escalatory": ["escalation", "provocation", "tension", "conflict", "crisis", "warning"],
    "conciliatory": ["conciliation", "cooperation", "dialogue", "peace", "stability", "negotiation"],
    "neutral": ["neutral", "balanced", "impartial", "objective", "factual"],
    "cooperative": ["cooperation", "partnership", "collaboration", "joint", "mutual", "shared"],
    "threatening": ["threat", "warning", "intimidation", "coercion", "pressure", "force"]
}

def extract_tags_from_text(text: str) -> Dict[str, int]:
    """
    Extract tags from text based on keyword matches.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary mapping tags to their occurrence count
    """
    text = text.lower()
    tag_counts = {}
    
    for tag, keywords in KEYWORD_MAP.items():
        count = 0
        for keyword in keywords:
            keyword = keyword.lower()
            count += text.count(keyword)
        
        if count > 0:
            tag_counts[tag] = count
    
    return tag_counts

def needs_human_review(tags: List[str]) -> bool:
    """
    Determine if an article needs human review based on its tags.
    
    Args:
        tags: List of tags assigned to the article
        
    Returns:
        True if the article needs human review, False otherwise
    """
    # If no tags or unknown tag, needs review
    if not tags or "unknown" in tags:
        return True
    
    # If fewer than 2 tags, needs review
    if len(tags) < 2:
        return True
    
    return False 