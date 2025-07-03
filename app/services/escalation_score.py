#!/usr/bin/env python3
"""
Escalation Score Calculation for OSINT Article Tagging System
Computes confidence scores based on identified tags and their escalation weights
"""

ESCALATION_WEIGHTS = {
    # High Escalation Events (0.8-1.0)
    "Live Fire Drill": 0.9,
    "Airspace Violation": 0.85,
    "Missile Launch": 0.95,
    "Missile Test": 0.9,
    "Military Exercise": 0.7,
    "Naval Patrol": 0.6,
    "Air Defense Exercise": 0.75,
    "Joint Exercise": 0.8,
    "Freedom of Navigation": 0.65,
    "Diplomatic Protest": 0.5,
    "Economic Coercion": 0.8,
    "Trade War": 0.75,
    "Technology Transfer": 0.6,
    "Intellectual Property Theft": 0.7,
    
    # Military Capabilities (0.6-0.8)
    "Electronic Warfare": 0.75,
    "Radar Jamming": 0.7,
    "Hypersonic": 0.85,
    "Anti-Ship Missile": 0.8,
    "Air Defense": 0.65,
    "Ballistic Missile": 0.9,
    "Cruise Missile": 0.8,
    "Cyber Warfare": 0.7,
    "Information Warfare": 0.6,
    "Cognitive Warfare": 0.65,
    "ISR": 0.5,
    "Surveillance": 0.55,
    
    # Geographic Locations (0.3-0.6)
    "South China Sea": 0.4,
    "Spratly Islands": 0.5,
    "Paracel Islands": 0.5,
    "Scarborough Shoal": 0.5,
    "Taiwan": 0.6,
    "China": 0.3,
    "Philippines": 0.4,
    "Vietnam": 0.4,
    "Japan": 0.4,
    "South Korea": 0.4,
    "East China Sea": 0.4,
    "Strait of Taiwan": 0.6,
    "Exclusive Economic Zone": 0.4,
    "Nine-Dash Line": 0.5,
    "Territorial Waters": 0.4,
    
    # Military Actors (0.5-0.7)
    "PLA Navy": 0.6,
    "Taiwan MND": 0.5,
    "U.S. 7th Fleet": 0.6,
    "Eastern Theater Command": 0.6,
    "Southern Theater Command": 0.6,
    "People's Liberation Army": 0.6,
    "Taiwan Armed Forces": 0.5,
    "U.S. Indo-Pacific Command": 0.6,
    "Japanese Self-Defense Forces": 0.5,
    "Philippine Navy": 0.4,
    "Vietnamese Navy": 0.4,
    "Coast Guard": 0.4,
    "Fishing Fleet": 0.3,
    
    # Political Leaders (0.2-0.4)
    "Xi Jinping": 0.2,
    "Lai Ching-te": 0.3,
    "Tsai Ing-wen": 0.3,
    "Wang Yi": 0.2,
    "Li Shangfu": 0.3,
    "Joe Biden": 0.2,
    "Lloyd Austin": 0.3,
    "Antony Blinken": 0.2,
    
    # Military Units (0.4-0.6)
    "74th Group Army": 0.5,
    "PLAN SSF": 0.5,
    "PLAN East Sea Fleet": 0.5,
    "PLAN South Sea Fleet": 0.5,
    "PLA Air Force": 0.5,
    "PLA Rocket Force": 0.6,
    "PLA Strategic Support Force": 0.5,
    "Taiwan Army": 0.4,
    "Taiwan Air Force": 0.4,
    "Taiwan Navy": 0.4,
    "U.S. Pacific Fleet": 0.5,
    
    # Military Platforms (0.4-0.7)
    "J-20": 0.6,
    "J-16": 0.5,
    "J-15": 0.5,
    "J-10": 0.4,
    "Y-8 EW": 0.6,
    "Y-9 EW": 0.6,
    "H-6": 0.5,
    "Y-20": 0.6,
    "Type 055": 0.7,
    "Type 052D": 0.6,
    "Type 054A": 0.5,
    "Type 056": 0.4,
    "Type 075": 0.6,
    "Type 003": 0.7,
    "F-16": 0.5,
    "F-35": 0.6,
    "P-8 Poseidon": 0.5,
    "E-2 Hawkeye": 0.5,
    "Arleigh Burke": 0.6,
    "Ticonderoga": 0.6,
    
    # Policy and Economic (0.3-0.6)
    "One China Policy": 0.4,
    "Belt and Road Initiative": 0.3,
    "Made in China 2025": 0.4,
    "Dual Circulation": 0.3,
    "Common Prosperity": 0.3,
    "Zero COVID": 0.3,
    "Real Estate Crackdown": 0.3,
    "Export Controls": 0.5,
    "Import Restrictions": 0.5,
    "Maritime Rights Protection": 0.4,
    
    # Diplomatic Relations (0.3-0.6)
    "Wolf Warrior Diplomacy": 0.5,
    "Panda Diplomacy": 0.3,
    "Checkbook Diplomacy": 0.4,
    "Debt Trap Diplomacy": 0.5,
    "Soft Power": 0.3,
    "Hard Power": 0.5,
    
    # Technology (0.4-0.6)
    "Artificial Intelligence": 0.5,
    "Machine Learning": 0.4,
    "Quantum Research": 0.6,
    "5G": 0.5,
    "6G": 0.5,
    "Big Data": 0.4,
    "Cloud Computing": 0.4,
    "Blockchain": 0.4,
    "Cybersecurity": 0.5,
    "Space Technology": 0.6,
    "AI Surveillance": 0.6,
    
    # Economic Terms (0.3-0.5)
    "Semiconductors": 0.5,
    "Debt Crisis": 0.4,
    "Foreign Investment": 0.3,
    "Trade Deficit": 0.3,
    "Currency Manipulation": 0.5,
    "Supply Chain": 0.4,
    "Rare Earth Elements": 0.5,
    "Market Access": 0.3,
    
    # Legal and Regulatory (0.3-0.5)
    "National Security Law": 0.5,
    "Data Privacy Law": 0.4,
    "Cybersecurity Law": 0.4,
    "Foreign Investment Law": 0.3,
    "Export Control Law": 0.5,
    "Anti-Foreign Sanctions Law": 0.5,
    "UNCLOS": 0.4,
    "Maritime Law": 0.4,
    "Trade Law": 0.3,
    
    # Social and Ideological (0.2-0.4)
    "Social Credit System": 0.4,
    "Patriotic Education": 0.3,
    "Xi Thought": 0.2,
    "Socialism with Chinese Characteristics": 0.2,
    "Chinese Dream": 0.2,
    "Youth Unemployment": 0.3,
    "COVID-19 Lockdown": 0.3,
    "Censorship": 0.4,
    "Human Rights": 0.3,
    "Democracy": 0.3,
    "Authoritarianism": 0.3,
    "Nationalism": 0.3,
    "Patriotism": 0.2
}

def compute_confidence_score(flat_tags: list) -> float:
    """
    Compute confidence score based on identified tags
    
    Args:
        flat_tags: List of tags in format "CATEGORY:Value"
        
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    score = 0.0
    tag_count = len(flat_tags)
    
    # Base score from tag weights
    for tag in flat_tags:
        if ':' in tag:
            value = tag.split(":")[-1]
            score += ESCALATION_WEIGHTS.get(value, 0.0)
    
    # Normalize by number of tags (more tags = higher confidence)
    if tag_count > 0:
        score = score / tag_count
    
    # Apply bonus for multiple high-value tags
    high_value_tags = [tag for tag in flat_tags if ':' in tag and 
                      ESCALATION_WEIGHTS.get(tag.split(":")[-1], 0.0) > 0.7]
    if len(high_value_tags) > 1:
        score += 0.1 * len(high_value_tags)
    
    # Cap at 1.0
    return min(score, 1.0)

def get_priority_level(confidence_score: float) -> str:
    """
    Convert confidence score to priority level
    
    Args:
        confidence_score: Score between 0.0 and 1.0
        
    Returns:
        str: Priority level (HIGH, MEDIUM, LOW)
    """
    if confidence_score >= 0.7:
        return "HIGH"
    elif confidence_score >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def get_escalation_emoji(confidence_score: float) -> str:
    """
    Get emoji for escalation score
    
    Args:
        confidence_score: Score between 0.0 and 1.0
        
    Returns:
        str: Emoji representation
    """
    if confidence_score >= 0.8:
        return "🚨"
    elif confidence_score >= 0.6:
        return "⚠️"
    elif confidence_score >= 0.4:
        return "📈"
    else:
        return "📊" 