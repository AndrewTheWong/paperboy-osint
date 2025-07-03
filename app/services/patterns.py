#!/usr/bin/env python3
"""
Pattern Matching for OSINT Article Tagging System
Regex patterns to identify specific events, capabilities, and activities
"""

PATTERNS = [
    # Military Events
    (r"live[- ]?fire (drill|exercise)", "event", "Live Fire Drill"),
    (r"airspace (violation|incursion|breach)", "event", "Airspace Violation"),
    (r"missile (launch|test|fire|firing)", "event", "Missile Launch"),
    (r"military (exercise|drill|maneuver)", "event", "Military Exercise"),
    (r"naval (patrol|exercise|drill)", "event", "Naval Patrol"),
    (r"air defense (exercise|drill)", "event", "Air Defense Exercise"),
    (r"joint (exercise|drill|operation)", "event", "Joint Exercise"),
    (r"freedom of navigation", "event", "Freedom of Navigation"),
    (r"port (visit|call)", "event", "Port Visit"),
    (r"diplomatic (protest|complaint)", "event", "Diplomatic Protest"),
    
    # Economic and Trade Events
    (r"economic (coercion|sanctions|pressure)", "security", "Economic Coercion"),
    (r"trade (war|dispute|conflict)", "event", "Trade War"),
    (r"technology (transfer|theft|espionage)", "event", "Technology Transfer"),
    (r"intellectual property (theft|violation)", "event", "Intellectual Property Theft"),
    (r"export (control|ban|restriction)", "policy", "Export Controls"),
    (r"import (ban|restriction|tariff)", "policy", "Import Restrictions"),
    
    # Military Capabilities
    (r"electronic warfare", "capability", "Electronic Warfare"),
    (r"radar (jamming|interference)", "capability", "Radar Jamming"),
    (r"hypersonic (missile|weapon)", "capability", "Hypersonic"),
    (r"anti-ship (missile|weapon)", "capability", "Anti-Ship Missile"),
    (r"air defense (system|missile)", "capability", "Air Defense"),
    (r"ballistic missile", "capability", "Ballistic Missile"),
    (r"cruise missile", "capability", "Cruise Missile"),
    (r"cyber (warfare|attack)", "capability", "Cyber Warfare"),
    (r"information warfare", "capability", "Information Warfare"),
    (r"cognitive warfare", "capability", "Cognitive Warfare"),
    
    # Surveillance and Intelligence
    (r"surveillance (system|operation)", "security", "Surveillance"),
    (r"intelligence (gathering|collection)", "capability", "ISR"),
    (r"reconnaissance (mission|flight)", "capability", "ISR"),
    (r"spy (plane|aircraft|satellite)", "capability", "ISR"),
    
    # Maritime Activities
    (r"exclusive economic zone", "geo", "Exclusive Economic Zone"),
    (r"nine-dash line", "geo", "Nine-Dash Line"),
    (r"maritime (rights|claims)", "policy", "Maritime Rights Protection"),
    (r"territorial (waters|sea)", "geo", "Territorial Waters"),
    (r"fishing (vessel|boat|fleet)", "actor", "Fishing Fleet"),
    (r"coast guard", "actor", "Coast Guard"),
    
    # Political and Policy
    (r"one china policy", "policy", "One China Policy"),
    (r"belt and road", "policy", "Belt and Road Initiative"),
    (r"made in china 2025", "policy", "Made in China 2025"),
    (r"dual circulation", "policy", "Dual Circulation"),
    (r"common prosperity", "policy", "Common Prosperity"),
    (r"zero covid", "policy", "Zero COVID"),
    (r"real estate crackdown", "policy", "Real Estate Crackdown"),
    
    # Diplomatic Relations
    (r"wolf warrior (diplomacy|rhetoric)", "diplomacy", "Wolf Warrior Diplomacy"),
    (r"panda diplomacy", "diplomacy", "Panda Diplomacy"),
    (r"checkbook diplomacy", "diplomacy", "Checkbook Diplomacy"),
    (r"debt trap diplomacy", "diplomacy", "Debt Trap Diplomacy"),
    (r"soft power", "diplomacy", "Soft Power"),
    (r"hard power", "diplomacy", "Hard Power"),
    
    # Technology and Innovation
    (r"artificial intelligence", "technology", "Artificial Intelligence"),
    (r"machine learning", "technology", "Machine Learning"),
    (r"quantum (computing|research)", "technology", "Quantum Research"),
    (r"5g (network|technology)", "technology", "5G"),
    (r"6g (network|technology)", "technology", "6G"),
    (r"big data", "technology", "Big Data"),
    (r"cloud computing", "technology", "Cloud Computing"),
    (r"blockchain", "technology", "Blockchain"),
    (r"cybersecurity", "technology", "Cybersecurity"),
    (r"space (technology|program)", "technology", "Space Technology"),
    
    # Economic Terms
    (r"semiconductor (industry|chip)", "economy", "Semiconductors"),
    (r"debt crisis", "economy", "Debt Crisis"),
    (r"foreign investment", "economy", "Foreign Investment"),
    (r"trade deficit", "economy", "Trade Deficit"),
    (r"currency manipulation", "economy", "Currency Manipulation"),
    (r"supply chain", "economy", "Supply Chain"),
    (r"rare earth (elements|minerals)", "economy", "Rare Earth Elements"),
    (r"market access", "economy", "Market Access"),
    
    # Legal and Regulatory
    (r"national security law", "law", "National Security Law"),
    (r"data privacy law", "law", "Data Privacy Law"),
    (r"cybersecurity law", "law", "Cybersecurity Law"),
    (r"foreign investment law", "law", "Foreign Investment Law"),
    (r"export control law", "law", "Export Control Law"),
    (r"anti-foreign sanctions law", "law", "Anti-Foreign Sanctions Law"),
    (r"unclos", "law", "UNCLOS"),
    (r"maritime law", "law", "Maritime Law"),
    (r"trade law", "law", "Trade Law"),
    
    # Social and Ideological
    (r"social credit system", "society", "Social Credit System"),
    (r"patriotic education", "ideology", "Patriotic Education"),
    (r"xi thought", "ideology", "Xi Thought"),
    (r"socialism with chinese characteristics", "ideology", "Socialism with Chinese Characteristics"),
    (r"chinese dream", "ideology", "Chinese Dream"),
    (r"youth unemployment", "society", "Youth Unemployment"),
    (r"covid-19 lockdown", "society", "COVID-19 Lockdown"),
    (r"censorship", "society", "Censorship"),
    (r"human rights", "society", "Human Rights"),
    (r"democracy", "society", "Democracy"),
    (r"authoritarianism", "society", "Authoritarianism"),
    (r"nationalism", "society", "Nationalism"),
    (r"patriotism", "society", "Patriotism")
] 