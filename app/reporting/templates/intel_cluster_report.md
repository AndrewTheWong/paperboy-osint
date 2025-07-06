# 🛰️ OSINT Intelligence Cluster Report — {{ cluster_id }}

**Label:** _{{ cluster_label }}_  
**Articles in Cluster:** {{ article_count }}  
**Description:** {{ cluster_description }}

---

## 🔥 Escalation Assessment

- **Avg. Confidence Score:** {{ avg_score }}
- **Escalation Level:** {{ escalation }}

---

## 🏷️ Tag Summary

{% for tag, count in top_tags %}
- {{ tag }} ({{ count }})
{% endfor %}

---

## 🧠 Named Entities

{% for ent, count in ner_entities %}
- {{ ent }} ({{ count }})
{% endfor %}

---

## 📚 Top Articles

{% for a in example_articles %}
**{{ a.title }}** — [source]({{ a.source }})  
_Confidence: {{ a.confidence }}_  
> {{ a.summary }}

---
{% endfor %} 