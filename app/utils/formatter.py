def format_digest(grouped: dict) -> str:
    """Format grouped digest as markdown"""
    lines = []
    for region, topics in grouped.items():
        lines.append(f"## {region}\n")
        for topic, summaries in topics.items():
            lines.append(f"### {topic}")
            for summary in summaries:
                lines.append(f"- {summary}")
            lines.append("")
    return "\n".join(lines) 