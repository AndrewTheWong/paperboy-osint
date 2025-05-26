import streamlit as st
from storage.db import get_unprocessed_osint_entries, update_osint_tags
import pandas as pd
from datetime import datetime

def review_tags_ui():
    """Streamlit UI for reviewing and updating tags for OSINT entries."""
    st.set_page_config(page_title="OSINT Tagging Review", layout="wide")
    
    st.title("OSINT Tagging Review")
    st.markdown("Review and update tags for incoming OSINT data entries.")
    
    # Fetch untagged entries
    data = get_unprocessed_osint_entries(limit=20)
    
    if not data:
        st.success("🎉 No entries to tag! All caught up!")
        return
    
    st.info(f"Found {len(data)} entries that need review.")
    
    # Available tags
    available_tags = [
        "military movement", "diplomatic meeting", "conflict", "cyberattack",
        "protest", "nuclear", "ceasefire", "election", "economic", "health crisis"
    ]
    
    # Process each entry
    for i, entry in enumerate(data):
        with st.expander(f"#{i+1} - Source: {entry.get('source_url', 'Unknown')}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                content = entry.get('content', '')
                if content:
                    # Display only first 1000 chars to avoid cluttering the UI
                    display_content = content[:1000] + "..." if len(content) > 1000 else content
                    st.markdown(f"**Content:**\n\n{display_content}")
                else:
                    st.warning("No content available.")
            
            with col2:
                # Pre-select existing tags if any
                existing_tags = entry.get("tags", [])
                
                # Display form for tagging
                with st.form(key=f"form_{entry.get('id', i)}"):
                    selected_tags = st.multiselect(
                        "Select Tags", 
                        options=available_tags,
                        default=existing_tags
                    )
                    
                    confidence = st.slider(
                        "Confidence Score", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=float(entry.get("confidence_score") or 0.5),
                        step=0.01
                    )
                    
                    submitted = st.form_submit_button("Submit Tags")
                    
                    if submitted:
                        try:
                            update_osint_tags(entry.get('id'), selected_tags, confidence)
                            st.success("✅ Tags updated successfully!")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error updating tags: {str(e)}")

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    import os
    import sys
    
    try:
        # Get the path to the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dashboard_path = os.path.join(current_dir, "tag_ui.py")
        
        # Run the Streamlit app
        cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
        process = subprocess.Popen(cmd)
        
        print(f"Launched dashboard at http://localhost:8501")
        return process
    except Exception as e:
        print(f"Error launching dashboard: {str(e)}")
        return None

if __name__ == "__main__":
    review_tags_ui() 