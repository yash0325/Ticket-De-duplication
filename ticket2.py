import streamlit as st
import pandas as pd
import openai
from datetime import timedelta

# Setup OpenAI
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="AI Ticket Deduplication", layout="wide")
st.title("🕵️‍♂️ AI Ticket Deduplication Assistant")

st.markdown("""
Upload your ConnectWise/PSA ticket export (CSV).  
This tool will find and group likely duplicate tickets (within the last 10 minutes) by parent (oldest) and children (duplicates), using structured logic and GPT-4o for semantic matching.
""")

# --- Upload Ticket CSV ---
uploaded_file = st.file_uploader(
    "Upload ticket export (CSV)", 
    type=["csv"],
    help="Must include: ticket_id, summary, customer_name, type, sub_type, date_entered"
)

def llm_semantic_similarity(text1, text2, threshold=0.75):
    prompt = (
        f"Are the following two ticket summaries describing the same underlying issue? "
        f"Respond YES or NO and provide a similarity score from 0 to 1.\n\n"
        f"Summary 1:\n{text1}\n\nSummary 2:\n{text2}\n"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()
    yes_no = "NO"
    score = 0.0
    import re
    if "YES" in answer.upper():
        yes_no = "YES"
    found = re.findall(r"([0-1]\.\d{1,2})", answer)
    if found:
        score = float(found[0])
    return yes_no == "YES" and score >= threshold, score

def detect_duplicates(df, time_window_min=10, sim_threshold=0.75):
    df['date_entered'] = pd.to_datetime(df['date_entered'])
    now = df['date_entered'].max()
    recent = df[df['date_entered'] >= now - pd.Timedelta(minutes=time_window_min)].copy()

    group_fields = ['customer_name', 'type', 'sub_type']
    parent_ids = []
    child_to_parent = {}
    already_linked = set()

    for _, group in recent.groupby(group_fields):
        tickets = group.sort_values("date_entered").to_dict('records')
        n = len(tickets)
        if n == 1:
            parent_ids.append(tickets[0]['ticket_id'])
            continue
        parent = tickets[0]
        parent_ids.append(parent['ticket_id'])
        for child in tickets[1:]:
            # LLM check for semantic similarity
            is_dup, sim = llm_semantic_similarity(parent['summary'], child['summary'], threshold=sim_threshold)
            if is_dup:
                child_to_parent[child['ticket_id']] = parent['ticket_id']
                already_linked.add(child['ticket_id'])
            else:
                # Not duplicate: treat as a new parent
                parent_ids.append(child['ticket_id'])

    deduped = df[df['ticket_id'].isin(parent_ids)].copy()
    return deduped, child_to_parent

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### All Tickets (showing first 20):")
    st.dataframe(df.head(20), use_container_width=True)

    if st.button("🚦 Run Deduplication"):
        with st.spinner("Detecting duplicates with GPT... (this may take a moment)"):
            deduped, child_to_parent = detect_duplicates(df)
        st.success("Deduplication Complete!")

        st.write("### Parent Tickets (to be sent to triage):")
        st.dataframe(deduped, use_container_width=True)

        if child_to_parent:
            st.write("### Parent/Child Mapping (duplicates found):")
            mapping_df = pd.DataFrame([
                {"Child Ticket ID": k, "Parent Ticket ID": v}
                for k, v in child_to_parent.items()
            ])
            st.dataframe(mapping_df, use_container_width=True)
        else:
            st.info("No duplicates detected in the selected time window.")

else:
    st.info("Upload a CSV file to get started!")
