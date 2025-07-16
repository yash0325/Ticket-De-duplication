import streamlit as st
import pandas as pd
import openai

st.set_page_config(page_title="Ticket Deduplication Demo", layout="wide")
st.title("🕵️‍♂️ Ticket Deduplication Assistant")

# Set your OpenAI API key (from Streamlit secrets for security)
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.markdown("""
Upload your ConnectWise (or other PSA) ticket export (CSV).<br>
Pick any two tickets and see if AI thinks they're describing the same issue!
""", unsafe_allow_html=True)

# --- Upload Ticket CSV ---
uploaded_file = st.file_uploader(
    "Upload ticket export (CSV)", 
    type=["csv"], 
    help="CSV with at least ticket_id, summary, customer_name, type, sub_type, date_entered."
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Clean/standardize columns if needed
    required_cols = ['ticket_id', 'summary', 'customer_name', 'type', 'sub_type', 'date_entered']
    df = df[[col for col in required_cols if col in df.columns]]

    st.write("### All Tickets (showing first 20)")
    st.dataframe(df.head(20), use_container_width=True)

    # --- Pick Two Tickets for Comparison ---
    ticket_options = df['ticket_id'].astype(str).tolist()
    id1 = st.selectbox("Select first ticket:", ticket_options, index=0)
    id2 = st.selectbox("Select second ticket:", ticket_options, index=1 if len(ticket_options) > 1 else 0)

    t1 = df[df['ticket_id'].astype(str) == id1].iloc[0]
    t2 = df[df['ticket_id'].astype(str) == id2].iloc[0]

    st.markdown(f"""
    #### 📝 Summary of Ticket 1 (ID: {t1['ticket_id']}):
    ```
    {t1['summary']}
    ```
    ---
    #### 📝 Summary of Ticket 2 (ID: {t2['ticket_id']}):
    ```
    {t2['summary']}
    ```
    """)
    if st.button("🔍 Compare for Duplicates"):
        # --- Build LLM prompt ---
        prompt = f"""Are the following two ticket summaries describing the same underlying issue? 
Respond YES or NO and provide a similarity score from 0 to 1.

Summary 1:
{t1['summary']}

Summary 2:
{t2['summary']}
"""
        with st.spinner("Asking GPT..."):
            response = openai.ChatCompletion.create(
                model="gpt-4o", # or "gpt-3.5-turbo"
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            output = response["choices"][0]["message"]["content"]

        st.success("**AI Assessment:**")
        st.markdown(f"> {output}")

else:
    st.info("Upload a CSV file to get started!")

