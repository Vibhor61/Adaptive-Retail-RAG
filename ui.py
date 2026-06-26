import streamlit as st
import requests
import json
from config.settings import settings

BACKEND_URL = settings.backend_url
STREAM_URL = settings.stream_url
PHOENIX_URL = settings.phoenix_url_ui

st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")

# Init session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

total_docs = sum(
    len(msg.get("metadata", {}).get("retrieved_contexts", []))
    for msg in st.session_state.messages if "metadata" in msg
)

with st.sidebar:
    st.title("Session")
    if st.session_state.session_id:
        st.caption(f"ID: {st.session_state.session_id}")
    else:
        st.caption("New Session")

    st.divider()

    st.subheader("Query History")
    history_count = 0
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"- {msg['content']}")
            history_count += 1
    if history_count == 0:
        st.caption("No queries yet.")

    st.divider()

    st.subheader("Retrieved Documents")
    st.metric(label="Total Contexts Fetched", value=total_docs)

    st.divider()
    if PHOENIX_URL:
        st.markdown(f"🔍 [View Phoenix Traces]({PHOENIX_URL})")

st.title("🛍️ Retail AI Assistant")

if not st.session_state.messages:
    st.divider()
    
    st.markdown("### ✨ Welcome to the Retail RAG Assistant!")
    st.markdown(
        "I am a specialized assistant built for answering queries on **Cell Phones & Accessories** data. "
        "You can compare products, check specifications, or retrieve grounded user reviews."
    )
    st.markdown("**Examples of what you can ask:**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔍 Specifications & Options**")
        st.markdown("- *'What wireless chargers do you have for iPhone?'*")
        st.markdown("- *'Does this phone case have drop protection?'*")
    with col2:
        st.markdown("**💬 Review Summaries & Sentiments**")
        st.markdown("- *'What are the pros and cons of the Galaxy S22 Ultra based on reviews?'*")
        st.markdown("- *'What is the general feedback about the screen protector brand?'*")
    st.divider()

# ── Render existing conversation history ──────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]

            contexts = meta.get("retrieved_contexts", [])
            if contexts:
                with st.expander(f"Retrieved Contexts ({len(contexts)})"):
                    for i, ctx in enumerate(contexts, 1):
                        st.markdown(f"**Context {i}:**")
                        st.text(ctx)
                        st.divider()

            citations = meta.get("citations", [])
            if citations:
                with st.expander(f"Citations ({len(citations)})"):
                    for cite in citations:
                        st.json(cite)

            control = meta.get("control", {})
            if control:
                with st.expander("Controller Decisions"):
                    st.json(control)

# ── Handle new user input ─────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        metadata = {}

        try:
            req_data = {"query": prompt}
            if st.session_state.session_id:
                req_data["session_id"] = st.session_state.session_id

            response = requests.post(
                STREAM_URL,
                json=req_data,
                stream=True,
                timeout=120,
            )
            response.encoding = "utf-8"

            if response.status_code != 200:
                st.error(f"Error {response.status_code}: {response.text}")
            else:
                # iter_lines() handles SSE framing correctly:
                # each "data: <payload>\n\n" becomes one line.
                for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data: "):
                        continue

                    data_str = raw_line[6:]
                    try:
                        token = json.loads(data_str)
                    except json.JSONDecodeError:
                        token = data_str

                    if isinstance(token, dict):
                        if token.get("type") == "metadata":
                            metadata = token
                        elif token.get("type") == "error":
                            st.error(
                                f"Pipeline error — {token.get('error')}: {token.get('detail')}"
                            )
                    elif isinstance(token, str) and token:
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")

                # Remove cursor blink once streaming is done
                response_placeholder.markdown(full_response)

                if metadata:
                    if metadata.get("session_id"):
                        st.session_state.session_id = metadata["session_id"]

                    contexts = metadata.get("retrieved_contexts", [])
                    if contexts:
                        with st.expander(f"Retrieved Contexts ({len(contexts)})"):
                            for i, ctx in enumerate(contexts, 1):
                                st.markdown(f"**Context {i}:**")
                                st.text(ctx)
                                st.divider()

                    citations = metadata.get("citations", [])
                    if citations:
                        with st.expander(f"Citations ({len(citations)})"):
                            for cite in citations:
                                st.json(cite)

                    control = metadata.get("control", {})
                    if control:
                        with st.expander("Controller Decisions"):
                            st.json(control)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "metadata": metadata,
                })

                st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("Connection error — is the backend running at " + STREAM_URL + "?")
        except requests.exceptions.Timeout:
            st.error("Request timed out after 120 s.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
