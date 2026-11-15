
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="ShopEase AI Support",
    page_icon="💬",
)

st.title("ShopEase AI Support")
st.caption("AI customer support demo using RAG and tools")

# Store chat messages for display across Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        for citation in message.get("citations", {}).values():
            st.caption(
                f"Source: {citation['source']} "
                f"({citation['chunk_id']})"
            )

# Customer enters a new question.
question = st.chat_input("Ask about an order or policy")

if question:
    # Display and store the customer message.
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.write(question)

    # Send the question to our FastAPI backend.
    with st.chat_message("assistant"):
        with st.spinner("Checking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=90,
                )

                response.raise_for_status()
                result = response.json()

                answer = result["answer"]
                citations = result.get("citations", {})

                st.write(answer)

                for citation in citations.values():
                    st.caption(
                        f"Source: {citation['source']} "
                        f"({citation['chunk_id']})"
                    )

            except requests.exceptions.RequestException:
                answer = (
                    "The support service is unavailable. "
                    "Please try again."
                )
                citations = {}
                st.error(answer)

            except (ValueError, KeyError, TypeError):
                answer = (
                    "The support service returned "
                    "an unexpected response."
                )
                citations = {}
                st.error(answer)

    # Store the answer for display on subsequent reruns.
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
    })
