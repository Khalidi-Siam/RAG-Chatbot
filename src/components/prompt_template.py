def build_rag_prompt(context: str, question: str, chat_history: str = "") -> str:
    # Step 1: Prepare the chat history section (only if chat_history is not empty)
    history_section = ""

    if chat_history.strip():
        history_section = (
            "CHAT HISTORY (prior conversation turns for reference):\n"
            f"{chat_history}\n\n"
        )

    # Step 2: Build the final prompt string
    prompt = (
        f"{history_section}"
        "CONTEXT (retrieved from the knowledge base):\n"
        f"{context}\n\n"
        "INSTRUCTIONS:\n"
        "You are a knowledgeable assistant. Answer the user's question by following these rules in order:\n"
        "1. Before answering, check if the question contains unresolved references — words like\n"
        "   'that', 'it', 'this', 'those', 'the incident', 'the event', 'he', 'she', 'they', etc.\n"
        "   that point to something not explicitly named in the question itself.\n"
        "2. If such references exist AND there is no CHAT HISTORY to resolve what they refer to,\n"
        "   do NOT attempt to answer. Instead respond with:\n"
        "   \"Your question refers to something (e.g. 'that incident') that hasn't been identified.\n"
        "    Could you clarify what you are referring to?\"\n"
        "3. If such references exist BUT the CHAT HISTORY clearly resolves what they refer to,\n"
        "   substitute the resolved meaning and proceed to answer.\n"
        "4. Use the CONTEXT above as your primary source of information.\n"
        "5. If the CONTEXT alone is insufficient but the CHAT HISTORY contains relevant information,\n"
        "   combine both to form a complete answer.\n"
        "6. If neither the CONTEXT nor the CHAT HISTORY contains enough information to answer,\n"
        "   respond with: \"Not found in the knowledge base.\"\n"
        "7. Do not use external knowledge beyond what is provided above.\n"
        "8. Do not guess or hallucinate facts.\n"
        "9. Keep the answer clear and concise.\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER:"
    )

    # Step 3: Return the prompt
    return prompt.strip()