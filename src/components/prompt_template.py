def build_rag_prompt(context: str, question: str, chat_history: str = "") -> str:
    # Step 1: Prepare the chat history section (only if chat_history is not empty)
    history_section = ""

    if chat_history.strip():  # checks if chat_history has any non-space characters
        history_section = (
            "CHAT HISTORY (prior conversation turns for reference):\n"
            f"{chat_history}\n\n"
        )

    # Step 2: Build the final prompt string (same text and logic as before)
    prompt = (
        f"{history_section}"
        "CONTEXT (retrieved from the knowledge base):\n"
        f"{context}\n\n"
        "INSTRUCTIONS:\n"
        "You are a knowledgeable assistant. Answer the user's question by following these rules in order:\n"
        "1. Use the CONTEXT above as your primary source of information.\n"
        "2. If the CONTEXT alone is insufficient but the CHAT HISTORY contains relevant information\n"
        "   (e.g. the question refers to something discussed earlier), combine both to form a complete answer.\n"
        "3. If neither the CONTEXT nor the CHAT HISTORY contains enough information to answer,\n"
        "   respond with: \"Not found in the knowledge base.\"\n"
        "4. Do not use external knowledge beyond what is provided above.\n"
        "5. Do not guess or hallucinate facts.\n"
        "6. Keep the answer clear and concise.\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER:"
    )

    # Step 3: Return the prompt (strip removes leading/trailing spaces/newlines)
    return prompt.strip()