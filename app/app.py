import torch
import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# 1. Load an industry-standard, ultra-lightweight sentence embedding model
# This runs fully locally, uses minimal memory, and understands temple names flawlessly
print("Loading high-performance 22M scale embedding encoder...")
encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Read your pilgrimage dataset CSV file
print("Reading pilgrimage data from CSV...")
df = pd.read_csv("pilgrimage_dataset.csv")
faq_questions = df["context_question"].tolist()
faq_answers = df["verified_answer"].tolist()

# 3. Generate mathematically distinct search indexes for your database
print("Generating search indexes for pilgrimage database...")
faq_embeddings = encoder_model.encode(faq_questions, convert_to_tensor=True)

# --- CHATBOT CORE INTELLIGENCE ---
class PilgrimageChatbot:
    def __init__(self):
        self.chat_history = []

    def process_message(self, user_input):
        if not user_input.strip():
            return "Please type a question about Indian pilgrimage centres."

        # Maintain natural conversation context across consecutive questions
        if self.chat_history:
            last_intent = self.chat_history[-1]['matched_intent']
            search_query = f"{last_intent} [SEP] {user_input}"
        else:
            search_query = user_input

        # Convert user intent into a precise mathematical vector
        query_embedding = encoder_model.encode(search_query, convert_to_tensor=True)

        # Compute semantic similarity scores using the proper framework matching tool
        cos_scores = util.cos_sim(query_embedding, faq_embeddings)[0]
        
        best_match_idx = torch.argmax(cos_scores).item()
        confidence = cos_scores[best_match_idx].item()

        print(f"User Input: '{user_input}' -> Best Match Confidence: {confidence:.4f}")

        # Strict safety margin: stops the bot from saying random things if outside the topic
        if confidence < 0.45:
            return "I am specialized only in famous Indian Pilgrimage Centres. I couldn't find a confident match for that question. Could you try asking about Tirupati, Varanasi, or the Char Dham?"

        # Map index directly to verified text rows
        bot_response = faq_answers[best_match_idx]
        matched_question = faq_questions[best_match_idx]

        # Update historical state tracking
        self.chat_history.append({"user_query": user_input, "matched_intent": matched_question})
        return bot_response

# --- WEB INTERFACE LAUNCHER ---
bot_instance = PilgrimageChatbot()

def chat_wrapper(message, history):
    return bot_instance.process_message(message)

demo = gr.ChatInterface(
    fn=chat_wrapper,
    title="🕉️ Indian Pilgrimage Assistant (Fixed Encoder)",
    description="Factual, reliable, and ultra-fast chatbot using a highly accurate local semantic search engine.",
    examples=["Where is Tirupati Balaji?", "Best time to visit Kashi?", "Tell me about Meenakshi Temple in Madurai."]
)

if __name__ == "__main__":
    demo.launch(share=False)
