"""
Integration Examples - How to use RAG in your applications
"""

# Example 1: Simple Integration with Chatbot
def chatbot_with_rag():
    """Integrate RAG with a chatbot"""
    from rag_pipeline import RAGPipeline
    
    rag = RAGPipeline()
    rag.ingest_documents()
    
    # Simulate chatbot conversation
    user_messages = [
        "What is this research about?",
        "Tell me more about the methodology",
        "What are the conclusions?"
    ]
    
    for user_query in user_messages:
        print(f"\nUser: {user_query}")
        answer = rag.query(user_query)
        print(f"Assistant: {answer}")


# Example 2: FastAPI Integration
def fastapi_rag_api():
    """Create a REST API with RAG"""
    
    code = """
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from rag_pipeline import RAGPipeline
import os

app = FastAPI()
rag = RAGPipeline()

@app.on_event("startup")
async def startup_event():
    \"\"\"Load documents on startup\"\"\"
    rag.ingest_documents()

@app.post("/query")
async def query(question: str):
    \"\"\"Process a query\"\"\"
    answer = rag.query(question)
    return {"question": question, "answer": answer}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    \"\"\"Upload a new PDF\"\"\"
    file_path = rag.pdf_loader.save_uploaded_pdf(
        await file.read(), 
        file.filename
    )
    # Add to vector store
    docs = rag.pdf_loader.load_pdf(file_path)
    rag.vector_store.add_documents(docs)
    return {"status": "success", "file": file.filename}

@app.get("/stats")
async def get_stats():
    \"\"\"Get pipeline statistics\"\"\"
    return rag.vector_store.get_vector_store_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    print(code)


# Example 3: Discord Bot with RAG
def discord_bot_with_rag():
    """Create a Discord bot with RAG capabilities"""
    
    code = """
import discord
from discord.ext import commands
from rag_pipeline import RAGPipeline

bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())
rag = RAGPipeline()

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    rag.ingest_documents()

@bot.command(name='ask')
async def ask_question(ctx, *, question):
    \"\"\"Ask a question about the documents\"\"\"
    async with ctx.typing():
        answer = rag.query(question)
    
    # Split long answers
    if len(answer) > 2000:
        chunks = [answer[i:i+2000] for i in range(0, len(answer), 2000)]
        for chunk in chunks:
            await ctx.send(chunk)
    else:
        await ctx.send(answer)

@bot.command(name='stats')
async def show_stats(ctx):
    \"\"\"Show RAG statistics\"\"\"
    stats = rag.vector_store.get_vector_store_stats()
    await ctx.send(f"**Vector Store Stats**\\n```{stats}```")

bot.run('YOUR_DISCORD_TOKEN')
"""
    print(code)


# Example 4: Telegram Bot with RAG
def telegram_bot_with_rag():
    """Create a Telegram bot with RAG"""
    
    code = """
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Ask me anything about the research documents!")

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = ' '.join(context.args)
    
    if not question:
        await update.message.reply_text("Please provide a question")
        return
    
    await update.message.chat.send_action("typing")
    answer = rag.query(question)
    await update.message.reply_text(answer)

async def main():
    rag.ingest_documents()
    
    application = Application.builder().token("YOUR_TELEGRAM_TOKEN").build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("ask", ask))
    
    await application.run_polling()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
"""
    print(code)


# Example 5: Multi-User RAG System
def multi_user_rag():
    """RAG system with per-user knowledge bases"""
    
    code = """
from rag_pipeline import RAGPipeline
import os

class MultiUserRAG:
    def __init__(self):
        self.user_rags = {}
    
    def get_rag_for_user(self, user_id: str) -> RAGPipeline:
        \"\"\"Get or create RAG instance for user\"\"\"
        if user_id not in self.user_rags:
            # Create separate vector store per user
            os.environ['VECTOR_STORE_PATH'] = f'./vector_store_{user_id}'
            os.environ['PDF_UPLOAD_FOLDER'] = f'./uploads_{user_id}'
            self.user_rags[user_id] = RAGPipeline()
        
        return self.user_rags[user_id]
    
    def query_for_user(self, user_id: str, question: str) -> str:
        \"\"\"Process query for specific user\"\"\"
        rag = self.get_rag_for_user(user_id)
        return rag.query(question)

# Usage
multi_rag = MultiUserRAG()
answer = multi_rag.query_for_user("user_123", "What's in my documents?")
"""
    print(code)


# Example 6: RAG with Context Caching
def rag_with_caching():
    """RAG with query caching for faster responses"""
    
    code = """
from rag_pipeline import RAGPipeline
import hashlib
from functools import lru_cache

class CachedRAG:
    def __init__(self):
        self.rag = RAGPipeline()
        self.rag.ingest_documents()
        self.query_cache = {}
    
    @lru_cache(maxsize=128)
    def query_with_cache(self, question: str) -> str:
        \"\"\"Query with caching\"\"\"
        # Check if already computed
        q_hash = hashlib.md5(question.encode()).hexdigest()
        
        if q_hash in self.query_cache:
            print(f"Cache hit for: {question}")
            return self.query_cache[q_hash]
        
        # Compute and cache
        answer = self.rag.query(question)
        self.query_cache[q_hash] = answer
        return answer

# Usage
cached_rag = CachedRAG()
answer1 = cached_rag.query_with_cache("What is AI?")
answer2 = cached_rag.query_with_cache("What is AI?")  # From cache
"""
    print(code)


# Example 7: RAG with Feedback Loop
def rag_with_feedback():
    """RAG with user feedback for continuous improvement"""
    
    code = """
import json
from rag_pipeline import RAGPipeline

class ImprovisedRAG:
    def __init__(self):
        self.rag = RAGPipeline()
        self.rag.ingest_documents()
        self.feedback_log = []
    
    def query_and_get_feedback(self, question: str, feedback_score: int = None):
        \"\"\"Query and optionally log feedback\"\"\"
        answer = self.rag.query(question)
        
        if feedback_score is not None:
            self.log_feedback(question, answer, feedback_score)
        
        return answer
    
    def log_feedback(self, question: str, answer: str, score: int):
        \"\"\"Log user feedback (1-5 stars)\"\"\"
        feedback = {
            "question": question,
            "answer": answer[:100] + "...",
            "rating": score,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        self.feedback_log.append(feedback)
        
        # Save feedback
        with open("feedback.jsonl", "a") as f:
            f.write(json.dumps(feedback) + "\\n")
    
    def get_feedback_stats(self):
        \"\"\"Analyze feedback\"\"\"
        if not self.feedback_log:
            return {}
        
        ratings = [f["rating"] for f in self.feedback_log]
        return {
            "total_queries": len(self.feedback_log),
            "avg_rating": sum(ratings) / len(ratings),
            "ratings": ratings
        }
"""
    print(code)


if __name__ == "__main__":
    print("RAG Integration Examples")
    print("=" * 60)
    print("\n1. Simple Chatbot Integration")
    chatbot_with_rag()
    print("\n2. FastAPI Integration")
    fastapi_rag_api()
    print("\n3. Discord Bot Integration")
    discord_bot_with_rag()
    print("\n4. Telegram Bot Integration")
    telegram_bot_with_rag()
    print("\n5. Multi-User RAG System")
    multi_user_rag()
    print("\n6. RAG with Caching")
    rag_with_caching()
    print("\n7. RAG with Feedback Loop")
    rag_with_feedback()
