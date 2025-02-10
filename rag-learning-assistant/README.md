# 📖 RAG-Based Learning Assistant  

A Retrieval-Augmented Generation (RAG) based learning assistant that allows users to:  
- Upload PDFs and extract meaningful text  
- Ask questions and receive in-depth explanations  
- Generate quizzes based on the uploaded content  
- Perform web searches for additional insights  
- Interact with an AI chatbot for detailed responses  

## 🚀 Features  
✅ **PDF Upload & Processing** – Extracts text from PDF and stores embeddings for retrieval  
✅ **AI-Powered Q&A** – Provides in-depth explanations using OpenRouter API  
✅ **Web Search Integration** – Fetches relevant search results from DuckDuckGo  
✅ **Quiz Generation** – Creates MCQs based on extracted content  
✅ **Interactive Chatbot** – Answers user queries elaborately  

---

## 🛠 Installation  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/PRASANNA-THE-PRASANN1/rag-Learning-Assistant.git
   cd RAG-Learning-Assistant

   Install Dependencies
Make sure you have Python installed (3.8+ recommended). Then, install the required libraries:


pip install -r requirements.txt
Run the Application


streamlit run app.py

⚙️ Tech Stack
Frontend: Streamlit
Backend: Python, FAISS, LangChain
AI Model: OpenRouter (GPT-3.5 Turbo)
Search API: DuckDuckGo
📌 Usage
1️⃣ Upload a PDF
Click on the Upload PDF section and select a .pdf file
The document will be processed and chunked for retrieval
2️⃣ Ask Questions
Type your query in the AI-Powered Research Assistant section
Click Get Answer to retrieve relevant context and generate an in-depth response
3️⃣ Generate a Quiz
Choose the number of MCQs you want
Click Generate Quiz to create questions from the extracted content
4️⃣ Chat with AI
Type a message in the Chat with AI section
Click Send to interact with the chatbot
🔑 Environment Variables
To use OpenRouter, set up your API key in .env or replace it directly in the code:

📜 License
This project is open-source under the MIT License. Feel free to modify and contribute! 🚀

🤝 Contributing
Pull requests are welcome! If you'd like to contribute, follow these steps:

Fork the repo
Create a new branch (git checkout -b feature-branch)
Commit changes (git commit -m "Added a new feature")
Push to the branch (git push origin feature-branch)
Create a Pull Request
💡 Future Improvements
🔹 Enhance UI/UX with better styling
🔹 Improve retrieval accuracy with embeddings tuning
🔹 Add support for more file types (DOCX, TXT)
🔹 Implement real-time AI chat mode

💖 Developed by Prasanna



