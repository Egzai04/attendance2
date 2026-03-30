# 🎓 Smart Attendance System

AI-powered attendance system using **Face Recognition**, **LangChain**, **Groq (Llama 3)**, **HuggingFace**, and **Streamlit**.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📸 **Face Registration** | Upload photo or use webcam to register student faces |
| 🤖 **Auto Attendance** | Mark attendance by recognizing faces in real-time |
| ✍️ **Manual Entry** | Mark attendance manually for any date/subject |
| 📊 **Reports** | Daily, per-student, and analytics dashboards |
| 🤖 **AI Assistant** | LangChain + Groq chatbot for attendance Q&A |
| ⚠️ **At-Risk Alerts** | Automatically flag students below 75% |

---

## 🏗️ Project Structure

```
smart_attendance/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── .env                    # API keys (create this)
├── pages/
│   ├── dashboard.py        # Main dashboard
│   ├── register.py         # Student registration + face upload
│   ├── attendance.py       # Mark attendance (auto + manual)
│   ├── reports.py          # Reports and analytics
│   ├── ai_assistant.py     # LangChain + Groq chatbot
│   └── settings.py         # API key config
├── utils/
│   ├── database.py         # JSON-based data store
│   ├── face_recognition.py # DeepFace / HF / fallback embeddings
│   └── ai_engine.py        # LangChain + Groq integration
└── database/               # Auto-created JSON storage
    ├── students.json
    ├── attendance.json
    └── embeddings.json
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env` file
```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxx
```
- **Groq** (free): https://console.groq.com
- **HuggingFace** (free): https://huggingface.co/settings/tokens

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🧠 AI Architecture

```
Face Input (Camera/Upload)
        ↓
  Face Detection (OpenCV Haar Cascade)
        ↓
  Embedding Extraction:
    1. DeepFace (Facenet) — local, most accurate
    2. HuggingFace DINO   — API-based
    3. Simulated          — demo/fallback
        ↓
  Cosine Similarity → Match Student
        ↓
  Mark Attendance in DB

AI Chatbot:
  User Query → LangChain → Groq Llama 3 → Answer
```

---

## 📝 Notes

- **Face recognition accuracy** depends on image quality and lighting
- Default **similarity threshold**: 0.75 (adjustable in the UI)
- Database is **local JSON** — can be upgraded to SQLite/PostgreSQL
- For production, use **DeepFace with GPU** for best performance
