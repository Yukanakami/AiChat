# 🤖 AiChat

![AiChat Banner](docs/images/aichat-banner.png)

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/Yukanakami/AiChat)  
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](#-license)

> A private, local AI chatbot powered by FastAPI and Ollama — run models like LLaMA, Mistral, or Gemma directly on your machine.

---

## 📚 Table of Contents

- [✨ Features](#-features)  
- [🛠 Tech Stack](#-tech-stack)  
- [📸 Demo](#-demo)  
- [⚙️ Installation](#-installation)  
- [🖥️ GUI](#-gui)  
- [📡 API Usage](#-api-usage)  
- [📂 Project Structure](#-project-structure)  
- [🤝 Contributing](#-contributing)  
- [📄 License](#-license)  
- [👤 Author](#-author)

---

## ✨ Features

- 🧠 Run AI chatbot models locally (no cloud required)  
- ⚙️ Lightweight API built with FastAPI  
- 🚀 Supports multiple models via Ollama (LLaMA, Mistral, Gemma, etc.)  
- 🖥️ Desktop GUI included (built with Tkinter)  
- 🔐 100% privacy — your data never leaves your device

---

## 🛠 Tech Stack

- **Backend:** Python, FastAPI  
- **Model Runtime:** [Ollama](https://ollama.com)  
- **GUI:** Tkinter (Python Desktop)  
- **API Docs:** Swagger UI (auto-generated)

---

## 📸 Demo

> ![AiChat GUI Screenshot](docs/images/aichat-gui-demo.png)

*A simple desktop interface for chatting with models like llama3*

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/Yukanakami/AiChat.git
cd AiChat

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
🖥️ GUI
bash
Salin
Edit
# Launch the desktop GUI
python gui.py
The GUI includes a message input, model dropdown, and response output from the AI.

📡 API Usage
Start the FastAPI server:
bash
Salin
Edit
uvicorn main:app --reload
Visit the docs: http://localhost:8000/docs

Sample API Request:
http
Salin
Edit
POST /chat
Content-Type: application/json

{
  "model": "llama3",
  "message": "Hello, who are you?"
}
Sample Response:
json
Salin
Edit
{
  "response": "Hi! I'm a local AI running on your machine."
}
📂 Project Structure
graphql
Salin
Edit
AiChat/
├── main.py            # FastAPI backend
├── model.py           # Chat processing logic
├── schema.py          # Pydantic models
├── gui.py             # Tkinter desktop interface
├── requirements.txt   # Python dependencies
└── README.md
🤝 Contributing
Contributions, issues, and suggestions are welcome!
Feel free to fork the repo and open a pull request.

📄 License
MIT License © 2025 Yukanakami

This project and its contents are open source. You may modify, use, and distribute freely under the terms of the MIT license.

👤 Author
Nyoman Maheka Wijananta Putra
