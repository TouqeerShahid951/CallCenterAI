# AI Call Center Agent

## Backend (FastAPI)

1. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. Run the backend server:
   ```bash
   uvicorn src.main:app --reload
   ```

## Frontend (Streamlit)

1. Install dependencies:
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```
2. Run the frontend app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Open the Streamlit app in your browser.
- Upload an audio file to interact with the backend.

## Project Structure
- `backend/`: FastAPI backend (controllers, services, business logic)
- `frontend/`: Streamlit frontend (UI only) 