from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
import os
import numpy as np
import faiss
from openai import OpenAI
from werkzeug.utils import secure_filename
import uuid
from dotenv import load_dotenv
from chatbox import extract_pdf_text, chunk_text, embed_text, create_vector_store, conversation_chain


# Initialize OpenAI client
load_dotenv()
client = OpenAI()

llm = ["gpt-4o-mini", "gemma3:27b-it-q4_K_M" ]
emb_models = ["text-embedding-3-small", "nomic-embed-text"]

# Initialize Flask application
app = Flask(__name__)
# Set a random secret key for session security
app.secret_key = os.urandom(24)  # Generate a random secret key
# Define upload folder for PDFs
app.config["UPLOAD_FOLDER"] = "uploads"
# Add vector store and text chunks as application variables
app.config["VECTOR_STORE"] = None
app.config["TEXT_CHUNKS"] = None

# Create upload directory if it doesn"t exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Define a function that runs before each request to initialize session variables
@app.before_request
def initialize_session():
    # Skip initialization for static files to improve performance
    if "/static/" in request.path:
        return
        
    # Initialize session variables if they don't exist
    if "chat_history" not in session:
        session["chat_history"] = []  # Store chat messages
    if "pdfs" not in session:
        session["pdfs"] = []  # Store information about uploaded PDFs
    if "vector_store_created" not in session:
        session["vector_store_created"] = False  # Flag to indicate if vectorstore is created

# Route for the main page (PDF upload interface)
@app.route("/")
def index():
    # Debug code to check if CSS file exists and print its path
    css_path = os.path.join(app.static_folder, "css", "style.css")
    print(f"Looking for CSS at: {css_path}")
    print(f"CSS file exists: {os.path.exists(css_path)}")
    # Render the main upload page
    return render_template("index.html")

# Route to directly serve CSS files (alternative to standard static file serving)
@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(app.static_folder, "css"), filename)

# API endpoint for handling PDF file uploads
@app.route("/upload", methods=["POST"])
def upload_pdf():
    # Check if files were included in the request
    if "pdfFiles" not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    # Get list of uploaded files
    uploaded_files = request.files.getlist("pdfFiles")
    pdf_files = []

    # Process each uploaded file
    for file in uploaded_files:
        # Skip files with empty names
        if file.filename == "":
            continue
        
        # Process only PDF files
        if file and file.filename.endswith(".pdf"):
            # Secure the filename to prevent path traversal attacks
            filename = secure_filename(file.filename)
            # Create a unique filename to prevent collisions
            unique_filename = f"{uuid.uuid4()}_{filename}"
            # Define the full path where the file will be saved
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            # Save the file to disk
            file.save(file_path)
            
            # Store file information for later processing
            pdf_files.append({
                "filename": filename,  # Original filename for display
                "path": file_path      # Storage path for processing
            })
    
    # Save PDF information in the session
    session["pdfs"] = pdf_files
    session.modified = True  # Mark session as modified to ensure it's saved
    
    # Return success response with list of uploaded filenames
    return jsonify({"success": True, "files": [pdf["filename"] for pdf in pdf_files]})

# Route for analyzing uploaded PDFs and preparing for chat
@app.route("/analyze")
def analyze():
    # Redirect to upload page if no PDFs have been uploaded
    if not session.get("pdfs"):
        return redirect(url_for("index"))
    
    # Process PDFs using imported NLP functions
    try:
        # Get paths of all uploaded PDFs
        pdf_paths = [pdf["path"] for pdf in session.get("pdfs", [])]
        
        # Step 1: Extract text content from all PDFs
        raw_text = extract_pdf_text(pdf_paths)
        
        # Step 2: Split the text into manageable chunks
        text_chunks = chunk_text(raw_text)
        
        # Step 3: Generate embeddings for text chunks
        embeddings = embed_text(text_chunks, emb_models[1])
        
        # Step 4: Create vector store and store it in the app config
        vector_store = create_vector_store(embeddings)
        app.config["VECTOR_STORE"] = vector_store
        app.config["TEXT_CHUNKS"] = text_chunks
        
        # Store a flag in the session that processing is complete
        session["vector_store_created"] = True
        session.modified = True
        
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")
        
    # Render the chat interface with the list of processed PDFs
    return render_template("chat.html", pdfs=session.get("pdfs", []))

# API endpoint for handling chat messages
@app.route("/send_message", methods=["POST"])
def send_message():
    # Parse the JSON request data
    data = request.json
    user_message = data.get("message", "")
    
    # Validate input
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Add user message to chat history for display
    display_chat_history = session.get("chat_history", [])
    display_chat_history.append({
        "sender": "user",
        "message": user_message
    })
    
    # Prepare chat history for the conversation chain
    # This converts the chat history from our format to the format expected by conversation_chain
    chain_chat_history = []
    for msg in display_chat_history:
        if msg["sender"] == "user":
            chain_chat_history.append(f"User: {msg['message']}")
        else:
            chain_chat_history.append(f"Bot: {msg['message']}")
    
    # Generate AI response if vector store has been created
    if session.get("vector_store_created"):
        try:
            # Get vector store and text chunks from app config
            vector_store = app.config["VECTOR_STORE"]
            text_chunks = app.config["TEXT_CHUNKS"]
            
            # Create a conversation chain for this request
            conversation = conversation_chain(
                query=user_message,
                text_chunks=text_chunks,
                vector_store=vector_store,
                chat_history=chain_chat_history,
                model = llm[1]
            )
            
            # Get response from conversation
            bot_response = conversation
            
        except Exception as e:
            # Handle any errors during processing
            print(f"Error using conversation chain: {str(e)}")
            bot_response = "I encountered an error processing your question. Please try again."
    else:
        # Inform user they need to upload and process PDFs first
        bot_response = "Please upload and process PDF documents first before asking questions."
    
    # Add AI response to chat history
    display_chat_history.append({
        "sender": "bot",
        "message": bot_response
    })
    
    # Update session with new chat history
    session["chat_history"] = display_chat_history
    session.modified = True
    
    # Return the AI response and updated chat history
    return jsonify({
        "success": True,
        "response": bot_response,
        "chat_history": display_chat_history
    })

# API endpoint to retrieve chat history
@app.route("/get_chat_history")
def get_chat_history():
    return jsonify({"chat_history": session.get("chat_history", [])})

# Route to reset the session and start over
@app.route("/reset")
def reset():
    # Clear session data
    session.pop("pdfs", None)
    session.pop("chat_history", None)
    session.pop("vector_store_created", None)
    
    # Clear app config data
    app.config["VECTOR_STORE"] = None
    app.config["TEXT_CHUNKS"] = None
    
    # Redirect to upload page
    return redirect(url_for("index"))

# Run the Flask application in debug mode if executed directly
if __name__ == "__main__":
    app.run(debug=True)