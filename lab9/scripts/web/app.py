from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import uuid
from pdf_processor import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain

# Initialize Flask application
app = Flask(__name__)
# Set a random secret key for session security
app.secret_key = os.urandom(24)  # Generate a random secret key
# Define upload folder for PDFs
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define a function that runs before each request to initialize session variables
@app.before_request
def initialize_session():
    # Skip initialization for static files to improve performance
    if '/static/' in request.path:
        return
        
    # Initialize session variables if they don't exist
    if 'chat_history' not in session:
        session['chat_history'] = []  # Store chat messages
    if 'pdfs' not in session:
        session['pdfs'] = []  # Store information about uploaded PDFs
    if 'vectorstore' not in session:
        session['vectorstore'] = None  # Flag to indicate if vectorstore is created
    if 'conversation_initialized' not in session:
        session['conversation_initialized'] = False  # Flag for conversation state

# Route for the main page (PDF upload interface)
@app.route('/')
def index():
    # Debug code to check if CSS file exists and print its path
    css_path = os.path.join(app.static_folder, 'css', 'style.css')
    print(f"Looking for CSS at: {css_path}")
    print(f"CSS file exists: {os.path.exists(css_path)}")
    # Render the main upload page
    return render_template('index.html')

# Route to directly serve CSS files (alternative to standard static file serving)
@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(app.static_folder, 'css'), filename)

# API endpoint for handling PDF file uploads
@app.route('/upload', methods=['POST'])
def upload_pdf():
    # Check if files were included in the request
    if 'pdfFiles' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    # Get list of uploaded files
    uploaded_files = request.files.getlist('pdfFiles')
    pdf_files = []

    # Process each uploaded file
    for file in uploaded_files:
        # Skip files with empty names
        if file.filename == '':
            continue
        
        # Process only PDF files
        if file and file.filename.endswith('.pdf'):
            # Secure the filename to prevent path traversal attacks
            filename = secure_filename(file.filename)
            # Create a unique filename to prevent collisions
            unique_filename = f"{uuid.uuid4()}_{filename}"
            # Define the full path where the file will be saved
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            # Save the file to disk
            file.save(file_path)
            
            # Store file information for later processing
            pdf_files.append({
                'filename': filename,  # Original filename for display
                'path': file_path      # Storage path for processing
            })
    
    # Save PDF information in the session
    session['pdfs'] = pdf_files
    session.modified = True  # Mark session as modified to ensure it's saved
    
    # Return success response with list of uploaded filenames
    return jsonify({'success': True, 'files': [pdf['filename'] for pdf in pdf_files]})

# Route for analyzing uploaded PDFs and preparing for chat
@app.route('/analyze')
def analyze():
    # Redirect to upload page if no PDFs have been uploaded
    if not session.get('pdfs'):
        return redirect(url_for('index'))
    
    # Process PDFs using imported NLP functions
    try:
        # Get paths of all uploaded PDFs
        pdf_paths = [pdf['path'] for pdf in session.get('pdfs', [])]
        
        # Step 1: Extract text content from all PDFs
        raw_text = get_pdf_text(pdf_paths)
        
        # Step 2: Split the text into manageable chunks for processing
        text_chunks = get_text_chunks(raw_text)
        
        # Step 3: Create vector embeddings from text chunks
        vectorstore = get_vectorstore(text_chunks)
        
        # Mark in the session that processing is complete
        # Note: We only store flags in the session since complex objects like vectorstores can't be serialized in the session
        session['vectorstore'] = True
        session['conversation_initialized'] = True
        session.modified = True
        
    except Exception as e:
        # Log any errors during processing but continue to chat page
        print(f"Error processing PDFs: {str(e)}")
        # We still render the chat page even if analysis fails, but the error will be communicated to the user when they try to chat
    
    # Render the chat interface with the list of processed PDFs
    return render_template('chat.html', pdfs=session.get('pdfs', []))

# API endpoint for handling chat messages
@app.route('/send_message', methods=['POST'])
def send_message():
    # Parse the JSON request data
    data = request.json
    user_message = data.get('message', '')
    
    # Validate input
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Add user message to chat history
    chat_history = session.get('chat_history', [])
    chat_history.append({
        'sender': 'user',
        'message': user_message
    })
    
    # Generate AI response if PDFs have been processed
    if session.get('conversation_initialized'):
        try:
            # Get conversation chain instance
            # This retrieves the previously created chain from the module
            conversation = get_conversation_chain()
            
            # Process user message through the conversation chain
            response = conversation({'question': user_message})
            ai_response = response['answer']
        except Exception as e:
            # Handle any errors during processing
            print(f"Error using conversation chain: {str(e)}")
            ai_response = f"I encountered an error processing your question. Please try again."
    else:
        # Inform user they need to upload and process PDFs first
        ai_response = "Please upload and process PDF documents first before asking questions."
    
    # Add AI response to chat history
    chat_history.append({
        'sender': 'ai',
        'message': ai_response
    })
    
    # Update session with new chat history
    session['chat_history'] = chat_history
    session.modified = True
    
    # Return the AI response and updated chat history
    return jsonify({
        'success': True,
        'response': ai_response,
        'chat_history': chat_history
    })

# API endpoint to retrieve chat history
@app.route('/get_chat_history')
def get_chat_history():
    return jsonify({'chat_history': session.get('chat_history', [])})

# Route to reset the session and start over
@app.route('/reset')
def reset():
    # Clear PDF data and chat history from session
    session.pop('pdfs', None)
    session.pop('chat_history', None)
    # Redirect to upload page
    return redirect(url_for('index'))

# Run the Flask application in debug mode if executed directly
if __name__ == '__main__':
    app.run(debug=True)