from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a random secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store chat history in session
@app.before_request
def initialize_session():
    # Skip initialization for static files
    if '/static/' in request.path:
        return
        
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'pdfs' not in session:
        session['pdfs'] = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdfFiles' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    uploaded_files = request.files.getlist('pdfFiles')
    pdf_files = []

    for file in uploaded_files:
        if file.filename == '':
            continue
        
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            pdf_files.append({
                'filename': filename,
                'path': file_path
            })
    
    # Store the PDF info in the session
    session['pdfs'] = pdf_files
    session.modified = True
    
    return jsonify({'success': True, 'files': [pdf['filename'] for pdf in pdf_files]})

@app.route('/analyze')
def analyze():
    if not session.get('pdfs'):
        return redirect(url_for('index'))
    
    # In a real application, you'd trigger your Python analysis script here
    # For now, we just render the chat page
    return render_template('chat.html', pdfs=session.get('pdfs', []))

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Add message to chat history
    chat_history = session.get('chat_history', [])
    chat_history.append({
        'sender': 'user',
        'message': user_message
    })
    
    # In a real app, this is where you'd query your vector database
    # and generate a response based on the PDF content
    ai_response = f"Here's information related to: {user_message}"
    
    chat_history.append({
        'sender': 'ai',
        'message': ai_response
    })
    
    session['chat_history'] = chat_history
    session.modified = True
    
    return jsonify({
        'success': True,
        'response': ai_response,
        'chat_history': chat_history
    })

@app.route('/get_chat_history')
def get_chat_history():
    return jsonify({'chat_history': session.get('chat_history', [])})

@app.route('/reset')
def reset():
    # Clear the session data
    session.pop('pdfs', None)
    session.pop('chat_history', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)