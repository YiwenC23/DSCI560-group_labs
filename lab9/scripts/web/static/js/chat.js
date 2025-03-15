document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chatMessages');
    const messageForm = document.getElementById('messageForm');
    const userMessageInput = document.getElementById('userMessage');
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');
    
    // Load chat history when the page loads
    loadChatHistory();
    
    // Handle sending messages
    messageForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const message = userMessageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessageToChat('user', message);
        
        // Clear input
        userMessageInput.value = '';
        
        // Send message to server
        fetch('/send_message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Add AI response to chat
                addMessageToChat('ai', data.response);
            } else {
                console.error('Error:', data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
    
    // Handle new analysis button click
    newAnalysisBtn.addEventListener('click', function() {
        window.location.href = '/reset';
    });
    
    // Function to load chat history
    function loadChatHistory() {
        fetch('/get_chat_history')
            .then(response => response.json())
            .then(data => {
                // Clear existing messages
                chatMessages.innerHTML = '';
                
                // Add welcome message if no history
                if (data.chat_history.length === 0) {
                    addMessageToChat('ai', "Hello! I've analyzed your PDFs. Ask me anything about them.");
                    return;
                }
                
                // Add messages from history
                data.chat_history.forEach(msg => {
                    addMessageToChat(msg.sender, msg.message);
                });
                
                // Scroll to bottom
                scrollToBottom();
            })
            .catch(error => {
                console.error('Error loading chat history:', error);
            });
    }
    
    // Function to add a message to the chat
    function addMessageToChat(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messagePara = document.createElement('p');
        messagePara.textContent = message;
        
        messageContent.appendChild(messagePara);
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        scrollToBottom();
    }
    
    // Function to scroll chat to bottom
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});