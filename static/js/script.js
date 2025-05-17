// script.js
document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const loadingContainer = document.getElementById('loading-container');

    // Function to add a message to the chat container
    function addMessage(content, type, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${type}-message`);
        
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        
        const paragraph = document.createElement('p');
        paragraph.textContent = content;
        messageContent.appendChild(paragraph);
        
        // Add sources if available
        if (sources && sources.length > 0) {
            const sourceList = document.createElement('div');
            sourceList.classList.add('source-list');
            
            const sourceTitle = document.createElement('h4');
            sourceTitle.textContent = 'Sources:';
            sourceList.appendChild(sourceTitle);
            
            const sourceUl = document.createElement('ul');
            sources.forEach(source => {
                const sourceLi = document.createElement('li');
                const sourceLink = document.createElement('a');
                sourceLink.href = source;
                sourceLink.textContent = source;
                sourceLink.target = "_blank";
                sourceLi.appendChild(sourceLink);
                sourceUl.appendChild(sourceLi);
            });
            sourceList.appendChild(sourceUl);
            messageContent.appendChild(sourceList);
        }
        
        messageDiv.appendChild(messageContent);
        chatContainer.appendChild(messageDiv);
        
        // Scroll to the bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Function to send user question and get response
    async function sendQuestion(question) {
        // Disable input and show loading
        userInput.disabled = true;
        sendButton.disabled = true;
        loadingContainer.style.display = 'flex';
        
        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Add the AI response to the chat
                addMessage(data.answer, 'system', data.sources);
            } else {
                // Handle errors
                addMessage(`Error: ${data.error || 'Failed to get response'}`, 'system');
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('Sorry, there was an error processing your request. Please try again later.', 'system');
        } finally {
            // Re-enable input and hide loading
            userInput.disabled = false;
            sendButton.disabled = false;
            loadingContainer.style.display = 'none';
            userInput.focus();
        }
    }

    // Event listener for send button
    sendButton.addEventListener('click', function() {
        const question = userInput.value.trim();
        if (question) {
            // Add user message to chat
            addMessage(question, 'user');
            // Clear input
            userInput.value = '';
            // Send question to server
            sendQuestion(question);
        }
    });

    // Event listener for Enter key
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendButton.click();
        }
    });

    // Auto-focus the input field on page load
    userInput.focus();

    // Auto-resize textarea as user types
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        // Limit max height to avoid filling the entire screen
        const maxHeight = 150;
        const newHeight = Math.min(this.scrollHeight, maxHeight);
        this.style.height = newHeight + 'px';
    });
});