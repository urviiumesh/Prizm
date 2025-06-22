document.addEventListener('DOMContentLoaded', () => {
    // Voice Commands
    const voiceToggle = document.getElementById('voice-toggle');
    let voiceEnabled = false;

    if (voiceToggle) {
        voiceToggle.addEventListener('click', () => {
            voiceEnabled = !voiceEnabled;
            voiceToggle.setAttribute('aria-pressed', voiceEnabled);
            voiceToggle.textContent = voiceEnabled ? 'Disable Voice Commands' : 'Enable Voice Commands';
            announceVoiceStatus();
        });
    }

    // Keyboard Navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'u' || e.key === 'U') {
            window.location.href = 'upload.html';
        } else if (e.key === 'c' || e.key === 'C') {
            window.location.href = 'chat.html';
        } else if (e.key === ' ' && voiceToggle) {
            e.preventDefault();
            voiceToggle.click();
        }
    });

    // File Upload
    const fileUpload = document.getElementById('file-upload');
    const fileList = document.getElementById('file-list');

    if (fileUpload) {
        fileUpload.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            fileList.innerHTML = '';
            
            files.forEach(file => {
                const fileInfo = document.createElement('p');
                fileInfo.textContent = `Selected: ${file.name}`;
                fileList.appendChild(fileInfo);
                announceFileUpload(file.name);
            });
        });
    }

    // Chat Functionality
    const chatForm = document.getElementById('chat-form');
    const chatHistory = document.getElementById('chat-history');

    if (chatForm) {
        chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const input = document.getElementById('message');
            const message = input.value.trim();
            
            if (message) {
                addMessage(message, 'user');
                input.value = '';
                
                // Simulate response
                setTimeout(() => {
                    addMessage('I received your message. How can I help?', 'assistant');
                }, 1000);
            }
        });
    }

    // Helper Functions
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.textContent = text;
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        // Announce new messages to screen readers
        if (sender === 'assistant') {
            announceMessage(text);
        }
    }

    function announceMessage(text) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'polite');
        announcement.className = 'visually-hidden';
        announcement.textContent = `Assistant: ${text}`;
        document.body.appendChild(announcement);
        setTimeout(() => announcement.remove(), 1000);
    }

    function announceVoiceStatus() {
        const status = voiceEnabled ? 'Voice commands enabled' : 'Voice commands disabled';
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'polite');
        announcement.className = 'visually-hidden';
        announcement.textContent = status;
        document.body.appendChild(announcement);
        setTimeout(() => announcement.remove(), 1000);
    }

    function announceFileUpload(filename) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'polite');
        announcement.className = 'visually-hidden';
        announcement.textContent = `File ${filename} selected`;
        document.body.appendChild(announcement);
        setTimeout(() => announcement.remove(), 1000);
    }

    document.addEventListener("DOMContentLoaded", () => {
            const fileInput = document.getElementById("file-upload");
            const fileList = document.getElementById("file-list");
            const uploadButton = document.getElementById("upload-button");
            const resultsSection = document.getElementById("results");
            const extractedTextEl = document.getElementById("extracted-text");
            const summaryEl = document.getElementById("summary");
    
            // Update file list when files are selected
            fileInput.addEventListener("change", () => {
                console.log("File selection changed.");
                fileList.innerHTML = ""; // Clear existing file list
                const files = Array.from(fileInput.files);
    
                if (files.length > 0) {
                    console.log("Files selected:", files);
                    files.forEach(file => {
                        const fileItem = document.createElement("div");
                        fileItem.textContent = file.name;
                        fileItem.classList.add("file-item");
                        fileList.appendChild(fileItem);
                    });
                    uploadButton.disabled = false; // Enable upload button
                } else {
                    console.log("No files selected.");
                    uploadButton.disabled = true; // Disable upload button
                }
            });
    
            // Upload files to the server
            uploadButton.addEventListener("click", async (event) => {
                event.preventDefault(); // Prevent the page from refreshing
    
                const files = fileInput.files;
    
                if (files.length > 0) {
                    const formData = new FormData();
                    formData.append("file", files[0]);
    
                    try {
                        const response = await fetch("http://192.168.1.7:5000/upload_image", {
                            method: "POST",
                            body: formData
                        });
    
                        if (response.ok) {
                            const text = await response.text(); // Read the response as text first
                            console.log("Response Text:", text); // Log response to debug
    
                            try {
                                const data = JSON.parse(text); // Try to parse JSON
                                if (data.extracted_text && data.summary) {
                                    extractedTextEl.innerHTML = `<strong>Extracted Text:</strong> ${data.extracted_text}`;
                                    summaryEl.innerHTML = `<strong>Summary:</strong> ${data.summary}`;
                                    resultsSection.hidden = false;
                                } else {
                                    console.error("Response data is missing required fields.");
                                    alert("Error: Response data is incomplete.");
                                }
                            } catch (error) {
                                console.error("Failed to parse JSON:", error);
                                alert("Error: Invalid response format.");
                            }
    
                            // Reset the input and file list
                            fileInput.value = "";
                            fileList.innerHTML = "";
                            uploadButton.disabled = true;
                        } else {
                            const error = await response.text();
                            console.error("API call failed with error:", error);
                            alert(`Failed to upload files. Error: ${error}`);
                        }
                    } catch (error) {
                        console.error("Error uploading files:", error);
                        alert("An error occurred while uploading files.");
                    }
                }
            });
        });

});