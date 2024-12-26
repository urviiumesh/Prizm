// DOM Elements
const toggleVoiceBotButton = document.getElementById('toggle-voice-bot');
const navButtons = document.querySelectorAll('.nav-button');
const tabContents = document.querySelectorAll('.tab-content');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatHistory = document.getElementById('chat-history');
const scanButton = document.getElementById('scan-button');
const uploadButton = document.getElementById('upload-button');
const increaseFontButton = document.getElementById('increase-font');
const decreaseFontButton = document.getElementById('decrease-font');

// Send user message to the Flask backend and update chat history
const sendMessage = async (message) => {
    const userMessageDiv = document.createElement('div');
    userMessageDiv.classList.add('message', 'user'); // Added 'message' class
    userMessageDiv.textContent = message;
    chatHistory.appendChild(userMessageDiv);

    const botMessageDiv = document.createElement('div');
    botMessageDiv.classList.add('message', 'bot'); // Added 'message' class
    botMessageDiv.textContent = '...';
    chatHistory.appendChild(botMessageDiv);

    chatHistory.scrollTop = chatHistory.scrollHeight;

    try {
        const response = await fetch('/chat', { // Relative URL
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Server error');
        }

        const data = await response.json();
        botMessageDiv.textContent = data.reply;
        chatHistory.scrollTop = chatHistory.scrollHeight;

    } catch (error) {
        console.error("Error calling backend:", error);
        botMessageDiv.textContent = 'Sorry, I encountered an error.';
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
};

// Handle the form submit event
chatForm.addEventListener('submit', (event) => {
    event.preventDefault();
    const userMessage = chatInput.value.trim();
    if (userMessage) {
        sendMessage(userMessage);
        chatInput.value = '';
    }
});

// --- Voice Bot Functionality ---
let voiceBotEnabled = true;
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'en-US';
recognition.interimResults = false;
const synth = window.speechSynthesis;

const speakText = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    synth.speak(utterance);
};

const startListening = () => {
    if (voiceBotEnabled) {
        recognition.start();
        speakText("Voice bot is active. Please speak your commands.");
    }
};

const handleVoiceCommand = (command) => {
    command = command.toLowerCase();

    if (command.includes('go to home')) {
        document.querySelector("[data-tab='home']").click();
    } else if (command.includes('go to upload')) {
        document.querySelector("[data-tab='upload']").click();
    } else if (command.includes('go to chatbot')) {
        document.querySelector("[data-tab='chat']").click();
    } else if (command.includes('scan document')) {
        scanButton.click();
    } else if (command.includes('upload document')) {
        uploadButton.click();
    } else if (command.includes('increase font')) {
        increaseFontButton.click();
    } else if (command.includes('decrease font')) {
        decreaseFontButton.click();
    } else if (command.includes('toggle voice bot')) {
        toggleVoiceBotButton.click();
    } else if (command.includes('send message')) { // Voice command to send message
        const message = command.replace('send message', '').trim(); // Extract the message
        if (message) {
            sendMessage(message);
        } else {
            speakText("Please provide a message to send.");
        }
    } else {
        speakText("Command not recognized.");
    }
};

navButtons.forEach(button => {
    button.addEventListener('click', () => {
        const targetTab = button.dataset.tab;
        tabContents.forEach(tab => tab.classList.add('hidden'));
        document.getElementById(targetTab).classList.remove('hidden');
        speakText(`You are now on the ${targetTab} tab.`);
    });
});

toggleVoiceBotButton.addEventListener('click', () => {
    voiceBotEnabled = !voiceBotEnabled;
    toggleVoiceBotButton.textContent = voiceBotEnabled ? 'Disable Voice Bot' : 'Enable Voice Bot';
    voiceBotEnabled ? startListening() : recognition.stop();
});

let currentFontSize = 16;
increaseFontButton.addEventListener('click', () => {
    currentFontSize += 2;
    document.body.style.fontSize = `${currentFontSize}px`;
});

decreaseFontButton.addEventListener('click', () => {
    if (currentFontSize > 10) {
        currentFontSize -= 2;
        document.body.style.fontSize = `${currentFontSize}px`;
    }
});

window.onload = () => {
    startListening();
};

recognition.onresult = (event) => {
    const command = event.results[0][0].transcript;
    handleVoiceCommand(command);
};

recognition.onend = () => {
    if (voiceBotEnabled) {
        recognition.start();
    }
};