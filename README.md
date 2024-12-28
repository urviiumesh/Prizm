## Overview

This project provides the following key features:

**Frontend Features:**

*   **Voice Bot:** Enables voice interaction with Gemini, allowing users to navigate through summarized content.
*   **Integrated Camera:** Allows users to capture images of documents, extract text, and receive summaries powered by Gemini. This includes continuous scanning functionality.
*   **Upload Box:** Facilitates uploading files for processing and summarization by Gemini. Users can preview uploaded files before submitting them.

**Backend Features:**

*   **Camera Processing:** Captures images of paper documents, extracts text, summarizes it using Gemini, and sends the results back to the frontend.
*   **File Upload Processing:** Takes PNG/JPG images or other uploaded files as input, extracts text, summarizes it using Gemini, and sends the results back to the frontend.
*   **Chatbot:** Enables two-way communication using the Gemini API, allowing users to have interactive conversations based on scanned or uploaded text.


## Architecture

The project follows a client-server architecture:

*   **Frontend:** Provides the user interface for interacting with the system, including voice input, camera capture, file uploads, and displaying summarized text and chat interactions.
*   **Backend:** Handles the core logic, including text extraction, summarization , chatbot interaction, and the recommendation system.
