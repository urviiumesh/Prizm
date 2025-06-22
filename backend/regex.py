import re


def block_personal_info(text):
    # Define general regex patterns to match personal information
    patterns = {
        'NAME': r'\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,2}\b',  # Matches names (e.g., John Michael Doe)
        'DOB': r'\d{1,2}[a-zA-Z]{3,9}\s\d{4}',  # Matches Date of Birth (e.g., 12th December 2024)
        'EMPLOYEE_ID': r'\b[A-Z]{3}\d{7}\b',  # Matches Employee ID (e.g., ABC1234567)
        'PHONE': r'\+91 \d{10}',  # Matches phone number in +91 format (e.g., +91 9876543210)
        'EMAIL': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Matches email (e.g., john.doe@example.com)
        'ADDRESS': r'\d{1,5}[A-Za-z\s]+(?:[A-Za-z]+\s?)+',  # Matches address patterns (e.g., 1234 Corporate Blvd)
        'EMERGENCY_CONTACT': r'\+91 \d{10}'  # Matches emergency contact phone number
    }

    # Replace personal information with placeholders
    for key, pattern in patterns.items():
        text = re.sub(pattern, f'[{key} BLOCKED]', text)
    return text


def get_user_input():
    # Take multi-line input from the user
    print("Please enter the text that contains personal information (type 'exit' to stop):")
    user_input = []
    while True:
        line = input()
        if line.lower() == 'exit':
            break
        user_input.append(line)

    # Join the input lines into a single string
    return "\n".join(user_input)


# Main function
if __name__ == "__main__":
    # Get user input
    input_text = get_user_input()

    # Call the function to block personal information from the input text
    cleaned_text = block_personal_info(input_text)

    # Output the cleaned text
    print("\nProcessed Text (Personal Information Blocked):\n")
    print(cleaned_text)