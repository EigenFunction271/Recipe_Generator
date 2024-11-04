# FoodEase: AI Family Hub

An AI-powered kitchen assistant that analyzes your fridge contents and suggests recipes.

## Prerequisites

- Python 3.8 or higher
- Internet connection
- Webcam or food images for analysis

## Setup Instructions

1. **Clone the repository**
bash
git clone <repository-url>
cd <project-folder>

2. **Set up Python virtual environment**

For Windows:
```bash
python -m venv venv
```

For Mac/Linux:
```bash
python3 -m venv venv
```

3. **Create `.env` file**
Create a file named `.env` in the project root and add your API keys:
```
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
SPOONACULAR_API_KEY=your_spoonacular_api_key
```

## Running the Application

### Windows Users
Option 1: Double click `run.bat`

Option 2: Open command prompt in project directory and type:
```bash
.\run.bat
```

### Mac/Linux Users
Option 1: Open terminal in project directory and type:
```bash
chmod +x run.sh  # Make script executable (first time only)
./run.sh
```

Option 2: Manual setup:
```bash
# Activate virtual environment
source venv/bin/activate  # For Mac/Linux
# Install dependencies
pip install -r requirements.txt
# Run application
streamlit run main.py
```

## Troubleshooting

If you encounter any errors:
1. Make sure all API keys are correctly set in `.env`
2. Verify Python version: `python --version`
3. Check if virtual environment is activated
4. Ensure all dependencies are installed: `pip install -r requirements.txt`
5. Try running with administrator/sudo privileges

## Features

- 📸 Fridge content analysis
- 🔍 Food item detection
- 🍳 Recipe suggestions
- 📊 Ingredient tracking

## Requirements

See `requirements.txt` for full list of dependencies:
- Python 3.8+
- Streamlit
- OpenCV
- Google Generative AI
- Groq
- PIL (Pillow)
- And more...

## Support

For issues or questions, please open a GitHub issue or contact support.

## License

[Your License Here]
