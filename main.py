import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np
import base64
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq, GroqError
import re
import requests
import time
import io
from functools import wraps
import threading
import queue
import sys
import tempfile
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables and initialize APIs
def initialize_apis():
    """Initialize API clients and check environment variables."""
    load_dotenv()
    
    # Check for required API keys
    required_vars = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "SPOONACULAR_API_KEY": os.getenv("SPOONACULAR_API_KEY")
    }
    
    missing_vars = [key for key, value in required_vars.items() if not value]
    if missing_vars:
        error_msg = f"Missing environment variables: {', '.join(missing_vars)}"
        st.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Initialize Groq client
        groq_client = Groq(api_key=required_vars["GROQ_API_KEY"])
        
        # Initialize Google API
        genai.configure(api_key=required_vars["GOOGLE_API_KEY"])
        
        return groq_client
    except Exception as e:
        st.error(f"Failed to initialize APIs: {str(e)}")
        raise

# Initialize global clients
groq_client = initialize_apis()

def encode_image(image_path):
    """Encode image to base64 format for API compatibility."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise Exception(f"Error encoding image: {str(e)}")

# Replace the timeout decorator with this cross-platform version
def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result_queue = queue.Queue()
            
            def worker():
                try:
                    result = func(*args, **kwargs)
                    result_queue.put(('success', result))
                except Exception as e:
                    result_queue.put(('error', e))
            
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            
            try:
                status, result = result_queue.get(timeout=seconds)
                if status == 'error':
                    raise result
                return result
            except queue.Empty:
                raise TimeoutError(f"Function timed out after {seconds} seconds")
            
        return wrapper
    return decorator

# Replace the existing safe_generate_content function with:
@timeout(30)  # 30 second timeout
def safe_generate_content(model, prompt, image):
    """Safely generate content with timeout."""
    return model.generate_content([prompt, image])

def analyze_fridge_image(image_path):
    """Analyze fridge contents using Google's Gemini model."""
    if not os.path.exists(image_path):
        st.error(f"Image file not found: {image_path}")
        return None, None, None
        
    image = None
    try:
        image = Image.open(image_path)
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        
        prompt_items = """
        Analyze this image of a refrigerator's contents and list all the food
        items you can identify.
        For each item, provide the following information in this exact format:
        Item: [Item name]
        Quantity: [Estimated quantity or 'Not visible' if you can't determine]
        Location: [Bounding box coordinates as [ymin, xmin, ymax, xmax] where
        each coordinate is an integer between 0 and 1000]
        """

        for attempt in range(3):
            try:
                response_items = safe_generate_content(model, prompt_items, image)
                if not response_items or not response_items.text:
                    st.warning(f"Empty response on attempt {attempt + 1}")
                    time.sleep(2)
                    continue
                    
                items_info = parse_item_info(response_items.text.strip())
                if not items_info:
                    st.warning(f"No items detected on attempt {attempt + 1}")
                    time.sleep(2)
                    continue
                    
                annotated_image = generate_annotated_image(image_path, items_info)
                analysis_result = "\n".join([f"{item}: {info['quantity']}" for item, info in items_info.items()])
                analysis_result += f"\n\nTotal number of distinct food items: {len(items_info)}"
                
                return analysis_result, annotated_image, items_info
                
            except Exception as e:
                if '429' in str(e):
                    if attempt < 2:
                        st.warning("Quota exceeded. Retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    else:
                        st.error("Quota exceeded. Please try again later.")
                        return None, None, None
                else:
                    st.error(f"An error occurred: {str(e)}")
                    return None, None, None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None
    finally:
        if image:
            image.close()

def parse_item_info(response_text):
    """Parse the model's response into structured item information."""
    if not response_text:
        return {}
        
    items_info = {}
    current_item = None
    
    try:
        for line in response_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Item:"):
                current_item = line.split(":", 1)[1].strip()
                items_info[current_item] = {'quantity': 'Not specified', 'box': [0, 0, 0, 0]}
            elif line.startswith("Quantity:") and current_item:
                items_info[current_item]['quantity'] = line.split(":", 1)[1].strip()
            elif line.startswith("Location:") and current_item:
                location = line.split(":", 1)[1].strip()
                box = parse_bounding_box(location)
                items_info[current_item]['box'] = box
    except Exception as e:
        st.warning(f"Error parsing item info: {str(e)}")
        return {}
        
    return items_info

def parse_bounding_box(location_str):
    """Parse bounding box coordinates from string."""
    try:
        numbers = re.findall(r'\d+', location_str)
        if len(numbers) >= 4:
            coords = [int(numbers[i]) for i in range(4)]
            if all(0 <= coord <= 1000 for coord in coords):
                return coords
        return [0, 0, 0, 0]
    except Exception:
        return [0, 0, 0, 0]

def generate_annotated_image(image_path, items_info):
    """Generate image with bounding boxes around detected items."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        height, width = image.shape[:2]
        
        for item, info in items_info.items():
            if not isinstance(info, dict) or 'box' not in info or 'quantity' not in info:
                continue
                
            box = info.get('box', [0, 0, 0, 0])
            quantity = info.get('quantity', 'Unknown')

            if box and box != [0, 0, 0, 0]:
                try:
                    ymin, xmin, ymax, xmax = convert_coordinates(box, width, height)
                    ymin = max(0, min(ymin, height - 1))
                    ymax = max(0, min(ymax, height - 1))
                    xmin = max(0, min(xmin, width - 1))
                    xmax = max(0, min(xmax, width - 1))
                    
                    if ymin < ymax and xmin < xmax:
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        text_y = max(20, ymin - 10)
                        label = f"{item}: {quantity}"
                        cv2.putText(image, label, (xmin, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    st.warning(f"Error drawing box for {item}: {str(e)}")
                    continue

        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        st.error(f"Error generating annotated image: {str(e)}")
        return None

def convert_coordinates(box, original_width, original_height):
    """Convert normalized coordinates to pixel coordinates."""
    ymin, xmin, ymax, xmax = box
    return [
        int(ymin / 1000 * original_height),
        int(xmin / 1000 * original_width),
        int(ymax / 1000 * original_height),
        int(xmax / 1000 * original_width)
    ]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_recipes_from_spoonacular(ingredients, number=4):
    """Get recipe suggestions from Spoonacular API."""
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        'apiKey': os.getenv("SPOONACULAR_API_KEY"),
        'ingredients': ','.join(ingredients),
        'number': number,
        'ranking': 2,
        'ignorePantry': True
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        recipes = response.json()
        
        if not recipes:
            st.warning("No recipes found for the given ingredients.")
            st.info("Try uploading a different image with more ingredients.")
            return []

        return recipes
    except requests.exceptions.RequestException as e:
        st.error(f"Recipe search failed: {str(e)}")
        return []

def create_recipe_card(recipe, recipe_details):
    """Create a styled recipe card with HTML/CSS formatting."""
    missed_ingredients = ', '.join([ing['name'] for ing in recipe.get('missedIngredients', [])])
    used_ingredients = ', '.join([ing['name'] for ing in recipe.get('usedIngredients', [])])
    key_info = parse_recipe_key_info(recipe_details)

    # Default values if key information is missing
    for field in ['Calories', 'Cooking Time', 'Price', 'Dietary', 'Cuisine', 'Difficulty']:
        key_info.setdefault(field, "Not Found")

    card_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); height: 600px; overflow-y: auto;">
        <img src="{recipe['image']}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px;">
        <h3 style="margin-top: 10px;">{recipe['title']}</h3>
        <p>🔥 **Calories**: {key_info['Calories']}</p>
        <p>⏱️ **Cooking Time**: {key_info['Cooking Time']}</p>
        <p>💰 **Price**: {key_info['Price']}</p>
        <p>🥗 **Dietary**: {key_info['Dietary']}</p>
        <p>🌎 **Cuisine**: {key_info['Cuisine']}</p>
        <p>📊 **Difficulty**: {key_info['Difficulty']}</p>
        <p><strong>Used ingredients:</strong> {used_ingredients}</p>
        <p><strong>Missing ingredients:</strong> {missed_ingredients}</p>
        <div>
            {recipe_details}
        </div>
    </div>
    """
    return card_html

def parse_recipe_key_info(recipe_details):
    """Parse key information from recipe details."""
    key_info = {}
    try:
        key_info_section = recipe_details.split("Key Information:")[1].split("\n\n")[0]
        for line in key_info_section.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key_info[key.strip()] = value.strip()
    except Exception:
        pass
    return key_info

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_recipes(ingredients_key):
    """Cache recipe results to improve performance."""
    return get_recipes_from_spoonacular(ingredients_key)

def display_recipes(items_info):
    """Display recipe cards in a grid layout."""
    ingredients_key = tuple(sorted(items_info.keys()))
    
    with st.spinner("🔍 Searching for recipes..."):
        recipes = get_cached_recipes(ingredients_key)
        
    if recipes:
        cols = st.columns(2)
        for idx, recipe in enumerate(recipes):
            with cols[idx % 2]:
                recipe_details = generate_recipe_details_groq(recipe)
                st.markdown(create_recipe_card(recipe, recipe_details), unsafe_allow_html=True)
    else:
        st.warning("No recipes found. Try with different ingredients.")

def process_image(uploaded_file, max_size=(800, 800)):
    """Process and resize uploaded image."""
    try:
        image = Image.open(uploaded_file)
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        image.save(temp_file.name, 'JPEG', quality=85)
        return temp_file.name
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None
    finally:
        if 'image' in locals():
            image.close()

def generate_recipe_details_groq(recipe):
    """Generate detailed recipe information using Groq API."""
    try:
        prompt = f"""
        Generate a detailed recipe for "{recipe['title']}" based on the following information:

        Ingredients:
        {' '.join([f"- {ingredient['original']}" for ingredient in recipe.get('usedIngredients', []) + recipe.get('missedIngredients', [])])}

        Provide the following information in this exact format:

        Key Information:
        Calories: [Estimated calories per serving]
        Cooking Time: [Estimated total time in minutes]
        Price: [Estimated price per serving in USD]
        Dietary: [List any dietary categories this recipe fits, e.g., Vegetarian, Vegan, Gluten-Free, etc.]
        Cuisine: [Type of cuisine, e.g., Italian, Mexican, etc.]
        Difficulty: [Easy/Medium/Hard]

        Description:
        [Provide a brief, enticing description of the dish in 2-3 sentences]

        Instructions:
        1. [Step 1]
        2. [Step 2]
        ...

        Additional Information:
        [Flavor Profile, Texture, Nutritional Highlights, Serving Suggestions, Tips]
        """

        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a helpful culinary assistant with expertise in various cuisines and cooking techniques."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating recipe details: {str(e)}")
        return "Recipe details unavailable"



def main():
    """Main application function with Streamlit UI."""
    try:
        st.set_page_config(layout="wide", page_title="FoodEase")
        st.title("🍽️ FoodEase: AI Family Hub")
        
        # Initialize APIs
        with st.spinner("Initializing..."):
            groq_client = initialize_apis()
            
        # Add help text
        st.markdown("""
        ### 📸 How to use:
        1. Upload a photo of your fridge contents
        2. Wait for AI analysis
        3. View detected items and recipe suggestions
        """)
        
        uploaded_file = st.file_uploader(
            "📸 Upload an image of your fridge", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of your fridge contents"
        )
        
        if uploaded_file:
            with st.spinner("🔍 Processing your image..."):
                temp_path = process_image(uploaded_file)
                if temp_path:
                    try:
                        analysis_result, annotated_image, items_info = analyze_fridge_image(temp_path)
                        
                        if analysis_result and items_info:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(annotated_image, caption="Analyzed Fridge Contents")
                            
                            with col2:
                                st.markdown("### 📝 Detected Items:")
                                st.write(analysis_result)
                            
                            st.markdown("### 🍳 Recommended Recipes")
                            display_recipes(items_info)
                        else:
                            st.error("Could not analyze the image. Please try again with a clearer photo.")
                    finally:
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                            
    except Exception as e:
        st.error("😔 Something went wrong!")
        st.error(str(e))
        if st.button("🔄 Restart Application"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()


