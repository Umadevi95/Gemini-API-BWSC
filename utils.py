# ============================================================
# RESTAURANT CHATBOT - UTILS.PY
# ============================================================

import os
import json
import ast
import re
import time

from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1500
DEFAULT_MAX_RETRIES = 5


# ============================================================
# RESTAURANT MENU
# ============================================================

def get_menu():

    return [

        {
            "category": "Starters",
            "items": [
                "Paneer Tikka",
                "Chicken 65",
                "Gobi Manchurian",
                "French Fries"
            ]
        },

        {
            "category": "South Indian",
            "items": [
                "Masala Dosa",
                "Idli",
                "Plain Dosa",
                "Vada",
                "Pongal"
            ]
        },

        {
            "category": "Rice and Biryani",
            "items": [
                "Chicken Biryani",
                "Mutton Biryani",
                "Vegetable Biryani",
                "Jeera Rice",
                "Curd Rice"
            ]
        },

        {
            "category": "Main Course",
            "items": [
                "Paneer Butter Masala",
                "Chicken Curry",
                "Mutton Curry",
                "Dal Tadka",
                "Mixed Vegetable Curry"
            ]
        },

        {
            "category": "Breads",
            "items": [
                "Butter Naan",
                "Plain Naan",
                "Chapati",
                "Tandoori Roti"
            ]
        },

        {
            "category": "Desserts",
            "items": [
                "Gulab Jamun",
                "Ice Cream",
                "Carrot Halwa",
                "Brownie"
            ]
        },

        {
            "category": "Beverages",
            "items": [
                "Tea",
                "Coffee",
                "Fresh Lime Juice",
                "Mango Juice",
                "Lassi"
            ]
        }
    ]


# ============================================================
# MENU INFORMATION
# ============================================================

MENU_INFO = {

    "Paneer Tikka": {
        "category": "Starters",
        "description": "Grilled Indian cottage cheese with spices.",
        "vegetarian": True,
        "spicy": "Medium",
        "price": 220,
        "allergens": ["Milk"]
    },

    "Chicken 65": {
        "category": "Starters",
        "description": "Crispy spicy fried chicken.",
        "vegetarian": False,
        "spicy": "Medium",
        "price": 240,
        "allergens": []
    },

    "Gobi Manchurian": {
        "category": "Starters",
        "description": "Crispy cauliflower tossed in Manchurian sauce.",
        "vegetarian": True,
        "spicy": "Medium",
        "price": 180,
        "allergens": ["Soy"]
    },

    "French Fries": {
        "category": "Starters",
        "description": "Crispy seasoned potato fries.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 150,
        "allergens": []
    },

    "Masala Dosa": {
        "category": "South Indian",
        "description": "Crispy dosa served with spiced potato masala.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 120,
        "allergens": []
    },

    "Idli": {
        "category": "South Indian",
        "description": "Soft steamed rice cakes served with chutney and sambar.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 80,
        "allergens": []
    },

    "Plain Dosa": {
        "category": "South Indian",
        "description": "Traditional crispy South Indian dosa.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 90,
        "allergens": []
    },

    "Vada": {
        "category": "South Indian",
        "description": "Crispy South Indian lentil fritter.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 80,
        "allergens": []
    },

    "Pongal": {
        "category": "South Indian",
        "description": "Soft rice and lentil dish seasoned with pepper and spices.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 100,
        "allergens": ["Milk"]
    },

    "Chicken Biryani": {
        "category": "Rice and Biryani",
        "description": "Aromatic basmati rice cooked with chicken and spices.",
        "vegetarian": False,
        "spicy": "Medium",
        "price": 280,
        "allergens": []
    },

    "Mutton Biryani": {
        "category": "Rice and Biryani",
        "description": "Aromatic rice cooked with tender mutton and spices.",
        "vegetarian": False,
        "spicy": "Medium",
        "price": 350,
        "allergens": []
    },

    "Vegetable Biryani": {
        "category": "Rice and Biryani",
        "description": "Aromatic rice cooked with mixed vegetables and spices.",
        "vegetarian": True,
        "spicy": "Medium",
        "price": 220,
        "allergens": []
    },

    "Jeera Rice": {
        "category": "Rice and Biryani",
        "description": "Basmati rice flavored with cumin.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 160,
        "allergens": []
    },

    "Curd Rice": {
        "category": "Rice and Biryani",
        "description": "Rice mixed with seasoned yogurt.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 100,
        "allergens": ["Milk"]
    },

    "Paneer Butter Masala": {
        "category": "Main Course",
        "description": "Paneer cooked in a creamy tomato-based gravy.",
        "vegetarian": True,
        "spicy": "Medium",
        "price": 240,
        "allergens": ["Milk"]
    },

    "Chicken Curry": {
        "category": "Main Course",
        "description": "Chicken cooked in traditional Indian curry.",
        "vegetarian": False,
        "spicy": "Medium",
        "price": 260,
        "allergens": []
    },

    "Mutton Curry": {
        "category": "Main Course",
        "description": "Tender mutton cooked in spicy curry.",
        "vegetarian": False,
        "spicy": "Medium",
        "price": 320,
        "allergens": []
    },

    "Dal Tadka": {
        "category": "Main Course",
        "description": "Yellow lentils tempered with Indian spices.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 180,
        "allergens": []
    },

    "Mixed Vegetable Curry": {
        "category": "Main Course",
        "description": "Mixed vegetables cooked in Indian spices.",
        "vegetarian": True,
        "spicy": "Medium",
        "price": 200,
        "allergens": []
    },

    "Butter Naan": {
        "category": "Breads",
        "description": "Soft naan brushed with butter.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 70,
        "allergens": ["Milk", "Gluten"]
    },

    "Plain Naan": {
        "category": "Breads",
        "description": "Soft traditional naan.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 60,
        "allergens": ["Gluten"]
    },

    "Chapati": {
        "category": "Breads",
        "description": "Traditional whole-wheat flatbread.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 50,
        "allergens": ["Gluten"]
    },

    "Tandoori Roti": {
        "category": "Breads",
        "description": "Whole-wheat roti cooked in a tandoor.",
        "vegetarian": True,
        "spicy": "Mild",
        "price": 50,
        "allergens": ["Gluten"]
    },

    "Gulab Jamun": {
        "category": "Desserts",
        "description": "Soft milk-based dumplings served in sugar syrup.",
        "vegetarian": True,
        "spicy": "None",
        "price": 100,
        "allergens": ["Milk"]
    },

    "Ice Cream": {
        "category": "Desserts",
        "description": "Creamy ice cream dessert.",
        "vegetarian": True,
        "spicy": "None",
        "price": 100,
        "allergens": ["Milk"]
    },

    "Carrot Halwa": {
        "category": "Desserts",
        "description": "Sweet carrot dessert prepared with milk and sugar.",
        "vegetarian": True,
        "spicy": "None",
        "price": 120,
        "allergens": ["Milk"]
    },

    "Brownie": {
        "category": "Desserts",
        "description": "Chocolate brownie dessert.",
        "vegetarian": True,
        "spicy": "None",
        "price": 150,
        "allergens": ["Milk", "Gluten"]
    },

    "Tea": {
        "category": "Beverages",
        "description": "Indian-style hot tea.",
        "vegetarian": True,
        "spicy": "None",
        "price": 40,
        "allergens": ["Milk"]
    },

    "Coffee": {
        "category": "Beverages",
        "description": "Hot freshly prepared coffee.",
        "vegetarian": True,
        "spicy": "None",
        "price": 50,
        "allergens": ["Milk"]
    },

    "Fresh Lime Juice": {
        "category": "Beverages",
        "description": "Refreshing lime juice.",
        "vegetarian": True,
        "spicy": "None",
        "price": 80,
        "allergens": []
    },

    "Mango Juice": {
        "category": "Beverages",
        "description": "Fresh mango-based juice.",
        "vegetarian": True,
        "spicy": "None",
        "price": 100,
        "allergens": []
    },

    "Lassi": {
        "category": "Beverages",
        "description": "Traditional yogurt-based drink.",
        "vegetarian": True,
        "spicy": "None",
        "price": 90,
        "allergens": ["Milk"]
    }
}


# ============================================================
# API KEY
# ============================================================

def get_gemini_api_key():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    return api_key


# ============================================================
# GEMINI CLIENT
# ============================================================

def create_gemini_client():

    return genai.Client(
        api_key=get_gemini_api_key()
    )


# ============================================================
# GEMINI REQUEST WITH RETRY
# ============================================================

def generate_with_retry(
    client,
    model,
    contents,
    config,
    max_retries=DEFAULT_MAX_RETRIES,
):
    """Call Gemini with retry handling for temporary failures."""

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Sending request to Gemini (attempt {attempt}/{max_retries})...")

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            print("Gemini response received")
            print("Response type:", type(response))
            return response

        except ServerError as e:
            print(f"Gemini Server Error (attempt {attempt}/{max_retries}):", e)
            if attempt >= max_retries:
                return None
            time.sleep(2 ** (attempt - 1))

        except ClientError as e:
            print("Gemini Client Error:", e)
            return None

        except Exception as e:
            message = str(e)
            temporary = (
                "503" in message
                or "UNAVAILABLE" in message.upper()
                or "temporarily unavailable" in message.lower()
            )

            if temporary and attempt < max_retries:
                print("Temporary Gemini error. Retrying...")
                time.sleep(2 ** (attempt - 1))
            else:
                print("Gemini Error:", e)
                return None

    return None


# ============================================================
# COMPLETION
# ============================================================

def convert_messages_to_gemini(messages):

    contents = []
    system_instruction = None

    for message in messages:

        if not isinstance(message, dict):
            raise ValueError(
                f"Message must be a dictionary: {message}"
            )

        role = message.get("role")
        content = message.get("content")

        if content is None:
            raise ValueError(
                f"Message is missing content: {message}"
            )

        content = str(content)

        # SYSTEM MESSAGE
        if role == "system":
            system_instruction = content

        # USER MESSAGE
        elif role == "user":
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=content
                        )
                    ],
                )
            )

        # ASSISTANT MESSAGE
        elif role == "assistant":
            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=content
                        )
                    ],
                )
            )

        else:
            raise ValueError(
                f"Unsupported message role: {role}"
            )

    return system_instruction, contents

# ============================================================
# GEMINI COMPLETION
# ============================================================

def get_completion_from_messages(
    messages,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
    model=DEFAULT_MODEL,
):
    """Convert chat messages, call Gemini, and return text."""

    system_instruction, contents = convert_messages_to_gemini(messages)

    if not contents:
        raise ValueError("At least one user or assistant message is required.")

    client = create_gemini_client()

    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }

    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    config = types.GenerateContentConfig(**config_kwargs)

    response = generate_with_retry(
        client=client,
        model=model,
        contents=contents,
        config=config,
    )

    if response is None:
        return None

    try:
        text = response.text
        if text:
            return text.strip()
    except Exception:
        pass

    try:
        for candidate in (getattr(response, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            for part in (getattr(content, "parts", None) or []):
                part_text = getattr(part, "text", None)
                if part_text:
                    return str(part_text).strip()
    except Exception as e:
        print("Response extraction error:", e)

    return None


def get_gemini_response(user_message, system_instruction=None):
    """Simple helper for testing Gemini with one user message."""

    messages = []

    if system_instruction:
        messages.append({
            "role": "system",
            "content": system_instruction
        })

    messages.append({
        "role": "user",
        "content": user_message
    })

    return get_completion_from_messages(
        messages,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )


# ============================================================
# PARSER
# ============================================================

def parse_model_response(response, debug=False):

    if response is None:
        return None

    text = str(response).strip()

    text = re.sub(
        r"```(?:json|python)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "").strip()

    try:

        result = json.loads(text)

        if isinstance(result, list):
            return result

    except Exception:
        pass

    try:

        result = ast.literal_eval(text)

        if isinstance(result, list):
            return result

    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end > start:

        extracted = text[start:end + 1]

        try:

            result = json.loads(extracted)

            if isinstance(result, list):
                return result

        except Exception:
            pass

    if debug:
        print("Unable to parse response:")
        print(text)

    return None


# ============================================================
# MENU CATEGORY / ITEM IDENTIFICATION
# ============================================================

def find_category_and_item(
    user_input,
    menu=None
):

    if menu is None:
        menu = get_menu()

    system_message = f"""

You are a restaurant menu identification assistant.

Identify menu categories and food items mentioned
or requested by the customer.

Allowed restaurant menu:

{json.dumps(menu, indent=2)}

Rules:

1. Use ONLY items from the menu.
2. Never invent food items.
3. Product/item names must match exactly.
4. If customer asks for a category, return all items
   in that category.
5. If customer asks for vegetarian food,
   identify appropriate vegetarian items.
6. If customer asks for non-vegetarian food,
   identify appropriate non-vegetarian items.
7. If nothing matches, return [].
8. Return ONLY JSON.
9. No markdown.
10. Format:

[
  {{
    "category": "Category Name",
    "items": ["Item 1", "Item 2"]
  }}
]
"""

    messages = [

        {
            "role": "system",
            "content": system_message
        },

        {
            "role": "user",
            "content": f"####{user_input}####"
        }
    ]

    return get_completion_from_messages(
        messages,
        temperature=0,
        max_tokens=1000
    )


# ============================================================
# GET MENU INFORMATION
# ============================================================

def get_menu_information(
    category_item_list
):

    if isinstance(category_item_list, str):

        category_item_list = parse_model_response(
            category_item_list
        )

    if not isinstance(category_item_list, list):
        return {}

    result = {}

    for group in category_item_list:

        if not isinstance(group, dict):
            continue

        category = group.get("category")
        items = group.get("items", [])

        if category not in result:
            result[category] = {}

        for item in items:

            if item in MENU_INFO:

                result[category][item] = MENU_INFO[item]

    return result


# ============================================================
# INPUT MODERATION
# ============================================================

def moderate_user_input(user_input):

    blocked_patterns = [

        r"\bkill\b",
        r"\bbomb\b",
        r"\bweapon\b",
        r"\bexplosive\b",
        r"\bterrorist\b",
        r"\bmalware\b",
        r"\bpassword\s*steal\b"
    ]

    text = user_input.lower()

    for pattern in blocked_patterns:

        if re.search(pattern, text):

            return {
                "allowed": False,
                "reason": "Unsafe or inappropriate request."
            }

    return {
        "allowed": True,
        "reason": "OK"
    }


# ============================================================
# OUTPUT CHECK
# ============================================================

def check_output(
    user_question,
    answer,
    menu_information
):

    if not answer:
        return False

    system_message = """

You are a restaurant chatbot output quality checker.

Check whether the restaurant assistant response:

1. Answers the customer's question.
2. Does not invent menu items.
3. Does not invent prices.
4. Does not invent ingredients or allergens.
5. Does not provide unsafe information.
6. Is polite and useful.

Return ONLY:

Y

or

N
"""

    messages = [

        {
            "role": "system",
            "content": system_message
        },

        {
            "role": "user",
            "content": f"""
Customer question:

{user_question}

Available menu information:

{json.dumps(menu_information, indent=2)}

Assistant response:

{answer}

Return ONLY Y or N.
"""
        }
    ]

    result = get_completion_from_messages(
        messages,
        temperature=0,
        max_tokens=5
    )

    if not result:
        return False

    result = result.strip().upper()

    return result.startswith("Y")


# ============================================================
# FINAL RESTAURANT ANSWER
# ============================================================

def answer_customer(
    user_question,
    menu_information
):

    system_message = """

You are a helpful restaurant customer-service assistant.

Answer ONLY using the supplied restaurant menu information.

Rules:

1. Never invent menu items.
2. Never invent prices.
3. Never invent ingredients.
4. Never invent allergen information.
5. Never claim an item is available if it is not in
   the supplied information.
6. If information is unavailable, clearly say:
   "That information is not available in the menu."
7. Be friendly and concise.
8. Answer all parts of the customer's question.
"""

    messages = [

        {
            "role": "system",
            "content": system_message
        },

        {
            "role": "user",
            "content": f"""

Customer question:

{user_question}

Restaurant menu information:

{json.dumps(menu_information, indent=2)}

Provide the final restaurant response.
"""
        }
    ]

    return get_completion_from_messages(
        messages,
        temperature=0.2,
        max_tokens=1000
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def process_restaurant_message(
    user_input,
    conversation=None,
    debug=True
):

    if conversation is None:
        conversation = []

    # --------------------------------------------------------
    # STEP 1 - INPUT VALIDATION
    # --------------------------------------------------------

    if not user_input or not user_input.strip():

        return (
            "Please enter your restaurant question.",
            conversation
        )

    if debug:
        print("STEP 1: Input validation passed.")

    # --------------------------------------------------------
    # STEP 2 - MODERATION
    # --------------------------------------------------------

    moderation = moderate_user_input(
        user_input
    )

    if not moderation["allowed"]:

        fallback = (
            "Sorry, I can only help with restaurant "
            "menu, food, orders and dining-related questions."
        )

        return fallback, conversation

    if debug:
        print("STEP 2: Moderation passed.")

    # --------------------------------------------------------
    # STEP 3 - IDENTIFY FOOD ITEMS
    # --------------------------------------------------------

    category_response = find_category_and_item(
        user_input
    )

    if debug:
        print("\nCategory response:")
        print(category_response)

    category_list = parse_model_response(
        category_response
    )

    if category_list is None:
        category_list = []

    if debug:
        print("\nIdentified menu items:")
        print(category_list)

    # --------------------------------------------------------
    # STEP 4 - FETCH MENU INFORMATION
    # --------------------------------------------------------

    menu_information = get_menu_information(
        category_list
    )

    if debug:
        print("\nMenu information:")
        print(
            json.dumps(
                menu_information,
                indent=2
            )
        )

    # --------------------------------------------------------
    # STEP 5 - GENERATE ANSWER
    # --------------------------------------------------------

    final_answer = answer_customer(
        user_input,
        menu_information
    )

    if not final_answer:

        final_answer = (
            "Sorry, I couldn't process your request."
        )

    if debug:
        print("\nSTEP 5: Response generated.")

    # --------------------------------------------------------
    # STEP 6 - OUTPUT CHECK
    # --------------------------------------------------------

    approved = check_output(
        user_input,
        final_answer,
        menu_information
    )

    if debug:
        print(
            f"\nSTEP 6: Output approved = {approved}"
        )

    # --------------------------------------------------------
    # STEP 7 - FINAL DECISION
    # --------------------------------------------------------

    if approved:

        response = final_answer

    else:

        response = (
            "I'm sorry, I couldn't confidently answer "
            "that from the available restaurant menu. "
            "Please ask me about our menu, prices, "
            "vegetarian options or food items."
        )

    # --------------------------------------------------------
    # STEP 8 - SAVE CONVERSATION
    # --------------------------------------------------------

    conversation.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    conversation.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    return response, conversation
