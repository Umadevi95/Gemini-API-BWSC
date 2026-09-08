
# ============================================================
# utils.py
# ============================================================
# Product / Category utilities + Gemini API helper
# Compatible with google-genai
#
# IMPORTANT:
# - No function calling / AFC is used.
# - Only plain text generation is used.
# - Includes retry handling for temporary 503 errors.
# - Includes fallback extraction from Gemini candidates.
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

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

DEFAULT_TEMPERATURE = 0

DEFAULT_MAX_TOKENS = 2000

DEFAULT_MAX_RETRIES = 5


# ============================================================
# PRODUCT CATALOG
# ============================================================

def get_products_and_category():

    return [
        {
            "category": "Computers and Laptops",
            "products": [
                "TechPro Ultrabook",
                "BlueWave Gaming Laptop",
                "PowerLite Convertible",
                "TechPro Desktop",
                "BlueWave Chromebook",
            ],
        },

        {
            "category": "Smartphones and Accessories",
            "products": [
                "SmartX ProPhone",
                "MobiTech PowerCase",
                "SmartX MiniPhone",
                "MobiTech Wireless Charger",
                "SmartX EarBuds",
                "ChargX Wireless Charger",
                "WallCharge Adapter",
            ],
        },

        {
            "category": "Televisions and Home Theater Systems",
            "products": [
                "CineView 4K TV",
                "CineView 8K TV",
                "CineView OLED TV",
                "SoundMax Home Theater",
                "SoundMax Soundbar",
            ],
        },

        {
            "category": "Gaming Consoles and Accessories",
            "products": [
                "GameSphere X",
                "GameSphere Y",
                "ProGamer Controller",
                "ProGamer Racing Wheel",
                "GameSphere VR Headset",
                "HandHeld Games",
            ],
        },

        {
            "category": "Audio Equipment",
            "products": [
                "SoundMax Headphones",
                "SoundMax Earphones",
                "SoundMax Bluetooth Speaker",
            ],
        },

        {
            "category": "Cameras and Camcorders",
            "products": [
                "FotoSnap DSLR Camera",
                "FotoSnap Mirrorless Camera",
                "ActionCam 4K",
                "ZoomMaster Camcorder",
                "FotoSnap Instant Camera",
            ],
        },
    ]


# ============================================================
# PRODUCT INFORMATION
# ============================================================

PRODUCT_INFO = {

    "SmartX ProPhone": {
        "category": "Smartphones and Accessories",
        "description": "A powerful smartphone with advanced camera features.",
        "features": [
            "12MP dual camera",
            "5G wireless",
            "128GB storage",
            "6.1-inch display",
        ],
        "price": "$899.99",
    },

    "FotoSnap DSLR Camera": {
        "category": "Cameras and Camcorders",
        "description": "A DSLR camera for capturing photos and videos.",
        "features": [
            "1080p video",
            "3-inch LCD",
            "24.2MP sensor",
            "interchangeable lenses",
        ],
        "price": "$599.99",
    },

    "CineView 4K TV": {
        "category": "Televisions and Home Theater Systems",
        "description": "A 4K TV with vibrant colors and smart features.",
        "features": [
            "55-inch display",
            "4K resolution",
            "HDR",
            "Smart TV",
        ],
        "price": "$599.00",
    },

    "CineView 8K TV": {
        "category": "Televisions and Home Theater Systems",
        "description": "A stunning 8K TV.",
        "features": [
            "65-inch display",
            "8K resolution",
            "HDR",
            "Smart TV",
        ],
        "price": "$2999.99",
    },

    "CineView OLED TV": {
        "category": "Televisions and Home Theater Systems",
        "description": "An OLED TV with vibrant colors.",
        "features": [
            "55-inch display",
            "4K resolution",
            "HDR",
            "Smart TV",
        ],
        "price": "$1499.99",
    },

    "SoundMax Home Theater": {
        "category": "Televisions and Home Theater Systems",
        "description": "A home theater system for an immersive audio experience.",
        "features": [
            "5.1 channel",
            "1000W output",
            "wireless subwoofer",
            "Bluetooth",
        ],
        "price": "$399.99",
    },

    "SoundMax Soundbar": {
        "category": "Televisions and Home Theater Systems",
        "description": "A sleek and powerful soundbar.",
        "features": [
            "2.1 channel",
            "300W output",
            "wireless subwoofer",
            "Bluetooth",
        ],
        "price": "$199.99",
    },
}


# ============================================================
# PRODUCT INFORMATION LOOKUP
# ============================================================

def get_mentioned_product_info(category_and_product_list):

    if category_and_product_list is None:
        return {}

    if isinstance(category_and_product_list, str):

        category_and_product_list = read_string_to_list(
            category_and_product_list
        )

    if not isinstance(category_and_product_list, list):
        return {}

    result = {}

    for item in category_and_product_list:

        if not isinstance(item, dict):
            continue

        category = item.get("category")

        products = item.get("products", [])

        if not isinstance(products, list):
            products = [products]

        if category not in result:
            result[category] = {}

        for product in products:

            if product in PRODUCT_INFO:

                result[category][product] = PRODUCT_INFO[product]

    return result


# ============================================================
# API KEY
# ============================================================

def get_gemini_api_key():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n\n"

            "Jupyter:\n"
            "os.environ['GEMINI_API_KEY'] = 'YOUR_API_KEY'\n\n"

            "Windows CMD:\n"
            "set GEMINI_API_KEY=YOUR_API_KEY\n\n"

            "PowerShell:\n"
            "$env:GEMINI_API_KEY='YOUR_API_KEY'"
        )

    return api_key


# ============================================================
# CREATE GEMINI CLIENT
# ============================================================

def create_gemini_client():

    api_key = get_gemini_api_key()

    return genai.Client(api_key=api_key)


# ============================================================
# MESSAGE CONVERSION
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

        # ----------------------------------------------------
        # SYSTEM
        # ----------------------------------------------------

        if role == "system":

            system_instruction = content

        # ----------------------------------------------------
        # USER
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ASSISTANT / MODEL
        # ----------------------------------------------------

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
# EXTRACT TEXT FROM GEMINI RESPONSE
# ============================================================

def extract_response_text(response):

    if response is None:
        return None

    # --------------------------------------------------------
    # METHOD 1
    # response.text
    # --------------------------------------------------------

    try:

        text = response.text

        if text:

            return str(text).strip()

    except Exception:
        pass

    # --------------------------------------------------------
    # METHOD 2
    # candidates -> content -> parts -> text
    # --------------------------------------------------------

    try:

        candidates = getattr(
            response,
            "candidates",
            None
        )

        if candidates:

            collected = []

            for candidate in candidates:

                content = getattr(
                    candidate,
                    "content",
                    None
                )

                if content is None:
                    continue

                parts = getattr(
                    content,
                    "parts",
                    None
                )

                if not parts:
                    continue

                for part in parts:

                    text = getattr(
                        part,
                        "text",
                        None
                    )

                    if text:

                        collected.append(
                            str(text)
                        )

            if collected:

                return "\n".join(
                    collected
                ).strip()

    except Exception as e:

        print(
            "Could not extract candidate text:",
            e
        )

    return None


# ============================================================
# ONE GEMINI REQUEST + RETRY
# ============================================================

def generate_with_retry(
    client,
    model,
    contents,
    config,
    max_retries=DEFAULT_MAX_RETRIES,
):

    """
    Sends a plain Gemini text-generation request.

    No tools.
    No function calling.
    No AFC.

    Temporary 503 / UNAVAILABLE errors are retried.
    """

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            print(
                f"Gemini request "
                f"(attempt {attempt}/{max_retries})"
            )

            # IMPORTANT:
            # No tools are supplied here.
            # Therefore this evaluation flow does not
            # intentionally use automatic function calling.

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            return response

        # ----------------------------------------------------
        # SERVER ERROR
        # ----------------------------------------------------

        except ServerError as e:

            message = str(e)

            if attempt >= max_retries:

                print(
                    "\nGemini remained unavailable "
                    "after all retries."
                )

                print(message)

                return None

            wait_time = 2 ** (attempt - 1)

            print(
                "\nTemporary Gemini server problem:"
            )

            print(message)

            print(
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

        # ----------------------------------------------------
        # CLIENT ERROR
        # ----------------------------------------------------

        except ClientError as e:

            print("\nGemini ClientError:")

            print(str(e))

            return None

        # ----------------------------------------------------
        # GENERIC ERROR
        # ----------------------------------------------------

        except Exception as e:

            message = str(e)

            temporary_error = (
                "503" in message
                or
                "UNAVAILABLE"
                in message.upper()
                or
                "high demand"
                in message.lower()
                or
                "temporarily unavailable"
                in message.lower()
            )

            if temporary_error:

                if attempt >= max_retries:

                    print(
                        "\nGemini remained unavailable "
                        "after all retries."
                    )

                    print(message)

                    return None

                wait_time = 2 ** (attempt - 1)

                print(
                    "\nTemporary Gemini "
                    "availability problem:"
                )

                print(message)

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    "\nUnexpected Gemini exception:"
                )

                print(
                    f"Type: {type(e).__name__}"
                )

                print(
                    f"Message: {e}"
                )

                return None

    return None


# ============================================================
# GEMINI COMPLETION
# ============================================================

def get_completion_from_messages(
    messages,
    model=DEFAULT_MODEL,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
    max_retries=DEFAULT_MAX_RETRIES,
):

    system_instruction, contents = (
        convert_messages_to_gemini(messages)
    )

    if not contents:

        raise ValueError(
            "At least one user/model message is required."
        )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config_kwargs = {

        "temperature": temperature,

        "max_output_tokens": max_tokens,
    }

    if system_instruction:

        config_kwargs[
            "system_instruction"
        ] = system_instruction

    # IMPORTANT:
    # We intentionally do NOT provide:
    #
    # tools=
    # automatic_function_calling=
    # function_declarations=
    #
    # because this project does not need function calling.

    config = types.GenerateContentConfig(
        **config_kwargs
    )

    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    client = create_gemini_client()

    response = generate_with_retry(
        client=client,
        model=model,
        contents=contents,
        config=config,
        max_retries=max_retries,
    )

    if response is None:

        return None

    # --------------------------------------------------------
    # EXTRACT RESPONSE
    # --------------------------------------------------------

    text = extract_response_text(
        response
    )

    if not text:

        print(
            "\nGemini returned an empty response."
        )

        # Helpful debugging information

        try:

            print(
                "Response type:",
                type(response).__name__
            )

        except Exception:
            pass

        try:

            candidates = getattr(
                response,
                "candidates",
                None
            )

            print(
                "Candidates:",
                candidates
            )

        except Exception:
            pass

        return None

    return text.strip()


# ============================================================
# PARSING HELPERS
# ============================================================

def parse_model_response(
    response,
    debug=False
):

    if response is None:

        return None

    text = str(response).strip()

    if not text:

        return None

    # --------------------------------------------------------
    # Remove markdown code fences
    # --------------------------------------------------------

    text = re.sub(
        r"```(?:json|python)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace(
        "```",
        ""
    ).strip()

    # --------------------------------------------------------
    # METHOD 1 - JSON
    # --------------------------------------------------------

    try:

        result = json.loads(text)

        if isinstance(result, list):

            return result

    except Exception:
        pass

    # --------------------------------------------------------
    # METHOD 2 - Python literal
    # --------------------------------------------------------

    try:

        result = ast.literal_eval(text)

        if isinstance(result, list):

            return result

    except Exception:
        pass

    # --------------------------------------------------------
    # METHOD 3 - Extract array from extra text
    # --------------------------------------------------------

    start = text.find("[")

    end = text.rfind("]")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        extracted = text[
            start:end + 1
        ]

        # JSON

        try:

            result = json.loads(
                extracted
            )

            if isinstance(result, list):

                return result

        except Exception:
            pass

        # Python literal

        try:

            result = ast.literal_eval(
                extracted
            )

            if isinstance(result, list):

                return result

        except Exception:
            pass

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    if debug:

        print(
            "\nCould not parse Gemini response:"
        )

        print(text)

    return None


# ============================================================
# STRING -> LIST
# ============================================================

def read_string_to_list(response):

    """
    Compatibility function used by Part II.

    Converts a Gemini JSON/Python-list string
    into a Python list.
    """

    return parse_model_response(
        response,
        debug=True
    )


# ============================================================
# FIND CATEGORY + PRODUCT - V2
# ============================================================

def find_category_and_product_v2(
    user_input,
    products_and_category=None,
):

    if products_and_category is None:

        products_and_category = (
            get_products_and_category()
        )

    delimiter = "####"

    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_message = f"""

You identify products and categories from customer questions.

The customer query is delimited by {delimiter}.

Return ONLY a JSON array.

Each object must have exactly this structure:

{{
  "category": "...",
  "products": ["...", "..."]
}}

Allowed catalog:

{json.dumps(
    products_and_category,
    indent=2
)}

Rules:

1. Use only products from the catalog.

2. Never invent a product.

3. Product names must match the catalog exactly.

4. If the user explicitly mentions a product,
   return that product.

5. If the user clearly asks for a category,
   return the products in that category.

6. If the user asks for TVs or TV-related products,
   return the relevant products from:
   Televisions and Home Theater Systems.

7. If nothing matches,
   return [].

8. Do not explain anything.

9. Return ONLY valid JSON.

10. Do not use markdown.

"""

    # --------------------------------------------------------
    # MESSAGES
    # --------------------------------------------------------

    messages = [

        {
            "role": "system",
            "content": system_message,
        },

        {
            "role": "user",
            "content": (
                f"{delimiter}"
                f"{user_input}"
                f"{delimiter}"
            ),
        },
    ]

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    return get_completion_from_messages(

        messages,

        model=DEFAULT_MODEL,

        temperature=0,

        max_tokens=1000,

        max_retries=DEFAULT_MAX_RETRIES,
    )


# ============================================================
# FIND CATEGORY + PRODUCT - V1
# ============================================================

def find_category_and_product_v1(
    user_input,
    products_and_category=None,
):

    # Keep V1 available for old notebooks.

    return find_category_and_product_v2(
        user_input,
        products_and_category,
    )


# ============================================================
# GET PRODUCTS FROM QUERY
# ============================================================

def get_products_from_query(
    user_input
):

    """
    Compatibility function expected by Part II.

    Returns a JSON string.

    Example:

    [
      {
        "category": "Smartphones and Accessories",
        "products": ["SmartX ProPhone"]
      }
    ]
    """

    catalog = (
        get_products_and_category()
    )

    response = find_category_and_product_v2(
        user_input,
        catalog,
    )

    if response is None:

        return "[]"

    parsed = parse_model_response(
        response
    )

    if parsed is None:

        return "[]"

    # --------------------------------------------------------
    # SAFETY FILTER
    # --------------------------------------------------------

    allowed = {

        item["category"]: set(
            item["products"]
        )

        for item in catalog
    }

    cleaned = []

    for item in parsed:

        if not isinstance(
            item,
            dict
        ):

            continue

        category = item.get(
            "category"
        )

        products = item.get(
            "products",
            []
        )

        if category not in allowed:

            continue

        if not isinstance(
            products,
            list
        ):

            products = [products]

        valid_products = []

        for product in products:

            if product in allowed[
                category
            ]:

                if product not in valid_products:

                    valid_products.append(
                        product
                    )

        if valid_products:

            cleaned.append(
                {
                    "category": category,
                    "products": valid_products,
                }
            )

    return json.dumps(
        cleaned,
        indent=2
    )


# ============================================================
# ANSWER USER MESSAGE
# ============================================================

def answer_user_msg(
    user_msg,
    product_info
):

    """
    Generate final customer-service answer
    using ONLY supplied product information.
    """

    system_message = """

You are a helpful customer service assistant.

Answer the customer's question using ONLY
the supplied product context.

Rules:

1. Do not invent product specifications.

2. Do not invent prices.

3. Do not invent features.

4. If information is not provided,
   clearly say that it is not provided.

5. Answer every part of the customer's question.

6. Be clear and friendly.

"""

    user_message = f"""

Customer question:

{user_msg}


Product context:

{json.dumps(
    product_info,
    indent=2
)}


Write the final customer-service answer.

"""

    messages = [

        {
            "role": "system",
            "content": system_message,
        },

        {
            "role": "user",
            "content": user_message,
        },
    ]

    return get_completion_from_messages(

        messages,

        model=DEFAULT_MODEL,

        temperature=0.2,

        max_tokens=1200,

        max_retries=DEFAULT_MAX_RETRIES,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_model_response(
    response,
    debug=False
):

    parsed = parse_model_response(
        response,
        debug=debug
    )

    if parsed is None:

        return None

    normalized = {}

    for item in parsed:

        if not isinstance(
            item,
            dict
        ):

            continue

        category = item.get(
            "category"
        )

        products = item.get(
            "products",
            []
        )

        if not category:

            continue

        if not isinstance(
            products,
            list
        ):

            products = [products]

        normalized.setdefault(
            category,
            set()
        )

        for product in products:

            if product is not None:

                normalized[
                    category
                ].add(
                    str(product).strip()
                )

    return normalized


# ============================================================
# NORMALIZE IDEAL ANSWER
# ============================================================

def normalize_ideal_answer(
    ideal
):

    if ideal == []:

        return {}

    if ideal is None:

        return None

    if not isinstance(
        ideal,
        dict
    ):

        return {}

    normalized = {}

    for category, products in ideal.items():

        if isinstance(
            products,
            set
        ):

            normalized[
                category
            ] = set(products)

        elif isinstance(
            products,
            list
        ):

            normalized[
                category
            ] = set(products)

        elif products is None:

            normalized[
                category
            ] = set()

        else:

            normalized[
                category
            ] = {
                str(products)
            }

    return normalized


# ============================================================
# EVALUATE RESPONSE WITH IDEAL
# ============================================================

def eval_response_with_ideal(
    response,
    ideal,
    debug=False
):

    if response is None:

        return None

    actual = normalize_model_response(
        response,
        debug=debug
    )

    if actual is None:

        return False

    expected = normalize_ideal_answer(
        ideal
    )

    return actual == expected


# ============================================================
# SIMPLE GEMINI CONNECTION TEST
# ============================================================

def test_gemini_connection():

    response = get_completion_from_messages(

        [
            {
                "role": "user",
                "content": (
                    "Reply with the word OK."
                ),
            }
        ],

        model=DEFAULT_MODEL,

        temperature=0,

        max_tokens=10,

        max_retries=DEFAULT_MAX_RETRIES,
    )

    print(
        "\nGemini test response:"
    )

    print(response)

    return response


# ============================================================
# DEBUG INFORMATION
# ============================================================

def print_configuration():

    print("\n========== GEMINI CONFIGURATION ==========")

    print(
        "Model:",
        DEFAULT_MODEL
    )

    print(
        "Temperature:",
        DEFAULT_TEMPERATURE
    )

    print(
        "Max tokens:",
        DEFAULT_MAX_TOKENS
    )

    print(
        "Max retries:",
        DEFAULT_MAX_RETRIES
    )

    print(
        "API key configured:",
        bool(os.getenv("GEMINI_API_KEY"))
    )

    print(
        "==========================================\n"
    )
