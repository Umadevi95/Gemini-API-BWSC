from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from typing import Any

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

PRODUCTS_FILE = "products.json"
CATEGORIES_FILE = "categories.json"


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_client() -> genai.Client:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "Set GEMINI_API_KEY before running the application."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# GEMINI COMPLETION
# ============================================================

def get_completion_from_messages(
    messages: list[dict[str, str]],
    model: str = MODEL_NAME,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    max_retries: int = 5,
) -> str:

    client = get_client()

    system_parts = []
    contents = []

    # --------------------------------------------------------
    # Convert messages
    # --------------------------------------------------------

    for message in messages:

        if (
            "role" not in message
            or "content" not in message
        ):

            raise ValueError(
                f"Invalid message: {message}"
            )

        role = message["role"]
        content = str(
            message["content"]
        )

        if role == "system":

            system_parts.append(
                content
            )

        elif role == "user":

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=content
                        )
                    ],
                )
            )

        elif role == "assistant":

            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=content
                        )
                    ],
                )
            )

        else:

            raise ValueError(
                f"Unsupported role: {role}"
            )

    # --------------------------------------------------------
    # Retry Gemini request
    # --------------------------------------------------------

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(

                model=model,

                contents=contents,

                config=types.GenerateContentConfig(

                    system_instruction=(
                        "\n\n".join(
                            system_parts
                        )
                    ),

                    temperature=temperature,

                    max_output_tokens=max_tokens,

                    # IMPORTANT:
                    # Disable automatic function calling.
                    automatic_function_calling=(
                        types.AutomaticFunctionCallingConfig(
                            disable=True
                        )
                    ),
                ),
            )

            # ------------------------------------------------
            # Debug finish reason
            # ------------------------------------------------

            print(
                "\nGemini finish reason:"
            )

            try:

                finish_reason = (
                    response
                    .candidates[0]
                    .finish_reason
                )

                print(
                    finish_reason
                )

            except Exception:

                print(
                    "Unknown"
                )

            # ------------------------------------------------
            # Return generated text
            # ------------------------------------------------

            text = (
                response.text or ""
            ).strip()

            return text

        except Exception as e:

            error_text = str(
                e
            ).upper()

            temporary_error = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "SERVICE UNAVAILABLE"
                in error_text
            )

            if (
                temporary_error
                and attempt < max_retries - 1
            ):

                wait_time = (
                    2 ** attempt
                )

                print(
                    "\nGemini temporarily "
                    "unavailable."
                )

                print(
                    f"Attempt "
                    f"{attempt + 1}/"
                    f"{max_retries}"
                )

                print(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

                continue

            raise


# ============================================================
# NORMALIZE TEXT
# ============================================================

def _normalize(
    text: str
) -> str:

    text = str(
        text
    ).lower().strip()

    text = re.sub(
        r"[^a-z0-9\s]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# PHRASE MATCH
# ============================================================

def _contains_phrase(
    text: str,
    phrase: str,
) -> bool:

    text_n = _normalize(
        text
    )

    phrase_n = _normalize(
        phrase
    )

    if not phrase_n:
        return False

    return re.search(
        rf"\b{re.escape(phrase_n)}\b",
        text_n,
    ) is not None


# ============================================================
# CATEGORIES
# ============================================================

PREDEFINED_CATEGORIES = [

    "Computers and Laptops",

    "Smartphones and Accessories",

    "Televisions and Home Theater Systems",

    "Gaming Consoles and Accessories",

    "Audio Equipment",

    "Cameras and Camcorders",
]


# ============================================================
# CATEGORY ALIASES
# ============================================================

CATEGORY_ALIASES = {

    # Computers
    "computer":
        "Computers and Laptops",

    "computers":
        "Computers and Laptops",

    "laptop":
        "Computers and Laptops",

    "laptops":
        "Computers and Laptops",

    "notebook":
        "Computers and Laptops",

    # Smartphones
    "phone":
        "Smartphones and Accessories",

    "phones":
        "Smartphones and Accessories",

    "smartphone":
        "Smartphones and Accessories",

    "smartphones":
        "Smartphones and Accessories",

    "mobile":
        "Smartphones and Accessories",

    "mobiles":
        "Smartphones and Accessories",

    # TVs
    "tv":
        "Televisions and Home Theater Systems",

    "tvs":
        "Televisions and Home Theater Systems",

    "television":
        "Televisions and Home Theater Systems",

    "televisions":
        "Televisions and Home Theater Systems",

    # Gaming
    "gaming":
        "Gaming Consoles and Accessories",

    "console":
        "Gaming Consoles and Accessories",

    "consoles":
        "Gaming Consoles and Accessories",

    # Audio
    "audio":
        "Audio Equipment",

    "speaker":
        "Audio Equipment",

    "speakers":
        "Audio Equipment",

    "soundbar":
        "Audio Equipment",

    # Cameras
    "camera":
        "Cameras and Camcorders",

    "cameras":
        "Cameras and Camcorders",

    "camcorder":
        "Cameras and Camcorders",

    "camcorders":
        "Cameras and Camcorders",

    "dslr":
        "Cameras and Camcorders",
}


# ============================================================
# CREATE PRODUCTS
# ============================================================

def create_products() -> list[dict[str, Any]]:

    return [

        # ====================================================
        # SMARTPHONE
        # ====================================================

        {
            "name": "SmartX ProPhone",

            "category":
                "Smartphones and Accessories",

            "brand":
                "SmartX",

            "model_number":
                "SX-PRO-001",

            "warranty":
                "2 years",

            "rating":
                4.7,

            "features": [
                "6.7-inch AMOLED display",
                "256GB storage",
                "12GB RAM",
                "5G connectivity",
                "5000mAh battery",
            ],

            "description":
                "A premium smartphone with a "
                "high-resolution AMOLED display, "
                "powerful processor and 5G support.",

            "price":
                69999,
        },


        # ====================================================
        # TV
        # ====================================================

        {
            "name":
                "CineView 4K TV",

            "category":
                "Televisions and Home Theater Systems",

            "brand":
                "CineView",

            "model_number":
                "CV-4K-55",

            "warranty":
                "2 years",

            "rating":
                4.6,

            "features": [
                "55-inch 4K display",
                "HDR support",
                "Smart TV",
                "Dolby Audio",
                "Wi-Fi",
            ],

            "description":
                "A 55-inch 4K smart television "
                "with HDR and immersive audio.",

            "price":
                54999,
        },


        {
            "name":
                "CineView 8K TV",

            "category":
                "Televisions and Home Theater Systems",

            "brand":
                "CineView",

            "model_number":
                "CV-8K-65",

            "warranty":
                "3 years",

            "rating":
                4.8,

            "features": [
                "65-inch 8K display",
                "AI upscaling",
                "HDR10+",
                "Dolby Vision",
                "Smart TV",
            ],

            "description":
                "A premium 65-inch 8K television "
                "with AI-powered upscaling and "
                "advanced HDR.",

            "price":
                129999,
        },


        {
            "name":
                "CineView OLED TV",

            "category":
                "Televisions and Home Theater Systems",

            "brand":
                "CineView",

            "model_number":
                "CV-OLED-55",

            "warranty":
                "3 years",

            "rating":
                4.9,

            "features": [
                "55-inch OLED display",
                "Perfect blacks",
                "Dolby Vision",
                "Dolby Atmos",
                "Smart TV",
            ],

            "description":
                "An OLED smart TV designed for "
                "deep blacks, excellent contrast "
                "and cinematic viewing.",

            "price":
                99999,
        },


        # ====================================================
        # HOME THEATER
        # ====================================================

        {
            "name":
                "SoundMax Home Theater",

            "category":
                "Televisions and Home Theater Systems",

            "brand":
                "SoundMax",

            "model_number":
                "SM-HT-500",

            "warranty":
                "2 years",

            "rating":
                4.5,

            "features": [
                "5.1 channel",
                "Bluetooth",
                "Dolby Digital",
                "Wireless subwoofer",
            ],

            "description":
                "A 5.1-channel home theater "
                "system for immersive movie audio.",

            "price":
                24999,
        },


        {
            "name":
                "SoundMax Soundbar",

            "category":
                "Televisions and Home Theater Systems",

            "brand":
                "SoundMax",

            "model_number":
                "SM-SB-300",

            "warranty":
                "1 year",

            "rating":
                4.4,

            "features": [
                "3.1 channel",
                "Bluetooth",
                "HDMI ARC",
                "Wireless subwoofer",
            ],

            "description":
                "A compact soundbar system with "
                "powerful audio and wireless connectivity.",

            "price":
                14999,
        },


        # ====================================================
        # CAMERAS
        # ====================================================

        {
            "name":
                "FotoSnap DSLR Camera",

            "category":
                "Cameras and Camcorders",

            "brand":
                "FotoSnap",

            "model_number":
                "FS-DSLR-100",

            "warranty":
                "2 years",

            "rating":
                4.7,

            "features": [
                "24MP sensor",
                "4K video",
                "Interchangeable lens",
                "Optical viewfinder",
                "Wi-Fi",
            ],

            "description":
                "A versatile DSLR camera with "
                "a 24MP sensor and 4K video recording.",

            "price":
                64999,
        },


        {
            "name":
                "FotoSnap Mirrorless Camera",

            "category":
                "Cameras and Camcorders",

            "brand":
                "FotoSnap",

            "model_number":
                "FS-MIRROR-200",

            "warranty":
                "2 years",

            "rating":
                4.6,

            "features": [
                "26MP sensor",
                "4K video",
                "Electronic viewfinder",
                "Wi-Fi",
                "Bluetooth",
            ],

            "description":
                "A lightweight mirrorless camera "
                "for photography and video.",

            "price":
                72999,
        },


        {
            "name":
                "FotoSnap Instant Camera",

            "category":
                "Cameras and Camcorders",

            "brand":
                "FotoSnap",

            "model_number":
                "FS-INSTANT-10",

            "warranty":
                "1 year",

            "rating":
                4.3,

            "features": [
                "Instant printing",
                "Built-in flash",
                "Compact design",
            ],

            "description":
                "A compact instant camera designed "
                "for quick printed photos.",

            "price":
                5999,
        },


        {
            "name":
                "ActionCam 4K",

            "category":
                "Cameras and Camcorders",

            "brand":
                "ActionCam",

            "model_number":
                "AC-4K-01",

            "warranty":
                "1 year",

            "rating":
                4.5,

            "features": [
                "4K recording",
                "Water resistant",
                "Image stabilization",
                "Wide-angle lens",
            ],

            "description":
                "A rugged action camera with 4K "
                "video and electronic image stabilization.",

            "price":
                12999,
        },


        {
            "name":
                "ZoomMaster Camcorder",

            "category":
                "Cameras and Camcorders",

            "brand":
                "ZoomMaster",

            "model_number":
                "ZM-CAM-500",

            "warranty":
                "2 years",

            "rating":
                4.4,

            "features": [
                "20x optical zoom",
                "4K video",
                "Image stabilization",
                "External microphone support",
            ],

            "description":
                "A 4K camcorder with powerful "
                "optical zoom for long-distance recording.",

            "price":
                45999,
        },


        # ====================================================
        # COMPUTER
        # ====================================================

        {
            "name":
                "PowerBook Pro 15",

            "category":
                "Computers and Laptops",

            "brand":
                "PowerBook",

            "model_number":
                "PB-15-PRO",

            "warranty":
                "2 years",

            "rating":
                4.7,

            "features": [
                "15-inch display",
                "16GB RAM",
                "512GB SSD",
                "Intel processor",
            ],

            "description":
                "A high-performance laptop for "
                "productivity and professional workloads.",

            "price":
                89999,
        },


        # ====================================================
        # GAMING
        # ====================================================

        {
            "name":
                "GameBox X",

            "category":
                "Gaming Consoles and Accessories",

            "brand":
                "GameBox",

            "model_number":
                "GB-X-001",

            "warranty":
                "2 years",

            "rating":
                4.8,

            "features": [
                "4K gaming",
                "1TB storage",
                "HDR",
                "Online multiplayer",
            ],

            "description":
                "A powerful gaming console designed "
                "for 4K gaming.",

            "price":
                49999,
        },


        # ====================================================
        # AUDIO
        # ====================================================

        {
            "name":
                "SoundMax Bluetooth Speaker",

            "category":
                "Audio Equipment",

            "brand":
                "SoundMax",

            "model_number":
                "SM-SPK-100",

            "warranty":
                "1 year",

            "rating":
                4.4,

            "features": [
                "Bluetooth 5.3",
                "20-hour battery",
                "Water resistant",
                "Stereo pairing",
            ],

            "description":
                "A portable Bluetooth speaker "
                "with long battery life.",

            "price":
                7999,
        },
    ]


# ============================================================
# LOAD PRODUCTS
# ============================================================

def load_products() -> list[dict[str, Any]]:

    if os.path.exists(
        PRODUCTS_FILE
    ):

        try:

            with open(
                PRODUCTS_FILE,
                "r",
                encoding="utf-8",
            ) as f:

                data = json.load(f)

                if isinstance(
                    data,
                    list,
                ):

                    return data

        except Exception:

            pass

    products = create_products()

    try:

        with open(
            PRODUCTS_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                products,
                f,
                indent=2,
                ensure_ascii=False,
            )

    except Exception:

        pass

    return products


# ============================================================
# LOAD CATEGORIES
# ============================================================

def load_categories() -> list[str]:

    if os.path.exists(
        CATEGORIES_FILE
    ):

        try:

            with open(
                CATEGORIES_FILE,
                "r",
                encoding="utf-8",
            ) as f:

                data = json.load(f)

                if isinstance(
                    data,
                    list,
                ):

                    return data

        except Exception:

            pass

    try:

        with open(
            CATEGORIES_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                PREDEFINED_CATEGORIES,
                f,
                indent=2,
                ensure_ascii=False,
            )

    except Exception:

        pass

    return PREDEFINED_CATEGORIES


# ============================================================
# PRODUCT ALIASES
# ============================================================

PRODUCT_ALIASES = {

    "smartx pro phone":
        "SmartX ProPhone",

    "smartx prophone":
        "SmartX ProPhone",

    "smart x pro phone":
        "SmartX ProPhone",

    "fotosnap camera":
        "FotoSnap DSLR Camera",

    "fotosnap dslr":
        "FotoSnap DSLR Camera",

    "dslr camera":
        "FotoSnap DSLR Camera",

    "the dslr one":
        "FotoSnap DSLR Camera",

    "cineview tv":
        "CineView 4K TV",

    "cineview 4k":
        "CineView 4K TV",

    "cineview 8k":
        "CineView 8K TV",

    "cineview oled":
        "CineView OLED TV",

    "oled tv":
        "CineView OLED TV",

    "actioncam":
        "ActionCam 4K",

    "actioncam 4k":
        "ActionCam 4K",

    "zoommaster":
        "ZoomMaster Camcorder",

    "fotosnap mirrorless":
        "FotoSnap Mirrorless Camera",

    "fotosnap instant":
        "FotoSnap Instant Camera",

    "gamebox":
        "GameBox X",
}


# ============================================================
# PRODUCT + CATEGORY EXTRACTION
# ============================================================

def find_category_and_product_only(
    user_input: str,
) -> list[dict[str, Any]]:

    query = _normalize(
        user_input
    )

    products = load_products()

    products_by_name = {
        product["name"]: product
        for product in products
        if product.get("name")
    }

    # --------------------------------------------------------
    # Build product lookup
    # --------------------------------------------------------

    product_lookup = {}

    for product in products:

        name = product.get(
            "name",
            "",
        )

        if name:

            product_lookup[
                _normalize(name)
            ] = name

    for alias, product_name in PRODUCT_ALIASES.items():

        product_lookup[
            _normalize(alias)
        ] = product_name

    # --------------------------------------------------------
    # STEP 1
    # Find specific products FIRST
    # --------------------------------------------------------

    matched_product_names = []

    aliases_sorted = sorted(
        product_lookup.keys(),
        key=len,
        reverse=True,
    )

    for alias in aliases_sorted:

        if _contains_phrase(
            query,
            alias,
        ):

            product_name = (
                product_lookup[alias]
            )

            if (
                product_name
                not in matched_product_names
            ):

                matched_product_names.append(
                    product_name
                )

    # --------------------------------------------------------
    # Map products to categories
    # --------------------------------------------------------

    matched_categories = defaultdict(
        list
    )

    for product_name in matched_product_names:

        product = products_by_name.get(
            product_name
        )

        if not product:
            continue

        category = product.get(
            "category",
            "Unknown",
        )

        if (
            product_name
            not in matched_categories[category]
        ):

            matched_categories[
                category
            ].append(
                product_name
            )

    # --------------------------------------------------------
    # STEP 2
    # Category matching
    # --------------------------------------------------------

    category_matches = set()

    generic_camera_words = {
        "camera",
        "cameras",
        "camcorder",
        "camcorders",
        "dslr",
    }

    for alias, category in CATEGORY_ALIASES.items():

        if not _contains_phrase(
            query,
            alias,
        ):
            continue

        # Check if user already specified a product
        # from this category.

        specific_product_exists = any(

            products_by_name.get(
                product_name,
                {}
            ).get(
                "category"
            ) == category

            for product_name
            in matched_product_names
        )

        # ----------------------------------------------------
        # Important fix:
        #
        # "FotoSnap Camera, the DSLR one"
        #
        # should NOT mean:
        #
        # all Cameras and Camcorders
        #
        # because a specific camera was already found.
        # ----------------------------------------------------

        if (
            alias in generic_camera_words
            and specific_product_exists
        ):

            continue

        category_matches.add(
            category
        )

    # --------------------------------------------------------
    # STEP 3
    # Add all products for explicit categories
    # --------------------------------------------------------

    for category in category_matches:

        for product in products:

            if (
                product.get("category")
                != category
            ):

                continue

            name = product.get(
                "name"
            )

            if not name:
                continue

            if (
                name
                not in matched_categories[
                    category
                ]
            ):

                matched_categories[
                    category
                ].append(
                    name
                )

    # --------------------------------------------------------
    # STEP 4
    # Build result in predefined category order
    # --------------------------------------------------------

    results = []

    for category in PREDEFINED_CATEGORIES:

        if category not in matched_categories:
            continue

        names = matched_categories[
            category
        ]

        if not names:
            continue

        results.append(
            {
                "category":
                    category,

                "products":
                    names,
            }
        )

    # Add any other categories
    for category, names in matched_categories.items():

        if (
            category
            in PREDEFINED_CATEGORIES
        ):
            continue

        if names:

            results.append(
                {
                    "category":
                        category,

                    "products":
                        names,
                }
            )

    return results


# ============================================================
# COMPATIBILITY
# ============================================================

def read_string_to_list(
    text: str
) -> list[str]:

    if not text:
        return []

    try:

        data = json.loads(
            text
        )

        if isinstance(
            data,
            list,
        ):

            return data

    except Exception:

        pass

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


# ============================================================
# GET PRODUCT INFORMATION
# ============================================================

def get_mentioned_product_info(
    extracted_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    products = load_products()

    product_map = {
        product["name"]:
            product

        for product in products

        if product.get("name")
    }

    result = []

    for item in extracted_items:

        names = item.get(
            "products",
            [],
        )

        for name in names:

            product = product_map.get(
                name
            )

            if not product:
                continue

            # Avoid duplicate products
            if any(
                existing.get("name")
                == name
                for existing in result
            ):
                continue

            result.append(
                product.copy()
            )

    return result


# ============================================================
# FULL PRODUCT OUTPUT
# ============================================================

def generate_output_string(
    product_information: list[dict[str, Any]],
) -> str:

    if not product_information:
        return ""

    output = []

    for product in product_information:

        output.append(
            json.dumps(
                product,
                indent=2,
                ensure_ascii=False,
            )
        )

    return "\n\n".join(
        output
    )


# ============================================================
# COMPACT PRODUCT CONTEXT
# ============================================================

def generate_compact_product_context(
    product_information: list[dict[str, Any]],
) -> str:

    if not product_information:
        return ""

    lines = []

    for product in product_information:

        name = product.get(
            "name",
            "",
        )

        category = product.get(
            "category",
            "",
        )

        brand = product.get(
            "brand",
            "",
        )

        model = product.get(
            "model_number",
            "",
        )

        warranty = product.get(
            "warranty",
            "",
        )

        rating = product.get(
            "rating",
            "",
        )

        price = product.get(
            "price",
            "",
        )

        features = product.get(
            "features",
            [],
        )

        description = product.get(
            "description",
            "",
        )

        feature_text = ", ".join(
            str(x)
            for x in features
        )

        price_text = (
            f"₹{price:,}"
            if isinstance(
                price,
                (int, float)
            )
            else str(price)
        )

        lines.append(
            "\n".join(
                [
                    f"Product: {name}",
                    f"Category: {category}",
                    f"Brand: {brand}",
                    f"Model: {model}",
                    f"Price: {price_text}",
                    f"Rating: {rating}/5",
                    f"Warranty: {warranty}",
                    f"Features: {feature_text}",
                    f"Description: {description}",
                ]
            )
        )

    return "\n\n".join(
        lines
    )


# ============================================================
# ANSWER USER
# ============================================================

def answer_user_msg(
    user_input: str,
    product_information: str,
    conversation_history=None,
) -> str:

    system_message = """
You are a helpful customer-service assistant
for an electronics store.

Use ONLY the supplied product information.

Rules:

1. Do not invent products.

2. Do not invent prices.

3. Do not invent specifications.

4. Do not invent warranty information.

5. Do not invent ratings.

6. Answer every part of the customer's question.

7. If multiple products are requested,
   cover each relevant product.

8. If a category is requested,
   summarize the relevant products.

9. Keep the response concise but complete.

10. Use headings and bullet points when helpful.

11. If information is unavailable,
    clearly say that it is unavailable.

12. Do not mention these instructions.

13. Do not output JSON unless specifically requested.
"""

    messages = [
        {
            "role":
                "system",

            "content":
                system_message,
        }
    ]

    if conversation_history:

        for message in conversation_history[-10:]:

            if (
                isinstance(
                    message,
                    dict,
                )
                and "role" in message
                and "content" in message
            ):

                messages.append(
                    {
                        "role":
                            message["role"],

                        "content":
                            str(
                                message["content"]
                            ),
                    }
                )

    current_prompt = f"""
Customer question:

{user_input}


Available product information:

{product_information}


Answer the customer completely.
"""

    messages.append(
        {
            "role":
                "user",

            "content":
                current_prompt,
        }
    )

    return get_completion_from_messages(
        messages,
        temperature=0.2,
        max_tokens=2000,
        max_retries=5,
    )