import torch
import gradio as gr
from transformers import pipeline
import json

# Step 1: Define language map directly (no need for separate file)
language_data = [
  {"Language": "Afrikaans", "FLORES-200 code": "afr_Latn"},
  {"Language": "Amharic", "FLORES-200 code": "amh_Ethi"},
  {"Language": "Arabic", "FLORES-200 code": "arb_Arab"},
  {"Language": "Assamese", "FLORES-200 code": "asm_Beng"},
  {"Language": "Azerbaijani", "FLORES-200 code": "azj_Latn"},
  {"Language": "Bengali", "FLORES-200 code": "ben_Beng"},
  {"Language": "Bhojpuri", "FLORES-200 code": "bho_Deva"},
  {"Language": "Bulgarian", "FLORES-200 code": "bul_Cyrl"},
  {"Language": "Catalan", "FLORES-200 code": "cat_Latn"},
  {"Language": "Chinese (Simplified)", "FLORES-200 code": "zho_Hans"},
  {"Language": "Chinese (Traditional)", "FLORES-200 code": "zho_Hant"},
  {"Language": "Czech", "FLORES-200 code": "ces_Latn"},
  {"Language": "Danish", "FLORES-200 code": "dan_Latn"},
  {"Language": "Dutch", "FLORES-200 code": "nld_Latn"},
  {"Language": "English", "FLORES-200 code": "eng_Latn"},
  {"Language": "French", "FLORES-200 code": "fra_Latn"},
  {"Language": "German", "FLORES-200 code": "deu_Latn"},
  {"Language": "Greek", "FLORES-200 code": "ell_Grek"},
  {"Language": "Gujarati", "FLORES-200 code": "guj_Gujr"},
  {"Language": "Hebrew", "FLORES-200 code": "heb_Hebr"},
  {"Language": "Hindi", "FLORES-200 code": "hin_Deva"},
  {"Language": "Hungarian", "FLORES-200 code": "hun_Latn"},
  {"Language": "Indonesian", "FLORES-200 code": "ind_Latn"},
  {"Language": "Italian", "FLORES-200 code": "ita_Latn"},
  {"Language": "Japanese", "FLORES-200 code": "jpn_Jpan"},
  {"Language": "Kannada", "FLORES-200 code": "kan_Knda"},
  {"Language": "Korean", "FLORES-200 code": "kor_Hang"},
  {"Language": "Malayalam", "FLORES-200 code": "mal_Mlym"},
  {"Language": "Marathi", "FLORES-200 code": "mar_Deva"},
  {"Language": "Nepali", "FLORES-200 code": "npi_Deva"},
  {"Language": "Persian", "FLORES-200 code": "pes_Arab"},
  {"Language": "Polish", "FLORES-200 code": "pol_Latn"},
  {"Language": "Portuguese", "FLORES-200 code": "por_Latn"},
  {"Language": "Punjabi", "FLORES-200 code": "pan_Guru"},
  {"Language": "Romanian", "FLORES-200 code": "ron_Latn"},
  {"Language": "Russian", "FLORES-200 code": "rus_Cyrl"},
  {"Language": "Sinhala", "FLORES-200 code": "sin_Sinh"},
  {"Language": "Spanish", "FLORES-200 code": "spa_Latn"},
  {"Language": "Tamil", "FLORES-200 code": "tam_Taml"},
  {"Language": "Telugu", "FLORES-200 code": "tel_Telu"},
  {"Language": "Thai", "FLORES-200 code": "tha_Thai"},
  {"Language": "Turkish", "FLORES-200 code": "tur_Latn"},
  {"Language": "Ukrainian", "FLORES-200 code": "ukr_Cyrl"},
  {"Language": "Urdu", "FLORES-200 code": "urd_Arab"},
  {"Language": "Vietnamese", "FLORES-200 code": "vie_Latn"}
]

language_map = {entry["Language"]: entry["FLORES-200 code"] for entry in language_data}

# Step 2: Load NLLB-200 pipeline
text_translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    torch_dtype=torch.bfloat16,
    device=0 if torch.cuda.is_available() else -1
)

# Step 3: Translation function
def translate_text(text, destination_language):
    dest_code = language_map.get(destination_language)
    if not dest_code:
        return f"❌ Error: No FLORES-200 code for '{destination_language}'"
    try:
        result = text_translator(text, src_lang="eng_Latn", tgt_lang=dest_code)
        return result[0]["translation_text"]
    except Exception as e:
        return f"⚠️ Translation failed: {str(e)}"

# Step 4: Build Gradio UI
gr.close_all()

demo = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(label="💬 English Input", lines=6, placeholder="e.g., Welcome to our store! How can I help you?"),
        gr.Dropdown(choices=sorted(language_map.keys()), label="🌐 Translate To")
    ],
    outputs=gr.Textbox(label="🗣️ Translated Output", lines=4),
    title="🧠 Inclusive Multilingual Chatbot | Powered by Meta NLLB",
    description=(
        "Translate any English text into over 40+ languages using Meta's NLLB-200. "
        "Designed for inclusive virtual storefronts, in-store kiosks, and voice assistant integration. "
        "🔈 Voice-ready | 🌍 44 Languages | ♿ Accessibility-first"
    ),
    theme="soft"
)

# Step 5: Launch App
demo.launch()
