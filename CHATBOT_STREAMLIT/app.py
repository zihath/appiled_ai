import os # Ensure os is imported early if used early
import tempfile
import streamlit as st
st.set_page_config(layout="wide") # MOVED HERE

from PIL import Image, UnidentifiedImageError
import io
import time # To create unique keys for file uploader if needed
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env")
except ImportError:
    st.warning("`python-dotenv` library not found. Skipping .env file loading. Make sure GOOGLE_API_KEY is set in your environment.", icon="⚠️")
    # You can proceed without dotenv if the key is set globally

# --- Other Imports ---
import google.generativeai as genai
try:
    from run_inference import load_segmentation_model, run_segmentation_pipeline
except ImportError as e:
    st.error(f"Failed to import helper modules: {e}. Make sure 'run_inference.py' and related files are accessible.")
    st.stop()

# --- API Key Configuration ---
API_KEY = os.environ.get("GOOGLE_API_KEY") # Now os is definitely imported

if not API_KEY:
    st.error("Missing `GOOGLE_API_KEY` environment variable. Please set it (e.g., in your .env file or system environment).")
    st.stop()
else:
    try:
        genai.configure(api_key=API_KEY)
        print("Gemini API Configured.") # Optional: print confirmation
    except Exception as e:
        st.error(f"❌ Failed to configure Gemini API: {e}. Check your API key and network connection.")
        st.stop()

# --- Model Configuration ---
TEXT_MODEL   = "gemini-1.5-flash"
VISION_MODEL = "gemini-1.5-pro"

# Initialize Gemini Models
try:
    text_model_instance = genai.GenerativeModel(TEXT_MODEL)
    vision_model_instance = genai.GenerativeModel(VISION_MODEL)
    print("Gemini Models Initialized.") # Optional
except Exception as e:
    st.error(f"Failed to initialize Gemini models: {e}")
    st.stop()

# Segmentation Model Loading
@st.cache_resource(show_spinner="Loading segmentation model...")
def load_tf_model():
    """Load your TF segmentation model once."""
    try:
        seg_model = load_segmentation_model()
        print("Segmentation Model Loaded!!!")
        return seg_model
    except Exception as e:
        # Using st.error here is fine now
        st.error(f"Error loading segmentation model: {e}")
        return None

# Load the segmentation model
segmentation_model = load_tf_model()
if segmentation_model is None:
    # st.error is already called inside load_tf_model on failure
    st.info("Stopping execution because segmentation model failed to load.")
    st.stop()

# --- Helper Functions ---
def format_stats(area_stats: dict) -> str:
    """Turn the area_stats dict into a neat prompt snippet."""
    if not area_stats:
        return "No segmentation stats available."
    lines = [f"- {cls}: {info.get('pixels', 'N/A')} px ({info.get('percent', 'N/A')}%)"
             for cls, info in area_stats.items()]
    return "\n".join(lines)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image_data" not in st.session_state:
    st.session_state.current_image_data = None
if "analysis_mode" not in st.session_state:
     st.session_state.analysis_mode = "Both"

st.title("🛰️ Automated Segmentation + Gemini Chatbot")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Image Upload & Segmentation")
    # Consider using a more stable key if time-based causes issues
    uploaded_file = st.file_uploader(
        "Upload Satellite Image",
        type=["png", "jpg", "jpeg"],
        key="file_uploader" # Simpler key might be sufficient
        )

    st.session_state.analysis_mode = st.radio(
        "Select Analysis Mode:",
        ["Text Stats Only", "Vision+Text Only", "Both"],
        key='analysis_mode_selector',
        index=["Text Stats Only", "Vision+Text Only", "Both"].index(st.session_state.analysis_mode)
    )

    # --- Image Processing Logic ---
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.current_image_data is None or st.session_state.current_image_data.get("id") != file_id:
            st.info("New image detected. Processing...")
            with st.spinner("Reading image and running segmentation..."):
                tmp_file_path = None
                try:
                    # ... (rest of the image processing code remains the same) ...
                    uploaded_bytes = uploaded_file.read()
                    image_pil = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", mode="wb") as tmp_file:
                        tmp_file.write(uploaded_bytes)
                        tmp_file_path = tmp_file.name
                    rgb_mask_np, area_stats_dict = run_segmentation_pipeline(
                        image_path=tmp_file_path, model=segmentation_model, save_vis=False, save_json=False
                    )
                    st.session_state.current_image_data = {
                        "id": file_id, "name": uploaded_file.name, "image": image_pil,
                        "mask": rgb_mask_np, "stats": area_stats_dict
                    }
                    st.session_state.messages = []
                    st.success("Image processed and segmented!")
                except UnidentifiedImageError:
                    st.error("Could not read the uploaded file. Invalid format or corrupted.")
                    st.session_state.current_image_data = None
                except Exception as e:
                    st.error(f"An error occurred during image processing or segmentation: {e}")
                    st.session_state.current_image_data = None
                finally:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        try:
                            os.unlink(tmp_file_path)
                        except Exception as e_unlink:
                            st.warning(f"Could not delete temporary file {tmp_file_path}: {e_unlink}")

    # --- Display Current Image/Mask/Stats ---
    if st.session_state.current_image_data:
        st.image(st.session_state.current_image_data['image'], caption="Original Image", use_container_width=True)
        st.image(st.session_state.current_image_data['mask'], caption="Predicted Mask", use_container_width=True)
        st.write("##### Class-wise Segmentation Stats")
        current_stats = st.session_state.current_image_data.get('stats', {})
        if current_stats:
            stats_df_data = {
                 "Class": list(current_stats.keys()),
                 "Pixels": [v.get("pixels", "N/A") for v in current_stats.values()],
                 "Percent": [v.get("percent", "N/A") for v in current_stats.values()]
            }
            st.dataframe(stats_df_data, use_container_width=True)
        else:
            st.info("No area statistics available for this image.")
    else:
        st.info("Upload a satellite image using the panel on the left.")


with col2:
    st.subheader("Chat Analysis")

    if not st.session_state.current_image_data:
        st.info("Upload and process an image first to enable chat.")
        st.stop()

    # --- Display Chat History ---
    st.write("##### Conversation History")
    chat_container = st.container(height=400)
    with chat_container:
        # ... (chat history display remains the same) ...
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                elif message["role"] == "assistant":
                    if message.get("text_response"):
                        st.markdown("**Text Model Answer:**\n" + message["text_response"])
                    if message.get("vision_response"):
                         st.markdown("**Vision Model Answer:**\n" + message["vision_response"])
                    if message.get("error"):
                        st.error(message["error"])
                    if not message.get("text_response") and not message.get("vision_response") and not message.get("error"):
                        st.markdown("_No response generated for this turn._")


    # --- Chat Input ---
    if prompt := st.chat_input("Ask a question about the current image..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()


# --- Process the latest user message ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    # ... (Gemini call logic remains the same) ...
    last_user_message = st.session_state.messages[-1]["content"]
    image_context = st.session_state.current_image_data['image']
    mask_context_np = st.session_state.current_image_data['mask']
    stats_context = st.session_state.current_image_data['stats']
    current_mode = st.session_state.analysis_mode

    mask_context_pil = None
    try:
        mask_context_pil = Image.fromarray(mask_context_np)
    except Exception as e:
        st.error(f"Failed to convert mask for vision model: {e}")
        st.session_state.messages.append({
            "role": "assistant",
            "error": f"Internal error: Could not process segmentation mask ({e}). Vision analysis unavailable."
        })
        st.rerun()

    if mask_context_pil or current_mode == "Text Stats Only":
        stats_prompt_text = format_stats(stats_context)
        base_text_prompt = (
            "You are an AI assistant analyzing a satellite image and its segmentation results.\n\n"
            "SEGMENTATION STATISTICS:\n"
            f"{stats_prompt_text}\n\n"
            "Based *only* on the statistics provided above, answer the following question.\n"
            f"Question: {last_user_message}"
        )
        vision_prompt_parts = None
        if mask_context_pil and image_context:
             vision_prompt_parts = [
                "You are an AI assistant analyzing a satellite image and its segmentation mask...\n" # Truncated for brevity
                f"Question: {last_user_message}\n\n"
                "Analyze *both* images visually...",
                image_context, mask_context_pil
            ]

        assistant_responses = {"role": "assistant", "text_response": None, "vision_response": None, "error": None}
        try:
            with st.spinner(f"Analyzing with Gemini ({current_mode})..."):
                if current_mode in ("Text Stats Only", "Both"):
                    if text_model_instance:
                        resp = text_model_instance.generate_content(base_text_prompt)
                        # Simple check for safety blocking (can be more sophisticated)
                        if resp.parts:
                             assistant_responses["text_response"] = resp.text
                        else:
                             assistant_responses["text_response"] = "_Response blocked by safety settings or model error._"
                    else:
                         assistant_responses["error"] = "Text model not initialized."
                if current_mode in ("Vision+Text Only", "Both"):
                    if vision_model_instance and vision_prompt_parts:
                        resp_v = vision_model_instance.generate_content(vision_prompt_parts)
                        if resp_v.parts:
                            assistant_responses["vision_response"] = resp_v.text
                        else:
                            assistant_responses["vision_response"] = "_Response blocked by safety settings or model error._"
                    elif not vision_prompt_parts and current_mode != "Text Stats Only":
                         assistant_responses["vision_response"] = "_Vision analysis skipped: Missing valid image or mask._"
                    elif current_mode != "Text Stats Only":
                         assistant_responses["error"] = (assistant_responses.get('error', '') + " Vision model not initialized.").strip()
        except Exception as e:
            error_message = f"An error occurred calling Gemini: {e}"
            st.error(error_message) # Show error immediately
            assistant_responses["error"] = error_message

        st.session_state.messages.append(assistant_responses)
        st.rerun()