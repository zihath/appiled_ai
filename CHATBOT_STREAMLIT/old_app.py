import os
import tempfile
import streamlit as st
from PIL import Image, UnidentifiedImageError # Import UnidentifiedImageError for specific handling if needed
import io

# Google Gemini SDK
# Use the standard import convention
# from google import genai
import google.generativeai as genai
# from google.genai import types # types is usually not needed for basic generation

# Your segmentation helpers
# Make sure these files are in the same directory or your PYTHONPATH
try:
    from run_inference import load_segmentation_model, run_segmentation_pipeline
    from segmentation_utils import load_and_preprocess_image # Assuming this is where the error originates from Keras call
except ImportError as e:
    st.error(f"Failed to import helper modules: {e}. Make sure 'run_inference.py' and 'segmentation_utils.py' are accessible.")
    st.stop() # Stop execution if imports fail

API_KEY = "AIzaSyCvDUF2BMLevfin1b5VS-zEbC49X4M_WfY"
if not API_KEY or "YOUR_API_KEY" in API_KEY:
     st.error("Please configure your Google API Key using Streamlit secrets or environment variables.")
     # You might want to stop execution here if the key is missing/invalid
     # st.stop()
else:
    try:
        # Configure the Gemini client (Correct way)
        genai.configure(api_key=API_KEY)
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
        st.stop()


TEXT_MODEL   = "gemini-1.5-flash" # Or "gemini-1.0-pro" if 1.5-flash isn't available or needed
VISION_MODEL = "gemini-1.5-pro"   # Or "gemini-pro-vision" which is the standard vision model

# Initialize Gemini Models (Correct way)
try:
    text_model_instance = genai.GenerativeModel(TEXT_MODEL)
    vision_model_instance = genai.GenerativeModel(VISION_MODEL)
except Exception as e:
    st.error(f"Failed to initialize Gemini models: {e}")
    # Handle the error appropriately, maybe fall back to defaults or stop
    text_model_instance = None
    vision_model_instance = None
    st.stop()


# Cache the segmentation model loading
@st.cache_resource(show_spinner="Loading segmentation model...")
def load_tf_model():
    """Load your TF segmentation model once."""
    try:
        seg_model = load_segmentation_model()
        print("Segmentation Model Loaded!!!")
        return seg_model
    except Exception as e:
        st.error(f"Error loading segmentation model: {e}")
        return None

def format_stats(area_stats: dict) -> str:
    """Turn the area_stats dict into a neat prompt snippet."""
    if not area_stats:
        return "No segmentation stats available."
    lines = [f"- {cls}: {info.get('pixels', 'N/A')} px ({info.get('percent', 'N/A')}%)"
             for cls, info in area_stats.items()]
    return "\n".join(lines)

def main():
    st.title("🛰️ Automated Segmentation + Gemini Chatbot")

    st.warning("⚠️ Using a hardcoded API Key. For production, use Streamlit secrets or environment variables.")

    st.write("""
    1. Upload an RGB satellite image.
    2. The app will segment it, compute class-wise pixel counts & area %.
    3. Ask any question—Gemini will answer using the stats and/or image+mask.
    """)

    # 1) Upload just the original image
    uploaded_file = st.file_uploader("Upload Satellite Image", type=["png", "jpg", "jpeg"])
    if not uploaded_file:
        st.info("Please upload a satellite image to begin.")
        return # Stop execution if no file is uploaded

    # Read uploaded file only once
    try:
        uploaded_bytes = uploaded_file.read()
        # Use BytesIO to open the image with PIL
        image_stream = io.BytesIO(uploaded_bytes)
        image = Image.open(image_stream).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)
    except UnidentifiedImageError:
        st.error("Could not read the uploaded file. It might be corrupted or not a valid image.")
        return
    except Exception as e:
        st.error(f"An error occurred while processing the uploaded image: {e}")
        return

    # Load the segmentation model (cached)
    model = load_tf_model()
    if model is None:
        st.error("Segmentation model failed to load. Cannot proceed.")
        return

    # --- Corrected Temporary File Handling ---
    tmp_file_path = None
    try:
        # Create a temporary file, write to it, and CLOSE it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", mode="wb") as tmp_file:
            tmp_file.write(uploaded_bytes)
            tmp_file_path = tmp_file.name # Get the path

        # Now tmp_file is closed, and data is flushed to disk.
        # Pass the PATH to the segmentation pipeline.
        st.write("Running segmentation...")
        with st.spinner("Segmenting image..."):
            rgb_mask, area_stats = run_segmentation_pipeline(
                image_path=tmp_file_path, # Use the path
                model=model,
                save_vis=False,
                save_json=False
            )

        # Show predicted mask
        st.image(rgb_mask, caption="Predicted Mask", use_container_width=True)

        # Show stats table
        st.write("### Class-wise Segmentation Stats")
        if area_stats:
            stats_df_data = {
                 "Class": list(area_stats.keys()),
                 "Pixels": [v.get("pixels", "N/A") for v in area_stats.values()],
                 "Percent": [v.get("percent", "N/A") for v in area_stats.values()]
            }
            st.dataframe(stats_df_data)
        else:
            st.info("No area statistics were generated.")

    except FileNotFoundError:
        st.error(f"Temporary file not found at path: {tmp_file_path}. Check file writing permissions.")
        return # Stop if file vanished somehow
    except Exception as e:
        st.error(f"An error occurred during segmentation: {e}")
        # Optionally display more traceback if needed for debugging
        # st.exception(e)
        return # Stop if segmentation fails
    finally:
        # Clean up the temporary file in a finally block
        # to ensure it runs even if errors occur above.
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
                print(f"Temporary file {tmp_file_path} deleted.")
            except PermissionError:
                st.warning(f"Could not delete temporary file {tmp_file_path}. It might be locked by another process (e.g., antivirus).")
            except Exception as e_unlink:
                 st.warning(f"Error deleting temporary file {tmp_file_path}: {e_unlink}")
    # --- End of Corrected Temporary File Handling ---


    # 3) Ask a question
    user_q = st.text_input("Ask a question about this image or its segmentation:")

    # Choose mode
    mode = st.selectbox("Response mode:", ["Text Stats Only", "Vision+Text Only", "Both"], index=2) # Default to Both

    if st.button("👉 Analyze with Gemini"):
        if not user_q.strip():
            st.error("Please type a question.")
            return
        if text_model_instance is None or vision_model_instance is None:
             st.error("Gemini models are not available. Cannot generate response.")
             return

        # Prepare data for Gemini
        stats_prompt_text = format_stats(area_stats if 'area_stats' in locals() else {}) # Handle case where stats failed
        base_text_prompt = (
            "You are analyzing a satellite image and its segmentation results.\n\n"
            "Segmentation statistics (pixels & % coverage):\n"
            f"{stats_prompt_text}\n\n"
            "Based on the available information (stats, image, mask), answer the following question.\n"
            f"Question: {user_q}"
        )

        # Convert predicted mask (numpy array) to PIL Image for Vision model
        try:
            mask_image = Image.fromarray(rgb_mask) # rgb_mask should be a numpy array
        except NameError:
             st.error("Segmentation mask (rgb_mask) is not available. Cannot use Vision model.")
             mask_image = None # Set to None if mask isn't ready
             if mode != "Text Stats Only":
                 st.warning("Switching to 'Text Stats Only' mode as the mask image is missing.")
                 mode = "Text Stats Only"
        except Exception as e:
            st.error(f"Could not convert segmentation mask to image: {e}")
            mask_image = None
            if mode != "Text Stats Only":
                 st.warning("Switching to 'Text Stats Only' mode due to mask conversion error.")
                 mode = "Text Stats Only"


        # 3a) Text-only model
        if mode in ("Text Stats Only", "Both"):
            st.subheader("💬 Answer from Text Model")
            with st.spinner("Querying text model..."):
                try:
                    # Correct Gemini API call
                    resp = text_model_instance.generate_content(base_text_prompt)
                    st.markdown(resp.text) # Use markdown for better formatting
                except Exception as e:
                    st.error(f"Error querying Text model ({TEXT_MODEL}): {e}")

        # 3b) Vision+Text model
        if mode in ("Vision+Text Only", "Both"):
            st.subheader("🖼️ Answer from Vision+Text Model")
            if image and mask_image: # Make sure both images are available
                with st.spinner("Querying vision model..."):
                    try:
                        # Correct Gemini API call with multiple parts (text, image, mask)
                        vision_prompt_parts = [
                            "You are analyzing a satellite image and its segmentation mask.\n"
                            "The first image is the original satellite photo.\n"
                            "The second image is the segmentation mask showing different land cover types.\n"
                            f"Question: {user_q}\n\n"
                            "Analyze both images and answer the question.",
                            image,  # Original PIL image
                            mask_image # Mask PIL image
                        ]
                        resp_v = vision_model_instance.generate_content(vision_prompt_parts)
                        st.markdown(resp_v.text) # Use markdown
                    except Exception as e:
                        st.error(f"Error querying Vision model ({VISION_MODEL}): {e}")
            else:
                st.warning("Original image or mask image is missing. Cannot query Vision model.")


if __name__ == "__main__":
    main()