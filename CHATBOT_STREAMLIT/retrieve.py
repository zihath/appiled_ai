import os
import json
import pandas as pd
# from google import genai # Not needed if using google.generativeai as genai
import google.generativeai as genai # Correct import
from sentence_transformers import SentenceTransformer
import streamlit as st
import numpy as np

# Optional: Load environment variables (e.g., your Gemini API key)
from dotenv import load_dotenv
load_dotenv()
# Use GOOGLE_API_KEY as the standard environment variable name
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found.")
    st.markdown("Please set the `GOOGLE_API_KEY` environment variable.")
    st.markdown("If you are using a `.env` file, make sure it contains `GOOGLE_API_KEY='YOUR_API_KEY'`.")
    st.stop() # Stop the app if API key is missing


# --- Configure Gemini API ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Optional: Check if API key is valid by listing models
    # for m in genai.list_models():
    #     print(m.name) # Uncomment to see available models on startup
    st.success("Gemini API configured successfully.")
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.warning("Please ensure your `GOOGLE_API_KEY` environment variable is set correctly and is valid.")
    st.stop() # Stop the app if API configuration fails


# --- 1. Load image stats into a DataFrame ---
# Assuming your data_set.json has the structure:
# {
#   "filename1.png": { "ClassA": 10.5, "ClassB": 89.5, ... },
#   "filename2.jpg": { "ClassA": 5.0, "ClassB": 95.0, ... },
#   ...
# }

try:
    with open("data_set.json", "r") as f:
        data_dict = json.load(f) # Load the dictionary directly

    # Build DataFrame from the dictionary structure
    rows = []
    for filename, class_stats in data_dict.items():
        # Create a row dictionary starting with the image path
        # Assuming filename in JSON is the relative path to the image file
        row = {"image_path": filename}
        # Add class percentages directly from the nested dictionary
        for cls, percent in class_stats.items():
            row[cls] = percent
        rows.append(row)

    df = pd.DataFrame(rows)

    # Fill any potentially missing classes with 0.0 (if some images lack a class entirely)
    df = df.fillna(0.0)

    # Precompute text summaries for each image (e.g. "land 55%, water 0.7%, road 24%")
    # Get the list of class columns (excluding 'image_path')
    classes = [col for col in df.columns if col != "image_path"]

    # Define the function to create the summary string
    def make_summary(row):
        # Ensure we only include classes that exist in the row and have non-zero percentages
        # Sort classes for consistent summary string order
        parts = [f"{cls} {row[cls]:.1f}%" for cls in sorted(classes) if cls in row and row[cls] > 0]
        return ", ".join(parts) if parts else "No classes identified" # Handle cases with 0 total pixels or only background

    # Apply the function to create the 'summary' column
    df["summary"] = df.apply(make_summary, axis=1)

    st.success(f"Successfully loaded stats for {len(df)} images.")

except FileNotFoundError:
    st.error("Error: data_set.json not found. Please ensure the file exists in the same directory as the script.")
    st.stop() # Stop the app if the data file is missing
except json.JSONDecodeError:
    st.error("Error: Could not decode data_set.json. Please ensure it's valid JSON.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading data: {e}")
    st.stop()


# Load sentence transformer model (for semantic similarity)
# This model is relatively small and runs locally
@st.cache_resource # Cache the model to avoid reloading on each rerun
def load_sentence_transformer():
    try:
        # Add a spinner to show loading is happening
        with st.spinner("Loading Sentence Transformer model..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")
        st.success("Sentence Transformer model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model: {e}")
        st.warning("Please ensure you have an internet connection to download the model on first run.")
        st.stop()

model = load_sentence_transformer()


@st.cache_data # Cache the results of the API call for a given query
def parse_query_to_filter(query_text): # Renamed query to query_text to avoid conflict with global query variable
    """
    Use the Gemini API to convert a natural language query string into a pandas query string.
    We instruct Gemini that df has columns for each class (e.g., 'land', 'water', etc.).
    """
    cols = ", ".join(classes)
    # Added more specific instructions to help Gemini
    prompt = (
        f"You are an expert in converting natural language image descriptions "
        f"to pandas DataFrame filter expressions based on image segmentation statistics.\n"
        f"The DataFrame `df` contains image statistics. It has columns: 'image_path', 'summary', and columns for each segmentation class: {cols}. "
        f"Each class column represents the percentage of the image area covered by that class (float, 0.0 to 100.0).\n"
        f"Your task is to convert the following image query into a pandas filter expression that can be used with `df.query()`. "
        f"Only use the class column names ({cols}) and standard numerical/logical operators (>, <, >=, <=, ==, !=, &, |, ~, (, )).\n"
        f"Do NOT use any other columns like 'image_path' or 'summary' in the filter expression.\n"
        f"Do NOT include `df.query()` or any other Python code. Output ONLY the boolean condition.\n"
        f"Example: Query 'more land than water' -> Output: `land > water`\n"
        f"Example: Query 'buildings cover more than 5%' -> Output: `building > 5.0`\n"
        f"Example: Query 'road present' -> Output: `road > 0.0`\n" # Assuming 'present' means > 0%
        f"Example: Query 'no vegetation' -> Output: `vegetation == 0.0`\n"
        f"Convert this query:\n'{query_text}'\n"
        f"Output ONLY the boolean condition:"
    )
    try:
        # Call the API using the top-level function after genai.configure()
        response = genai.generate_content(
            model="gemini-1.5-flash-latest", # Using a stable flash model alias
            contents=prompt,
            generation_config=genai.GenerationConfig(temperature=0.0) # Low temperature for predictable output
        )
        # Added safety check for empty or malformed response
        if response and response.text:
             filter_str = response.text.strip().strip('`').strip('"').strip("'").replace('\\', '') # Clean extra markdown/quotes/slashes
             # Basic validation: ensure it doesn't contain function calls etc.
             # Added check for minimum complexity (at least one operator)
             if any(op in filter_str for op in ['>', '<', '>=', '<=', '==', '!=', '&', '|', '~', '(', ')']):
                  # Further check for obvious code injection attempts
                  if all(word not in filter_str.lower() for word in ['import ', ' os', ' sys', ' eval', ' exec', ' lambda']):
                       return filter_str
                  else:
                       st.warning(f"Generated filter looks suspicious: {filter_str}")
                       return None # Reject suspicious output
             else:
                 st.warning(f"Gemini returned simple text, not a filter condition: {filter_str}. Trying raw text.")
                 # Fallback: Sometimes the raw text part is just the condition
                 if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                      raw_text_filter = response.candidates[0].content.parts[0].text.strip().strip('`').strip('"').strip("'").replace('\\', '')
                      if any(op in raw_text_filter for op in ['>', '<', '>=', '<=', '==', '!=', '&', '|', '~', '(', ')']):
                           return raw_text_filter
                 st.error("Gemini API returned unusable output for filter.")
                 return None # Indicate failure
        else:
             raise ValueError("Gemini API returned an empty or invalid response.")

    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        st.warning("Could not parse query using AI. Please try a simpler query or check API setup.")
        return None # Indicate failure


@st.cache_data # Cache embeddings and ranking results for a given query and filtered data
def semantic_rank(query_text, candidates_df, top_k=5): # Renamed query to query_text
    """
    Compute semantic similarity between the user query string and a list of candidate summary strings.
    Return top_k rows from the candidates_df sorted by similarity.
    """
    if candidates_df.empty:
        return pd.DataFrame() # Return empty DataFrame if no candidates

    candidates_summaries = candidates_df["summary"].tolist()

    # Encode the query and candidate summaries
    try:
        # Add spinner for embedding process if it's not cached
        with st.spinner("Computing embeddings..."):
             query_emb = model.encode(query_text, convert_to_numpy=True)
             # Encode summaries in batches if dataset is huge to manage memory (optional for now)
             # cand_embs = model.encode(candidates_summaries, convert_to_numpy=True, batch_size=64) # Example batching
             cand_embs = model.encode(candidates_summaries, convert_to_numpy=True)

        st.success("Embeddings computed.")

    except Exception as e:
        st.error(f"Error during sentence embedding: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

    # Compute cosine similarity scores
    # Add small epsilon for numerical stability if needed, although norm() should handle zero vectors
    query_norm = np.linalg.norm(query_emb)
    cand_norms = np.linalg.norm(cand_embs, axis=1)

    # Avoid division by zero if query or candidate embeddings are zero vectors (unlikely but possible)
    # Replace 0 norms with 1 to avoid NaNs during division, resulting in 0 similarity
    query_norm = query_norm if query_norm > 1e-6 else 1
    cand_norms[cand_norms < 1e-6] = 1

    sims = np.dot(cand_embs, query_emb) / (cand_norms * query_norm)

    # Get top_k indices (highest cosine)
    # Use argsort for sorting and slicing
    if top_k >= len(sims):
        top_idx = np.argsort(sims)[::-1] # Sort all if k is >= total
    else:
        top_idx = np.argsort(sims)[::-1][:top_k]


    # Return the top_k rows from the original candidates_df, sorted by similarity
    # Use .iloc based on the indices of the filtered DataFrame
    return candidates_df.iloc[top_idx]


# --- 2. Streamlit interface ---
st.title("Satellite Image Retrieval")
st.write("Enter a query to filter and rank satellite images based on segmentation stats.")
st.write("Examples: `'road > 10.0'`, `'building > land'`, `'water == 0'`, `'high vegetation cover'`, `'image with forests and rivers'`") # Added a semantic example


query = st.text_input("Enter Query")
top_k = st.slider("Number of results to display (after filtering)", 1, 20, 5)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    # --- Step 1: Filter using the AI-parsed expression ---
    # Pass the query string to the parsing function
    filter_str = parse_query_to_filter(query)

    if filter_str is None: # Error occurred in parsing or output was invalid
         st.error("Failed to generate a valid filter expression from your query.")
         st.stop()

    st.write(f"Applying filter: `{filter_str}`")

    try:
        # Apply filter to DataFrame
        # Ensure the filter_str is a valid pandas query string
        df_filtered = df.query(filter_str)
        st.write(f"Filter matched {len(df_filtered)} images.")
    except Exception as e:
        st.error(f"Invalid filter expression generated or applied: `{filter_str}`. Error: {e}")
        st.warning("Please check your query or the generated filter syntax. Ensure class names match exactly.")
        st.stop()

    if df_filtered.empty:
        st.write("No images match the filter criteria.")
    else:
        # --- Step 2: Semantic ranking among filtered images ---
        # Pass the original query string and filtered DataFrame
        ranked_results_df = semantic_rank(query, df_filtered, top_k=top_k)

        if ranked_results_df.empty:
            st.write("Could not perform semantic ranking on the filtered results.")
            st.stop()

        st.subheader(f"Top {len(ranked_results_df)} Results (Ranked Semantically)")

        # --- Step 3: Display top results ---
        # Iterate through the rows of the ranked DataFrame
        for rank, (index, row) in enumerate(ranked_results_df.iterrows(), start=1):
            img_path = row["image_path"]
            # Build explanation text from the row data
            # Ensure all classes defined in 'classes' are included in the display
            stats_parts = []
            for cls in sorted(classes): # Sort for consistent display order
                 if cls in row:
                      stats_parts.append(f"{cls}: {row[cls]:.1f}%")
                 else:
                      # Should not happen with fillna(0.0), but as a safeguard
                      stats_parts.append(f"{cls}: 0.0%")

            stats_text = ", ".join(stats_parts)

            explanation = (
                f"**Rank {rank}** – {stats_text}. "
                f"Matches filter `{filter_str}`."
            )
            # Check if image file actually exists before trying to display
            if os.path.exists(img_path):
                 st.image(img_path, caption=explanation, use_column_width=True)
            else:
                 st.warning(f"Image file not found: {img_path}. Stats: {stats_text}")