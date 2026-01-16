import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import gc
import random

# Page configuration
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Title and description
st.title("🎨 AI Text-to-Image Generator")

# IMPORTANT: Check if running on Streamlit Cloud
if not torch.cuda.is_available():
    st.error("""
    ⚠️ **IMPORTANT: No GPU Detected**
    
    This app is running on CPU which makes image generation **extremely slow** (5-10 minutes per image).
    
    **For fast local usage with GPU:**
    1. Download this code to your local machine
    2. Install requirements: `pip install -r requirements.txt`
    3. Run locally: `streamlit run app.py`
    4. Your NVIDIA Quadro P1000 will generate images in ~10-30 seconds
    
    **You can still try it here on CPU, but expect long wait times.**
    """)
    
    st.markdown("---")

st.markdown("""
Generate beautiful images from text descriptions using Stable Diffusion.
This app runs completely locally using open-source models.
""")

# Device selection
@st.cache_resource
def get_device():
    """Determine the best available device (CUDA or CPU)"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.sidebar.success(f"✅ GPU Detected: {gpu_name}")
        st.sidebar.info(f"VRAM: {vram:.1f} GB")
    else:
        device = "cpu"
        st.sidebar.warning("⚠️ CPU Mode (Very Slow)")
        st.sidebar.info("Download and run locally for GPU acceleration")
    return device

device = get_device()

# Model selection with caching
@st.cache_resource
def load_model(model_name):
    """Load the Stable Diffusion model with caching"""
    try:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text("🔄 Downloading model files... (This happens only once, ~2-4 GB)")
        progress_bar.progress(20)
        
        # Use smaller model for CPU, optimized for Streamlit Cloud
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True  # Important for CPU
        )
        
        progress_bar.progress(60)
        progress_text.text("🔄 Loading model to device...")
        
        pipe = pipe.to(device)
        
        # Enable optimizations
        if device == "cuda":
            pipe.enable_attention_slicing()
        else:
            # CPU optimizations
            pipe.enable_attention_slicing(slice_size=1)
        
        progress_bar.progress(100)
        progress_text.text("✅ Model loaded successfully!")
        
        # Clear progress indicators after 2 seconds
        import time
        time.sleep(2)
        progress_text.empty()
        progress_bar.empty()
        
        st.sidebar.success("✅ Model ready!")
        return pipe
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Try refreshing the page or selecting a different model.")
        return None

# Sidebar - Model selection
st.sidebar.header("⚙️ Model Settings")

# Only offer smaller, faster models for CPU/Streamlit Cloud
if device == "cpu":
    model_options = {
        "Stable Diffusion v1.4 (Fastest, Recommended for CPU)": "CompVis/stable-diffusion-v1-4",
        "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    }
    default_index = 0
else:
    model_options = {
        "Stable Diffusion v1.5 (Recommended)": "runwayml/stable-diffusion-v1-5",
        "Stable Diffusion 2.1 Base": "stabilityai/stable-diffusion-2-1-base",
        "Stable Diffusion v1.4 (Fastest)": "CompVis/stable-diffusion-v1-4"
    }
    default_index = 0

selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    options=list(model_options.keys()),
    index=default_index
)

model_id = model_options[selected_model_name]

# Load the model
pipe = load_model(model_id)

# Sidebar - Generation parameters
st.sidebar.header("🎛️ Generation Parameters")

# Adjust defaults based on device
default_steps = 15 if device == "cpu" else 30
max_steps = 30 if device == "cpu" else 100

num_inference_steps = st.sidebar.slider(
    "Inference Steps",
    min_value=10,
    max_value=max_steps,
    value=default_steps,
    step=5,
    help="More steps = better quality but MUCH slower on CPU. 15-20 recommended for CPU."
)

guidance_scale = st.sidebar.slider(
    "Guidance Scale",
    min_value=1.0,
    max_value=20.0,
    value=7.5,
    step=0.5,
    help="How closely to follow the prompt. 7-9 is typical."
)

# Seed control
use_random_seed = st.sidebar.checkbox("Random Seed", value=True)
if use_random_seed:
    seed = random.randint(0, 2**32 - 1)
else:
    seed = st.sidebar.number_input(
        "Seed",
        min_value=0,
        max_value=2**32 - 1,
        value=42,
        help="Set a specific seed for reproducible results"
    )

# Image dimensions - force 512x512 on CPU for speed
if device == "cpu":
    width = 512
    height = 512
    st.sidebar.info("📏 Resolution: 512x512 (optimal for CPU)")
else:
    width = st.sidebar.selectbox("Width", [512, 768], index=0)
    height = st.sidebar.selectbox("Height", [512, 768], index=0)

# Show estimated time
if device == "cpu":
    estimated_time = num_inference_steps * 15  # ~15 seconds per step on CPU
    st.sidebar.warning(f"⏱️ Estimated time: ~{estimated_time//60} minutes {estimated_time%60} seconds")

# Main interface - Prompt inputs
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area(
        "✍️ Describe the image you want to generate",
        height=100,
        placeholder="Example: a serene lake at sunset, mountains in background, oil painting style",
        help="Be descriptive! Include style, mood, and details."
    )

    negative_prompt = st.text_area(
        "🚫 Negative Prompt (what to avoid)",
        height=80,
        value="blurry, bad quality, distorted, ugly",
        help="Describe what you don't want in the image"
    )

with col2:
    st.markdown("### 💡 Tips")
    if device == "cpu":
        st.markdown("""
        - **Use 15-20 steps** for CPU
        - Keep prompts simple
        - Be patient - CPU is slow
        - **Run locally with GPU** for fast results
        """)
    else:
        st.markdown("""
        - Be specific and descriptive
        - Add style keywords
        - Use quality boosters
        - 25-35 steps for quality
        """)

# Generate button
generate_button = st.button("🎨 Generate Image", type="primary", use_container_width=True)

# Image generation
if generate_button:
    if not prompt:
        st.warning("⚠️ Please enter a prompt first!")
    elif pipe is None:
        st.error("❌ Model not loaded. Please refresh the page.")
    else:
        try:
            # CPU warning
            if device == "cpu":
                st.warning(f"⏳ Generating on CPU... This will take approximately {num_inference_steps * 15 // 60} minutes. Please be patient!")
            
            # Clear cache before generation
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            # Set seed for reproducibility
            generator = torch.Generator(device=device).manual_seed(seed)
            
            # Generate image with progress
            progress_placeholder = st.empty()
            
            with st.spinner(f"🎨 Generating image... (Seed: {seed})"):
                start_time = st.empty()
                import time
                start = time.time()
                
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                ).images[0]
                
                elapsed = time.time() - start
            
            # Display the generated image
            st.success(f"✅ Image generated successfully in {elapsed:.1f} seconds!")
            
            # Show image in a nice container
            st.image(image, caption=f"Generated Image | Seed: {seed}", use_container_width=True)
            
            # Download button
            from io import BytesIO
            buf = BytesIO()
            image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="💾 Download Image",
                data=byte_im,
                file_name=f"generated_image_seed_{seed}.png",
                mime="image/png",
                use_container_width=True
            )
            
            # Show generation info
            with st.expander("ℹ️ Generation Details"):
                st.write(f"**Prompt:** {prompt}")
                st.write(f"**Negative Prompt:** {negative_prompt}")
                st.write(f"**Model:** {model_id}")
                st.write(f"**Steps:** {num_inference_steps}")
                st.write(f"**Guidance Scale:** {guidance_scale}")
                st.write(f"**Seed:** {seed}")
                st.write(f"**Dimensions:** {width}x{height}")
                st.write(f"**Device:** {device.upper()}")
                st.write(f"**Generation Time:** {elapsed:.1f} seconds")
            
            # Clear cache after generation
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                st.error("""
                ❌ **Out of Memory Error!**
                
                Try these solutions:
                1. Reduce inference steps to 15-20
                2. Use Stable Diffusion v1.4 (smallest model)
                3. Restart the app
                """)
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                st.error(f"❌ Error during generation: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### Status
- **Device:** {device.upper()}
- **Model:** {selected_model_name.split('(')[0].strip()}

### About
This app uses open-source Stable Diffusion models.
All processing happens on the device shown above.

**Note:** First run downloads the model (~2-4GB).
""")

# Additional info
with st.expander("📖 Complete Usage Guide"):
    st.markdown("""
    ### How to Use
    
    1. **Enter your prompt** - describe what you want to see
    2. **Adjust parameters** (optional):
       - Steps: 15-20 for CPU, 25-35 for GPU
       - Guidance: 7-9 is typical
       - Seed: for reproducible results
    3. **Click Generate** and wait
    4. **Download** your image!
    
    ### Performance Guide
    
    **On Streamlit Cloud (CPU):**
    - ⏱️ Very slow: 5-10 minutes per image
    - 📏 Use 512x512 resolution
    - 🔢 Use 15-20 steps maximum
    
    **On Local Machine with GPU:**
    - ⚡ Fast: 10-30 seconds per image
    - 📏 Can use 512x512 or 768x768
    - 🔢 Use 25-50 steps for quality
    
    ### Running Locally
    
    To run this on your machine with GPU acceleration:
    
    ```bash
    # Install dependencies
    pip install -r requirements.txt
    
    # Run the app
    streamlit run app.py
    ```
    
    Your NVIDIA Quadro P1000 will make generation **much faster**!
    """)

# Download instructions
with st.expander("💻 Download & Run Locally (Recommended)"):
    st.markdown("""
    ### Why Run Locally?
    - ⚡ **100x faster** with your GPU
    - 🎨 Better quality settings
    - 🔒 Complete privacy
    - 💾 No download limits
    
    ### Quick Start
    
    1. **Save these files:**
       - `app.py` (this code)
       - `requirements.txt`
    
    2. **Install & Run:**
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    
    3. **Enjoy fast GPU generation!**
    
    Your Quadro P1000 will generate images in ~15-30 seconds instead of 5-10 minutes.
    """)
