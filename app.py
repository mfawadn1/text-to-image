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
st.markdown("""
Generate beautiful images from text descriptions using Stable Diffusion.
This app runs completely locally on your machine using open-source models.
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
        st.sidebar.warning("⚠️ No GPU detected. Using CPU (will be slower)")
    return device

device = get_device()

# Model selection with caching
@st.cache_resource
def load_model(model_name):
    """Load the Stable Diffusion model with caching"""
    try:
        with st.spinner(f"🔄 Loading model: {model_name}... This may take a few minutes on first run."):
            # Load pipeline with optimizations for limited VRAM
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,  # Disable for speed
                requires_safety_checker=False
            )
            
            pipe = pipe.to(device)
            
            # Enable memory efficient attention if on GPU
            if device == "cuda":
                pipe.enable_attention_slicing()
            
            st.sidebar.success("✅ Model loaded successfully!")
            return pipe
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Sidebar - Model selection
st.sidebar.header("⚙️ Model Settings")

model_options = {
    "Stable Diffusion v1.5 (Recommended for 4GB VRAM)": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion 2.1 Base (Better quality)": "stabilityai/stable-diffusion-2-1-base",
    "Stable Diffusion v1.4 (Fastest)": "CompVis/stable-diffusion-v1-4"
}

selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    options=list(model_options.keys()),
    index=0
)

model_id = model_options[selected_model_name]

# Load the model
pipe = load_model(model_id)

# Sidebar - Generation parameters
st.sidebar.header("🎛️ Generation Parameters")

num_inference_steps = st.sidebar.slider(
    "Inference Steps",
    min_value=10,
    max_value=100,
    value=30,
    step=5,
    help="More steps = better quality but slower. 25-35 is a good balance."
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

# Image dimensions
width = st.sidebar.selectbox("Width", [512, 768], index=0)
height = st.sidebar.selectbox("Height", [512, 768], index=0)

if width > 512 or height > 512:
    st.sidebar.warning("⚠️ Larger images require more VRAM and may be slower")

# Main interface - Prompt inputs
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area(
        "✍️ Describe the image you want to generate",
        height=100,
        placeholder="Example: a beautiful sunset over mountains, digital art, highly detailed, trending on artstation",
        help="Be descriptive! Include style, mood, and details."
    )

    negative_prompt = st.text_area(
        "🚫 Negative Prompt (what to avoid)",
        height=80,
        value="blurry, bad quality, distorted, ugly, bad anatomy",
        help="Describe what you don't want in the image"
    )

with col2:
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Be specific and descriptive
    - Add style keywords (e.g., "oil painting", "photorealistic", "anime style")
    - Use quality boosters: "highly detailed", "8k", "masterpiece"
    - Adjust steps: 20-30 for drafts, 40-50 for final images
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
            # Clear GPU cache before generation
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Set seed for reproducibility
            generator = torch.Generator(device=device).manual_seed(seed)
            
            # Generate image with progress
            with st.spinner(f"🎨 Generating image... (Seed: {seed})"):
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                ).images[0]
            
            # Display the generated image
            st.success("✅ Image generated successfully!")
            
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
            
            # Clear cache after generation
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                st.error("""
                ❌ **Out of Memory Error!**
                
                Your GPU ran out of memory. Try these solutions:
                1. Reduce image size to 512x512
                2. Reduce inference steps to 20-25
                3. Use a smaller/faster model (Stable Diffusion v1.4)
                4. Close other applications using GPU
                5. Restart the app
                """)
                # Clear CUDA cache
                if device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                st.error(f"❌ Error during generation: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This app uses open-source Stable Diffusion models from Hugging Face.
All processing happens locally on your machine.

**Note:** First run will download the model (~4-7GB) which may take several minutes.
""")

# Additional info in main area
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Select a model** in the sidebar (Stable Diffusion v1.5 recommended for 4GB VRAM)
    2. **Enter your prompt** - be descriptive and specific
    3. **Adjust parameters** in the sidebar:
       - More steps = better quality but slower
       - Guidance scale controls prompt adherence
       - Seed controls randomness
    4. **Click Generate** and wait for your image
    5. **Download** your creation!
    
    **Performance Tips:**
    - First generation is slower (model loading)
    - GPU is ~10-20x faster than CPU
    - 512x512 images work best on 4GB VRAM
    - Use 25-30 steps for good quality/speed balance
    """)
