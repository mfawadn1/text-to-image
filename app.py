import streamlit as st
import torch
from diffusers import DiffusionPipeline
import gc
import random
from io import BytesIO
import time

# Page configuration
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Critical: Streamlit Cloud has very limited resources
# This version uses the smallest possible model and includes retry logic

st.title("🎨 AI Text-to-Image Generator")

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    st.warning("""
    ⚠️ **Running on CPU - Very Limited Performance**
    
    Streamlit Cloud has limited resources:
    - Limited RAM (~8GB)
    - No GPU
    - Generation takes 5-15 minutes per image
    
    **For practical use, please run locally with your GPU.**
    """)

# Use the absolutely smallest model that works on Streamlit Cloud
MODEL_ID = "CompVis/stable-diffusion-v1-4"  # Smallest SD model

@st.cache_resource(show_spinner=False)
def load_model_safe():
    """Load model with maximum safety and minimal memory usage"""
    try:
        with st.spinner("📥 Downloading model (one-time, ~4GB)... This may take 5-10 minutes."):
            # Use the most memory-efficient loading possible
            pipe = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,  # Must use float32 on CPU
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            
            st.write("✓ Model downloaded")
            
            # Move to device
            pipe = pipe.to(device)
            st.write("✓ Model loaded to CPU")
            
            # Maximum memory optimization for CPU
            pipe.enable_attention_slicing(slice_size=1)
            pipe.enable_vae_slicing()
            
            st.write("✓ Memory optimizations applied")
            
            return pipe
            
    except Exception as e:
        st.error(f"""
        ❌ **Model Loading Failed**
        
        Error: {str(e)}
        
        This usually means:
        1. Streamlit Cloud ran out of memory
        2. Network timeout during download
        3. Resource limits exceeded
        
        **Solution:** This app needs to run locally with more resources.
        """)
        return None

# Sidebar
st.sidebar.header("⚙️ Settings")
st.sidebar.info(f"**Device:** {device.upper()}")
st.sidebar.warning("**Model:** SD v1.4 (smallest)")

# Simplified parameters for Streamlit Cloud
num_steps = st.sidebar.slider("Steps", 10, 25, 15, 5, help="Keep low (10-15) on Streamlit Cloud")
guidance = st.sidebar.slider("Guidance", 5.0, 12.0, 7.5, 0.5)

use_random_seed = st.sidebar.checkbox("Random Seed", value=True)
if use_random_seed:
    seed = random.randint(0, 2**32 - 1)
else:
    seed = st.sidebar.number_input("Seed", 0, 2**32-1, 42)

# Force small size on Streamlit Cloud
width, height = 512, 512

# Estimated time
est_time = num_steps * 20  # ~20 sec per step on Streamlit Cloud CPU
st.sidebar.warning(f"⏱️ Est. time: ~{est_time//60}min {est_time%60}sec")

# Main input
prompt = st.text_area(
    "✍️ Enter your prompt",
    height=100,
    placeholder="Example: a beautiful landscape, oil painting",
    help="Keep it simple for faster results"
)

negative = st.text_area(
    "🚫 Negative prompt (optional)",
    height=60,
    value="blurry, bad quality",
)

# Generate button
if st.button("🎨 Generate Image", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a prompt first!")
    else:
        # Try to load model
        pipe = load_model_safe()
        
        if pipe is None:
            st.error("❌ Cannot proceed without model. Please try running locally.")
        else:
            try:
                # Clear memory before generation
                gc.collect()
                
                st.warning(f"⏳ Generating... This will take ~{est_time//60} minutes on Streamlit Cloud CPU. Please wait patiently!")
                
                # Generate with progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                generator = torch.Generator(device=device).manual_seed(seed)
                
                start_time = time.time()
                
                # Generate with callback for progress
                def progress_callback(step, timestep, latents):
                    progress = (step + 1) / num_steps
                    progress_bar.progress(progress)
                    elapsed = time.time() - start_time
                    status_text.text(f"Step {step+1}/{num_steps} - {elapsed:.0f}s elapsed")
                
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    generator=generator,
                    callback=progress_callback,
                    callback_steps=1
                ).images[0]
                
                elapsed = time.time() - start_time
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Success!
                st.success(f"✅ Generated in {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)!")
                
                # Display image
                st.image(image, caption=f"Seed: {seed}", use_container_width=True)
                
                # Download button
                buf = BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    "💾 Download Image",
                    data=buf.getvalue(),
                    file_name=f"image_{seed}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                # Details
                with st.expander("ℹ️ Details"):
                    st.write(f"**Prompt:** {prompt}")
                    st.write(f"**Steps:** {num_steps}")
                    st.write(f"**Guidance:** {guidance}")
                    st.write(f"**Seed:** {seed}")
                    st.write(f"**Time:** {elapsed:.0f}s")
                
                # Clean up
                gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    st.error("""
                    ❌ **Out of Memory**
                    
                    Streamlit Cloud doesn't have enough RAM.
                    
                    **Solutions:**
                    1. Reduce steps to 10
                    2. Try again (memory might free up)
                    3. **Run locally** (recommended)
                    """)
                else:
                    st.error(f"❌ Error: {str(e)}")
                gc.collect()
                
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.info("This app works best when run locally with adequate resources.")

# Instructions
st.markdown("---")

st.markdown("""
### ⚠️ Important Information

**Streamlit Cloud Limitations:**
- Very limited RAM (~8GB total)
- No GPU acceleration
- Generation takes 5-15 minutes per image
- May crash due to memory limits
- Model download takes 5-10 minutes on first run

**Recommended: Run Locally**

This app is designed to run on your **local machine with GPU**. To use it properly:

```bash
# Download the files from GitHub
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**On your NVIDIA Quadro P1000:**
- ⚡ Generation time: 15-30 seconds (vs 5-15 minutes here)
- 💪 Stable, won't crash
- 🎨 Can use better models
- 📏 Can generate larger images

### Current Setup
This online demo uses the smallest model (SD v1.4) with maximum memory optimization, but it's still constrained by Streamlit Cloud's limits.
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 Tips
- Use 10-15 steps on Streamlit Cloud
- Keep prompts simple
- Be very patient
- **Best experience = run locally**

### Download Code
Get the full code from your GitHub repo to run locally with GPU acceleration.
""")
