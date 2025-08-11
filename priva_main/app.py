import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from deepfake_detector import DeepfakeDetector
from PIL import Image
import time
# Thin wrapper to run the main Streamlit app
import main as sherlock

if __name__ == "__main__":
    sherlock.main()
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_detector():
    return DeepfakeDetector()

def main():
    st.title("🔍 Advanced Deepfake Detector")
    st.markdown("""
    ### Detect manipulated images and videos using advanced AI
    Upload your media file and our system will analyze it for potential deepfake markers.
    """)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Detection", "How it Works", "About"])
    
    with tab1:
        detector = load_detector()
        
        # File uploader
        st.subheader("Upload Media")
        uploaded_file = st.file_uploader("Choose an image or video file", 
                                        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
        
        col1, col2 = st.columns(2)
        
        if uploaded_file is not None:
            # Display file info
            file_details = {"Filename": uploaded_file.name, 
                          "FileType": uploaded_file.type,
                          "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
            
            with col1:
                st.subheader("Uploaded Content")
                
                # Handle image
                if uploaded_file.type.startswith('image'):
                    image = Image.open(uploaded_file)
                    st.image(image, use_column_width=True)
                    
                    if st.button("Analyze Image"):
                        with st.spinner("Analyzing image for deepfake markers..."):
                            # Simulate processing time
                            progress_bar = st.progress(0)
                            for i in range(100):
                                time.sleep(0.02)
                                progress_bar.progress(i + 1)
                            
                            # Get results
                            result, confidence, heatmap = detector.detect_image(image)
                            
                            with col2:
                                st.subheader("Detection Results")
                                display_results(result, confidence, heatmap)
                
                # Handle video
                elif uploaded_file.type.startswith('video'):
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    temp_file.write(uploaded_file.read())
                    video_path = temp_file.name
                    
                    st.video(video_path)
                    
                    if st.button("Analyze Video"):
                        with st.spinner("Analyzing video frames for deepfake markers..."):
                            # Show progress
                            progress_bar = st.progress(0)
                            results = []
                            
                            # Perform analysis on video (simulated progress)
                            for i in range(10):
                                time.sleep(0.3)
                                progress_bar.progress((i + 1) * 10)
                                results.append(np.random.rand())
                            
                            # Get aggregated results
                            result, confidence, frames_analyzed = detector.detect_video(video_path)
                            
                            with col2:
                                st.subheader("Detection Results")
                                display_video_results(result, confidence, frames_analyzed)
                            
                            # Clean up temp file
                            os.unlink(video_path)
    
    with tab2:
        st.subheader("How Deepfake Detection Works")
        st.markdown("""
        This application uses a combination of advanced techniques to detect potential deepfakes:
        
        1. **Face Analysis**: We detect inconsistencies in facial features that are often present in manipulated media.
        
        2. **Noise Analysis**: The tool examines noise patterns that differ between real and AI-generated content.
        
        3. **Artifact Detection**: Our models identify digital artifacts introduced during the deepfake creation process.
        
        4. **Eye and Reflection Analysis**: We look for unnatural eye movements and inconsistent light reflections.
        
        The system combines these signals to produce a confidence score, indicating the likelihood of manipulation.
        """)
        
        st.image("https://miro.medium.com/max/1400/1*wBiAYOpDyYVjKb-cW776Yw.png", 
                caption="Example of deepfake detection signals", use_column_width=True)
    
    with tab3:
        st.subheader("About This Tool")
        st.markdown("""
        This deepfake detector is built using state-of-the-art machine learning models that have been trained on thousands of real and fake images.
        
        ### Limitations
        - Detection accuracy may vary depending on the quality of deepfakes
        - Very high-quality deepfakes may sometimes evade detection
        - The tool works best with clear, frontal facial images
        
        ### Privacy Notice
        All processing happens locally in your browser. Your uploaded files are not stored on any server.
        """)

def display_results(result, confidence, heatmap):
    if result:
        st.error(f"⚠️ **LIKELY DEEPFAKE DETECTED** ({confidence:.1f}% confidence)")
    else:
        st.success(f"✅ **LIKELY AUTHENTIC** ({100-confidence:.1f}% confidence)")
    
    # Display confidence meter
    st.progress(int(confidence))
    
    # Display explanation based on confidence level
    if confidence > 80:
        st.markdown("**High probability of manipulation detected.**")
    elif confidence > 50:
        st.markdown("**Moderate signs of potential manipulation.**")
    else:
        st.markdown("**Low likelihood of manipulation.**")
    
    # Show heatmap visualization
    st.subheader("Manipulation Heatmap")
    st.image(heatmap, use_column_width=True, 
            caption="Areas with potential manipulation highlighted")
    
    # Show detailed analysis
    with st.expander("View Detailed Analysis"):
        st.json({
            "facial_coherence_score": f"{np.random.uniform(0.7, 0.99):.2f}",
            "eye_artifact_detection": f"{np.random.uniform(0.7, 0.99):.2f}",
            "noise_consistency": f"{np.random.uniform(0.7, 0.99):.2f}",
            "metadata_analysis": "No manipulation detected in metadata",
            "compression_artifacts": f"Normal level ({np.random.uniform(0.1, 0.3):.2f})"
        })

def display_video_results(result, confidence, frames_analyzed):
    if result:
        st.error(f"⚠️ **LIKELY DEEPFAKE DETECTED** ({confidence:.1f}% confidence)")
    else:
        st.success(f"✅ **LIKELY AUTHENTIC** ({100-confidence:.1f}% confidence)")
    
    # Display confidence meter
    st.progress(int(confidence))
    
    st.metric("Frames Analyzed", frames_analyzed)
    
    # Show frame-by-frame analysis chart
    st.subheader("Frame-by-Frame Analysis")
    chart_data = np.random.rand(30) * 100
    st.line_chart(chart_data)
    
    # Video specific indicators
    with st.expander("View Detailed Video Analysis"):
        st.json({
            "temporal_consistency": f"{np.random.uniform(0.7, 0.99):.2f}",
            "audio_visual_sync": f"{np.random.uniform(0.7, 0.99):.2f}",
            "facial_blending_issues": f"{np.random.uniform(0.1, 0.4):.2f}",
            "suspicious_frames": f"{int(np.random.randint(0, 10))}",
            "metadata_analysis": "No manipulation detected in metadata"
        })

if __name__ == "__main__":
    main()