import os
import re
import cv2
import base64
import together
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import tempfile

load_dotenv()

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class ImageProcessor:
    def __init__(self):
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        os.environ["TOGETHER_API_KEY"] = api_key
        self.client = together.Together()
        self.model = "meta-llama/Llama-Vision-Free"
        self.document_prompts = {
            "id_card": {
                "prompt": """Extract information from this ID card and format as HTML:
                <div class='document-header'><h2>🆔 ID Card Information</h2></div>
                <div class='info-section'>
                <p><strong>Name:</strong> [Extracted Name]</p>
                <p><strong>Father Name:</strong> [Extracted Father Name]</p>
                <p><strong>Identity Number:</strong> [Extracted Identity Number]</p>
                <p><strong>Gender:</strong> [M/F]</p>
                <p><strong>Date of Birth:</strong> [DD.MM.YYYY]</p>
                <p><strong>Date of Issue:</strong> [DD.MM.YYYY]</p>
                <p><strong>Date of Expiry:</strong> [DD.MM.YYYY or Lifetime]</p>
                </div>""",
                "icon": "🆔"
            },
            "passport": {
                "prompt": """Extract information from this passport and format as HTML:
                <div class='document-header'><h2>📘 Passport Information</h2></div>
                <div class='info-section'>
                <p><strong>Full Name:</strong> [Extracted Name]</p>
                <p><strong>Passport Number:</strong> [Passport Number]</p>
                <p><strong>Nationality:</strong> [Country]</p>
                <p><strong>Date of Birth:</strong> [DD.MM.YYYY]</p>
                <p><strong>Place of Birth:</strong> [Place]</p>
                <p><strong>Date of Issue:</strong> [DD.MM.YYYY]</p>
                <p><strong>Date of Expiry:</strong> [DD.MM.YYYY]</p>
                <p><strong>Issuing Authority:</strong> [Authority]</p>
                </div>""",
                "icon": "📘"
            },
            "driving_license": {
                "prompt": """Extract information from this driving license and format as HTML:
                <div class='document-header'><h2>🚗 Driving License</h2></div>
                <div class='info-section'>
                <p><strong>Name:</strong> [Extracted Name]</p>
                <p><strong>License Number:</strong> [License Number]</p>
                <p><strong>Date of Birth:</strong> [DD.MM.YYYY]</p>
                <p><strong>Address:</strong> [Address]</p>
                <p><strong>Vehicle Classes:</strong> [Classes]</p>
                <p><strong>Issue Date:</strong> [DD.MM.YYYY]</p>
                <p><strong>Expiry Date:</strong> [DD.MM.YYYY]</p>
                </div>""",
                "icon": "🚗"
            },
            "certificate": {
                "prompt": """Extract information from this certificate/document and format as HTML:
                <div class='document-header'><h2>📜 Certificate/Document</h2></div>
                <div class='info-section'>
                <p><strong>Document Title:</strong> [Title]</p>
                <p><strong>Recipient Name:</strong> [Name]</p>
                <p><strong>Institution/Authority:</strong> [Institution]</p>
                <p><strong>Date Issued:</strong> [Date]</p>
                <p><strong>Additional Details:</strong> [Other relevant information]</p>
                </div>""",
                "icon": "📜"
            },
            "general_text": {
                "prompt": """Extract and organize all text from this document in a well-formatted HTML structure:
                <div class='document-header'><h2>📄 Document Text</h2></div>
                <div class='text-content'>
                [Organize extracted text with appropriate headings and formatting]
                </div>""",
                "icon": "📄"
            }
        }

    def detect_document_type(self, image_path):
        """Detect document type using a quick analysis"""
        try:
            base64_image = self.encode_image(image_path)
            
            detection_prompt = """Analyze this image and identify the document type. 
            Respond with ONLY one of these options:
            - id_card
            - passport  
            - driving_license
            - certificate
            - general_text
            
            Base your decision on visual elements, layout, and text patterns."""
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": detection_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                stream=True,
            )
            
            response = "".join(
                chunk.choices[0].delta.content for chunk in stream
                if hasattr(chunk, 'choices') and chunk.choices and 
                hasattr(chunk.choices[0], 'delta') and 
                hasattr(chunk.choices[0].delta, 'content')
            ).strip().lower()
            
            # Return detected type or default to general_text
            return response if response in self.document_prompts else "general_text"
            
        except Exception:
            return "general_text"

    def encode_image(self, image_path):
        """ Convert image to Base64 """
        if not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def detect_and_crop_face(self, image_path):
        """ Detects and crops the face using OpenCV Haar Cascade """
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            cropped_face = image[y:y+h, x:x+w]
            
            # Create temporary file for cropped face
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                cv2.imwrite(tmp_file.name, cropped_face)
                return self.encode_image(tmp_file.name)
        return None

    def analyze_image(self, image_path):
        """Enhanced analysis with document type detection and formatting"""
        try:
            # Detect document type first
            doc_type = self.detect_document_type(image_path)
            
            base64_image = self.encode_image(image_path)
            face_base64 = self.detect_and_crop_face(image_path)
            
            # Use appropriate prompt based on document type
            prompt = self.document_prompts[doc_type]["prompt"]
            
            # Process OCR with document-specific prompt
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                stream=True,
            )
            
            response_text = "".join(
                chunk.choices[0].delta.content for chunk in stream
                if hasattr(chunk, 'choices') and chunk.choices and 
                hasattr(chunk.choices[0], 'delta') and 
                hasattr(chunk.choices[0].delta, 'content')
            )
            
            return {
                "text": response_text,
                "photo": face_base64 if face_base64 else None,
                "document_type": doc_type,
                "icon": self.document_prompts[doc_type]["icon"]
            }
            
        except Exception as e:
            return {"error": str(e)}

# Streamlit App Configuration
st.set_page_config(
    page_title="OCR System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2c3e50;
    font-size: 3rem;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.document-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}

.info-section {
    background: rgba(255,255,255,0.1);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

.document-header h2 {
    color: #2c3e50;
    border-bottom: 2px solid #4CAF50;
    padding-bottom: 10px;
    margin-bottom: 15px;
}

.success-message {
    background: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #c3e6cb;
    margin: 1rem 0;
}

.error-message {
    background: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #f5c6cb;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 OCR System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Document Types Supported")
        st.markdown("""
        - 🆔 **ID Cards**: National ID, Identity Cards
        - 📘 **Passports**: International Passports
        - 🚗 **Driving Licenses**: Driver's Licenses
        - 📜 **Certificates**: Academic, Professional Certificates
        - 📄 **General Text**: Any document with text
        """)
        
        st.header("ℹ️ How it works")
        st.markdown("""
        1. Upload your document image
        2. AI automatically detects document type
        3. Extracts information with proper formatting
        4. Displays results with bold headings
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload ID cards, passports, certificates, or any document with text"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Document", use_container_width=True)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                temp_image_path = tmp_file.name
    
    with col2:
        st.header("📊 Extraction Results")
        
        if uploaded_file is not None:
            if st.button("🚀 Extract Information", type="primary"):
                with st.spinner("🔄 Processing document..."):
                    try:
                        # Initialize processor
                        processor = ImageProcessor()
                        
                        # Analyze image
                        result = processor.analyze_image(temp_image_path)
                        
                        if "error" in result:
                            st.markdown(f'<div class="error-message">❌ Error: {result["error"]}</div>', unsafe_allow_html=True)
                        else:
                            # Display document type
                            st.success(f"Document Type Detected: {result['icon']} {result['document_type'].replace('_', ' ').title()}")
                            
                            # Display extracted text with HTML formatting
                            st.markdown("### 📄 Extracted Information")
                            st.markdown(result["text"], unsafe_allow_html=True)
                            
                            # Display extracted photo if available
                            if result["photo"]:
                                st.markdown("### 👤 Extracted Photo")
                                # Decode base64 image
                                import io
                                photo_data = base64.b64decode(result["photo"])
                                photo_image = Image.open(io.BytesIO(photo_data))
                                st.image(photo_image, caption="Extracted Face", width=200)
                            
                            # Success message
                            st.markdown('<div class="success-message">✅ Document processed successfully!</div>', unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.markdown(f'<div class="error-message">❌ Error processing document: {str(e)}</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Please upload a document image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🤖 Powered by LLaMA Vision AI | 🔒 Your documents are processed securely</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()