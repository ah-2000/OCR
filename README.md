# OCR
          
# 📋 Complete Code Description: Advanced OCR System

This is a comprehensive **Streamlit-based OCR (Optical Character Recognition) application** that intelligently processes various document types using AI-powered text extraction and computer vision techniques.

## 🏗️ **Architecture Overview**

The application consists of a single Python file (<mcfile name="streamlit_app.py" path="d:\flutter projects\llama OCR\streamlit_app.py"></mcfile>) with the following key components:

### 📦 **Dependencies & Setup**
- **Streamlit**: Web interface framework
- **OpenCV (cv2)**: Computer vision for face detection
- **Together AI**: LLaMA Vision model integration
- **PIL/Pillow**: Image processing
- **Base64**: Image encoding for API calls
- **dotenv**: Environment variable management

### 🧠 **Core Class: ImageProcessor**

The <mcsymbol name="ImageProcessor" filename="streamlit_app.py" path="d:\flutter projects\llama OCR\streamlit_app.py" startline="15" type="class"></mcsymbol> class handles all OCR and image processing functionality:

#### **🎯 Document Type Detection**
- **Automatic Classification**: Uses LLaMA Vision AI to identify document types
- **Supported Types**: ID cards, passports, driving licenses, certificates, general text
- **Smart Prompting**: Each document type has specialized extraction prompts

#### **📝 Document-Specific Prompts**
The system includes tailored HTML-formatted prompts for:
- **🆔 ID Cards**: Name, father name, identity number, gender, dates
- **📘 Passports**: Full name, passport number, nationality, birth details
- **🚗 Driving Licenses**: License number, vehicle classes, address
- **📜 Certificates**: Title, recipient, institution, issue date
- **📄 General Text**: Organized text extraction with proper formatting

#### **👤 Face Detection & Extraction**
- **OpenCV Integration**: Uses Haar Cascade classifiers
- **Automatic Cropping**: Detects and extracts face photos from documents
- **Base64 Encoding**: Converts images for API transmission

#### **🔄 Main Processing Pipeline**
1. **Document Type Detection**: AI analyzes image to determine document category
2. **Text Extraction**: Uses document-specific prompts with LLaMA Vision
3. **Face Processing**: Detects and crops face photos if present
4. **Structured Output**: Returns formatted HTML with extracted information

## 🎨 **User Interface Features**

### **📱 Layout & Design**
- **Wide Layout**: Two-column design for upload and results
- **Custom CSS**: Professional styling with gradients and shadows
- **Responsive Design**: Adapts to different screen sizes
- **Modern UI**: Clean, intuitive interface with emojis and icons

### **🔧 Interactive Components**
- **File Uploader**: Supports PNG, JPG, JPEG formats
- **Image Preview**: Shows uploaded document before processing
- **Processing Spinner**: Visual feedback during AI analysis
- **Results Display**: Formatted extraction results with HTML rendering

### **📊 Information Sidebar**
- **Document Types**: Lists all supported document categories
- **How It Works**: Step-by-step process explanation
- **User Guidance**: Clear instructions for optimal usage

## ⚡ **Key Functionality**

### **🤖 AI-Powered Processing**
- **LLaMA Vision Model**: Uses `meta-llama/Llama-Vision-Free` for OCR
- **Streaming Responses**: Real-time processing with Together AI
- **Error Handling**: Graceful fallbacks and error messages

### **🎯 Smart Features**
- **Automatic Document Recognition**: No manual type selection needed
- **Structured Data Extraction**: Organized output with bold headings
- **Face Photo Extraction**: Automatically crops and displays face images
- **HTML Formatting**: Rich text output with proper styling

### **🔒 Security & Privacy**
- **Environment Variables**: Secure API key management
- **Temporary Files**: Automatic cleanup of uploaded images
- **Local Processing**: Face detection happens locally

## 🚀 **Usage Workflow**

1. **Upload**: User selects and uploads a document image
2. **Preview**: System displays the uploaded document
3. **Process**: Click "Extract Information" to start AI analysis
4. **Detect**: AI automatically identifies document type
5. **Extract**: System extracts text using specialized prompts
6. **Display**: Results shown with proper formatting and extracted photos

## 💡 **Technical Highlights**

- **Modular Design**: Clean separation of concerns with dedicated methods
- **Error Resilience**: Comprehensive exception handling throughout
- **Performance Optimized**: Efficient image processing and API calls
- **Extensible Architecture**: Easy to add new document types
- **Professional UI**: Modern Streamlit interface with custom styling

        
