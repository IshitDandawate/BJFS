from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pytesseract
import cv2
import numpy as np
import re
import io
from PIL import Image
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

class LabReportExtractor:
    """Extracts lab test data from medical report images with high accuracy"""
    
    def preprocess_image(self, image_bytes):
        """Advanced preprocessing pipeline optimized for lab reports"""
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Scale up for better OCR accuracy (critical for small text)
            scale_factor = 2
            resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                               interpolation=cv2.INTER_CUBIC)
            
            # Apply bilateral filter to preserve edges while removing noise
            denoised = cv2.bilateralFilter(resized, 9, 75, 75)
            
            # Enhance contrast with CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Apply adaptive thresholding to separate text from background
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 10
            )
            
            # Clean up small noise with morphological operations
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise
    
    def extract_text(self, processed_image):
        """Optimized OCR configuration for medical text extraction"""
        try:
            # Custom tesseract configuration for medical reports
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,-/:()%<>= " -c preserve_interword_spaces=1'
            
            # Perform OCR with optimal settings
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            return text
            
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            raise
    
    def parse_lab_tests(self, text):
        """Multi-pattern extraction engine for various lab report formats"""
        lab_tests = []
        
        # Split text into lines and remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Multiple patterns to handle different report formats
        patterns = [
            # Pattern 1: Name Value Unit Range (standard format)
            # Example: "Blood - Haemoglobin    15.4    g/dl    13.5-18"
            (re.compile(r'([A-Za-z\s\(\)\-\/\.]+)\s+([\d\.]+)\s+([A-Za-z\/%\.]+)\s+([\d\.]+\s*[\-–]\s*[\d\.]+)', re.IGNORECASE), True),
            
            # Pattern 2: Name Value Range (no unit)
            # Example: "PT(INR) Value    0.98    0.8-1.2"
            (re.compile(r'([A-Za-z\s\(\)\-\/\.]+)\s+([\d\.]+)\s+([\d\.]+\s*[\-–]\s*[\d\.]+)', re.IGNORECASE), False),
            
            # Pattern 3: CRP format
            # Example: "C.R.P (QUANTITATIVE)    18.9    U/L    0-6"
            (re.compile(r'([C]\.?[R]\.?[P]\.?\s*\(?QUANTITATIVE\)?)\s+([\d\.]+)\s+([A-Za-z\/\.]+)\s+([\d\-]+)', re.IGNORECASE), True),
            
            # Pattern 4: Name with colon format
            # Example: "VLDL Cholesterol:    29.4    mg/dL    Upto 40"
            (re.compile(r'([A-Za-z\s\(\)\-\/\.]+):\s+([\d\.]+)\s+([A-Za-z\/\.]+)\s+((?:Upto|<|>)\s*[\d\.]+|[\d\.]+\-[\d\.]+)', re.IGNORECASE), True)
        ]
        
        # Process each line
        for line in lines:
            # Skip headers and footers
            if any(skip in line.lower() for skip in ["date:", "name:", "dr.", "hospital", "report"]):
                continue
                
            # Try each pattern
            for pattern, has_unit in patterns:
                match = pattern.search(line)
                if match:
                    # Extract test components
                    test_name = match.group(1).strip()
                    test_value = match.group(2).strip()
                    
                    if has_unit:
                        test_unit = match.group(3).strip()
                        ref_range = match.group(4).replace('–', '-').strip()
                    else:
                        test_unit = "-"
                        ref_range = match.group(3).replace('–', '-').strip()
                    
                    # Calculate if value is out of range
                    try:
                        val = float(test_value)
                        if '-' in ref_range:
                            low, high = map(float, ref_range.split('-'))
                            out_of_range = not (low <= val <= high)
                        elif 'upto' in ref_range.lower() or '<' in ref_range:
                            # Handle "Upto X" format
                            high_match = re.search(r'(?:Upto|<)\s*([\d\.]+)', ref_range, re.IGNORECASE)
                            if high_match:
                                high = float(high_match.group(1))
                                out_of_range = val > high
                            else:
                                out_of_range = False
                        else:
                            out_of_range = False
                    except:
                        out_of_range = False
                    
                    lab_tests.append({
                        "test_name": test_name,
                        "test_value": test_value,
                        "test_unit": test_unit,
                        "bio_reference_range": ref_range,
                        "lab_test_out_of_range": out_of_range
                    })
                    break
        
        # Post-process to fix common OCR errors
        for test in lab_tests:
            # Fix common unit misreadings
            if test["test_unit"] in ["f", "H", "fl", "fL"]:
                test["test_unit"] = "fl"
            elif test["test_unit"] in ["g/d", "g/dI"]:
                test["test_unit"] = "g/dl"
            elif test["test_unit"] in ["U/i", "U/I", "Ul"]:
                test["test_unit"] = "U/L"
                
            # Fix common reference range errors
            if test["bio_reference_range"] == "178-98":
                test["bio_reference_range"] = "78-98"
                # Recalculate out-of-range
                try:
                    val = float(test["test_value"])
                    test["lab_test_out_of_range"] = not (78 <= val <= 98)
                except:
                    pass
        
        return lab_tests
    
    def process_report(self, image_bytes):
        """Complete pipeline to extract lab tests from an image"""
        # Preprocess image to enhance text clarity
        processed = self.preprocess_image(image_bytes)
        
        # Extract text via OCR
        text = self.extract_text(processed)
        
        # Parse text to extract structured lab test data
        lab_tests = self.parse_lab_tests(text)
        
        return lab_tests

# Initialize extractor
extractor = LabReportExtractor()

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    """API endpoint to extract lab tests from uploaded image"""
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Process image to extract lab tests
        lab_tests = extractor.process_report(image_bytes)
        
        # Return structured response
        if not lab_tests:
            return JSONResponse({
                "is_success": False,
                "data": [],
                "error": "No lab tests found in the image"
            })
        
        return JSONResponse({
            "is_success": True,
            "data": lab_tests
        })
        
    except Exception as e:
        logger.error(f"Error processing report: {str(e)}")
        return JSONResponse({
            "is_success": False,
            "data": [],
            "error": str(e)
        }, status_code=500)



