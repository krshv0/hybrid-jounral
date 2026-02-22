"""
Handwriting Page Processing System
Hybrid Handwritten + Digital Journal - OCR Module with User Corrections
"""

import uuid
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum

import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
import logging


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy scalar and array types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. ENUMS & CONSTANTS
# ============================================================================

class ProcessingStage(str, Enum):
    """Page processing pipeline stages"""
    OCR_EXTRACTION = "ocr_extraction"
    REVIEW_PENDING = "review_pending"
    USER_CORRECTING = "user_correcting"
    READY_TO_FINALIZE = "ready_to_finalize"
    FINALIZED = "finalized"


class CorrectionStatus(str, Enum):
    """Status of individual line correction"""
    PENDING = "pending"
    CORRECTED = "corrected"
    ACCEPTED = "accepted"


class CorrectionSource(str, Enum):
    """Source of correction"""
    USER = "user"
    USER_SKIP = "user_skip"
    MODEL = "model"


# Constants
CONFIDENCE_THRESHOLD = 0.50  # Flag lines below 50% confidence
FALLBACK_THRESHOLD = 0.50    # Use Tesseract if below 50%
LINE_GROUPING_TOLERANCE = 15  # Pixels tolerance for grouping chars into lines


# ============================================================================
# 2. DATA MODELS
# ============================================================================

@dataclass
class OCRResult:
    """Unified OCR output schema for single line"""
    line_number: int
    original_text: str
    corrected_text: Optional[str] = None
    confidence: float = 0.0
    source_model: str = "easyocr"  # 'easyocr' or 'tesseract'
    character_confidences: List[float] = field(default_factory=list)
    bboxes: List[Tuple] = field(default_factory=list)
    flagged: bool = False
    correction_status: str = CorrectionStatus.PENDING.value
    correction_timestamp: Optional[str] = None
    correction_source: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict) -> 'OCRResult':
        """Create from dictionary"""
        return OCRResult(**data)
    
    def get_display_text(self) -> str:
        """Get text to display (corrected if available, else original)"""
        return self.corrected_text or self.original_text


@dataclass
class ConfidenceSummary:
    """Page-level confidence metrics"""
    total_lines: int = 0
    flagged_lines: int = 0
    avg_confidence: float = 0.0
    corrected_lines: int = 0
    flagged_percentage: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PageProcessingState:
    """Tracks processing state of entire page"""
    page_id: str
    image_path: str
    original_image: bytes  # Binary image data
    ocr_results: List[OCRResult] = field(default_factory=list)
    processing_stage: str = ProcessingStage.OCR_EXTRACTION.value
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    user_id: Optional[str] = None
    confidence_summary: ConfidenceSummary = field(default_factory=ConfidenceSummary)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'page_id': self.page_id,
            'image_path': self.image_path,
            'ocr_results': [r.to_dict() for r in self.ocr_results],
            'processing_stage': self.processing_stage,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'user_id': self.user_id,
            'confidence_summary': self.confidence_summary.to_dict()
        }
    
    @staticmethod
    def from_dict(data: Dict, original_image: Optional[bytes] = None) -> 'PageProcessingState':
        """Create from dictionary"""
        state = PageProcessingState(
            page_id=data['page_id'],
            image_path=data['image_path'],
            original_image=original_image or b'',
            processing_stage=data.get('processing_stage', ProcessingStage.OCR_EXTRACTION.value),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            user_id=data.get('user_id')
        )
        state.ocr_results = [OCRResult.from_dict(r) for r in data.get('ocr_results', [])]
        
        summary_data = data.get('confidence_summary', {})
        state.confidence_summary = ConfidenceSummary(
            total_lines=summary_data.get('total_lines', 0),
            flagged_lines=summary_data.get('flagged_lines', 0),
            avg_confidence=summary_data.get('avg_confidence', 0.0),
            corrected_lines=summary_data.get('corrected_lines', 0),
            flagged_percentage=summary_data.get('flagged_percentage', 0.0)
        )
        return state


@dataclass
class ReviewDTO:
    """Data transfer object for review interface"""
    page_id: str
    original_image_url: str
    processing_stage: str
    summary: ConfidenceSummary
    lines: List[Dict]


@dataclass
class FinalizedPageOutput:
    """Output after finalization"""
    page_id: str
    final_text: str
    lines: List[OCRResult]
    metadata: Dict
    original_image_path: str
    original_image_bytes: bytes


# ============================================================================
# 3. IMAGE PREPROCESSING
# ============================================================================

class ImagePreprocessor:
    """Handles image preprocessing before OCR"""
    
    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """
        Straighten tilted/rotated pages
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No contours found for deskewing, returning original")
            return image
        
        # Get largest contour (likely the page)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        if angle < -45:
            angle = angle + 90
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        logger.debug(f"Deskewed image by {angle:.1f} degrees")
        return rotated
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance image contrast for better OCR"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        denoised = cv2.morphologyEx(
            image,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        return denoised
    
    @staticmethod
    def preprocess_for_ocr(image_path: str) -> np.ndarray:
        """
        Full preprocessing pipeline
        Returns image ready for OCR
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Could not load image: {image_path}")
        
        logger.info(f"Loaded image: {image_path}, shape: {image.shape}")
        
        # 1. Deskew
        image = ImagePreprocessor.deskew_image(image)
        
        # 2. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 3. Enhance contrast
        enhanced = ImagePreprocessor.enhance_contrast(image)
        
        # 4. Apply thresholding
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Denoise
        denoised = ImagePreprocessor.denoise(thresh)
        
        logger.info("Preprocessing complete")
        return denoised


# ============================================================================
# 4. OCR EXTRACTION SERVICE
# ============================================================================

class OCRExtractionService:
    """Handles OCR text extraction"""
    
    def __init__(self, gpu: bool = False):
        """
        Initialize OCR service
        
        Args:
            gpu: Whether to use GPU acceleration for EasyOCR
        """
        logger.info("Initializing OCR services...")
        self.easy_reader = easyocr.Reader(['en'], gpu=gpu)
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.fallback_threshold = FALLBACK_THRESHOLD
        logger.info("OCR services initialized")
    
    def extract_text_from_page(
        self,
        image_path: str,
        user_id: Optional[str] = None
    ) -> PageProcessingState:
        """
        Main extraction pipeline
        
        Flow:
        1. Preprocess image
        2. Run EasyOCR
        3. Group results by line
        4. Calculate confidence per line
        5. Flag low-confidence lines
        6. Return for review
        
        Args:
            image_path: Path to journal page image
            user_id: Optional user identifier
            
        Returns:
            PageProcessingState with extracted text and confidence scores
        """
        logger.info(f"Starting OCR extraction for: {image_path}")
        
        # Load original image bytes
        with open(image_path, 'rb') as f:
            original_image = f.read()
        
        # Preprocess image
        processed_image = ImagePreprocessor.preprocess_for_ocr(image_path)
        
        # Save temporary processed image for EasyOCR
        temp_path = '/tmp/processed_page.jpg'
        cv2.imwrite(temp_path, processed_image)
        
        # Run EasyOCR
        logger.info("Running EasyOCR...")
        easy_results = self.easy_reader.readtext(temp_path)
        
        # Group by lines
        lines = self._group_by_line(easy_results)
        logger.info(f"Grouped into {len(lines)} lines")
        
        # Create OCR results with unified schema
        ocr_results = []
        for line_num, line_data in enumerate(lines, 1):
            result = self._create_ocr_result(
                line_number=line_num,
                text=line_data['text'],
                confidence=line_data['confidence'],
                character_confidences=line_data['char_confidences'],
                bboxes=line_data['bboxes'],
                source_model='easyocr'
            )
            ocr_results.append(result)
        
        # Flag low confidence
        ocr_results = self._flag_low_confidence(ocr_results)
        flagged_count = sum(1 for r in ocr_results if r.flagged)
        logger.info(f"Flagged {flagged_count} lines for review")
        
        # Create page state
        page_state = PageProcessingState(
            page_id=str(uuid.uuid4()),
            image_path=image_path,
            original_image=original_image,
            ocr_results=ocr_results,
            processing_stage=ProcessingStage.REVIEW_PENDING.value,
            user_id=user_id
        )
        
        # Calculate summary
        page_state.confidence_summary = self._calculate_summary(ocr_results)
        
        logger.info(f"OCR extraction complete. Page ID: {page_state.page_id}")
        return page_state
    
    def _group_by_line(self, easy_results: List) -> List[Dict]:
        """
        Group EasyOCR character detections into lines
        Characters with similar Y coordinates = same line
        
        Args:
            easy_results: List of (bbox, text, confidence) tuples from EasyOCR
            
        Returns:
            List of line dictionaries with text, confidence, character confidences, bboxes
        """
        lines_dict = {}
        
        for bbox, text, confidence in easy_results:
            # Extract Y coordinate (top of character)
            y_pos = int(bbox[0][1])
            
            # Group with tolerance (characters on same line within LINE_GROUPING_TOLERANCE px)
            line_key = round(y_pos / LINE_GROUPING_TOLERANCE) * LINE_GROUPING_TOLERANCE
            
            if line_key not in lines_dict:
                lines_dict[line_key] = {
                    'text': '',
                    'confidences': [],
                    'bboxes': [],
                    'y_position': y_pos
                }
            
            lines_dict[line_key]['text'] += text
            lines_dict[line_key]['confidences'].append(confidence)
            lines_dict[line_key]['bboxes'].append(bbox)
        
        # Convert to sorted list
        lines = []
        for y_pos in sorted(lines_dict.keys()):
            line_data = lines_dict[y_pos]
            avg_confidence = np.mean(line_data['confidences'])
            lines.append({
                'text': line_data['text'].strip(),
                'confidence': float(avg_confidence),
                'char_confidences': line_data['confidences'],
                'bboxes': line_data['bboxes'],
                'y_position': line_data['y_position']
            })
        
        return lines
    
    def _create_ocr_result(
        self,
        line_number: int,
        text: str,
        confidence: float,
        character_confidences: List[float],
        bboxes: List,
        source_model: str
    ) -> OCRResult:
        """Create unified OCR result"""
        return OCRResult(
            line_number=line_number,
            original_text=text,
            corrected_text=None,
            confidence=confidence,
            source_model=source_model,
            character_confidences=character_confidences,
            bboxes=bboxes,
            flagged=confidence < self.confidence_threshold,
            correction_status=CorrectionStatus.PENDING.value,
            correction_timestamp=None,
            correction_source=None
        )
    
    def _flag_low_confidence(self, results: List[OCRResult]) -> List[OCRResult]:
        """Mark lines with confidence < threshold as flagged"""
        for result in results:
            if result.confidence < self.confidence_threshold:
                result.flagged = True
            else:
                result.flagged = False
        
        return results
    
    def _calculate_summary(self, results: List[OCRResult]) -> ConfidenceSummary:
        """Calculate page-level confidence metrics"""
        total = len(results)
        flagged = sum(1 for r in results if r.flagged)
        avg_conf = float(np.mean([r.confidence for r in results])) if total > 0 else 0.0
        flagged_pct = (flagged / total * 100) if total > 0 else 0.0
        
        return ConfidenceSummary(
            total_lines=total,
            flagged_lines=flagged,
            avg_confidence=avg_conf,
            corrected_lines=0,
            flagged_percentage=flagged_pct
        )


# ============================================================================
# 5. USER CORRECTION SERVICE
# ============================================================================

class UserCorrectionService:
    """Handles user corrections to OCR results"""
    
    @staticmethod
    def submit_correction(
        page_state: PageProcessingState,
        line_number: int,
        corrected_text: str,
        user_id: str
    ) -> PageProcessingState:
        """
        User submits correction for a specific line
        
        Args:
            page_state: Current page processing state
            line_number: Line number to correct
            corrected_text: Corrected text
            user_id: User making the correction
            
        Returns:
            Updated PageProcessingState
        """
        logger.info(f"Submitting correction for line {line_number}")
        
        # Find line to correct
        line_to_correct = None
        for result in page_state.ocr_results:
            if result.line_number == line_number:
                line_to_correct = result
                break
        
        if not line_to_correct:
            raise ValueError(f"Line {line_number} not found")
        
        # Update correction fields
        line_to_correct.corrected_text = corrected_text
        line_to_correct.correction_status = CorrectionStatus.CORRECTED.value
        line_to_correct.correction_timestamp = datetime.now().isoformat()
        line_to_correct.correction_source = CorrectionSource.USER.value
        
        # User correction = perfect confidence
        line_to_correct.confidence = 1.0
        line_to_correct.flagged = False
        
        # Update page state
        page_state.updated_at = datetime.now().isoformat()
        page_state.confidence_summary.corrected_lines += 1
        page_state = UserCorrectionService._check_completion(page_state)
        
        logger.info(f"Correction submitted for line {line_number}")
        return page_state
    
    @staticmethod
    def submit_batch_corrections(
        page_state: PageProcessingState,
        corrections: List[Dict],
        user_id: str
    ) -> PageProcessingState:
        """
        User submits multiple corrections at once
        
        Args:
            page_state: Current page processing state
            corrections: List of {'line_number': int, 'corrected_text': str}
            user_id: User making the corrections
            
        Returns:
            Updated PageProcessingState
        """
        logger.info(f"Submitting batch corrections for {len(corrections)} lines")
        
        for correction in corrections:
            line_number = correction['line_number']
            corrected_text = correction['corrected_text']
            
            # Find and update line
            line_result = next(
                (r for r in page_state.ocr_results if r.line_number == line_number),
                None
            )
            
            if not line_result:
                logger.warning(f"Line {line_number} not found, skipping")
                continue
            
            line_result.corrected_text = corrected_text
            line_result.correction_status = CorrectionStatus.CORRECTED.value
            line_result.correction_timestamp = datetime.now().isoformat()
            line_result.correction_source = CorrectionSource.USER.value
            line_result.confidence = 1.0
            line_result.flagged = False
        
        # Update page state
        page_state.updated_at = datetime.now().isoformat()
        page_state.confidence_summary.corrected_lines = len(corrections)
        page_state = UserCorrectionService._check_completion(page_state)
        
        logger.info(f"Batch corrections submitted for {len(corrections)} lines")
        return page_state
    
    @staticmethod
    def skip_correction(
        page_state: PageProcessingState,
        line_number: int,
        reason: str,
        user_id: str
    ) -> PageProcessingState:
        """
        User skips correction (accepts OCR result despite low confidence)
        
        Args:
            page_state: Current page processing state
            line_number: Line number to skip
            reason: Reason for skipping
            user_id: User making the decision
            
        Returns:
            Updated PageProcessingState
        """
        logger.info(f"Skipping correction for line {line_number}, reason: {reason}")
        
        # Find line
        line_result = next(
            (r for r in page_state.ocr_results if r.line_number == line_number),
            None
        )
        
        if not line_result:
            raise ValueError(f"Line {line_number} not found")
        
        # Mark as accepted without correction
        line_result.correction_status = CorrectionStatus.ACCEPTED.value
        line_result.correction_timestamp = datetime.now().isoformat()
        line_result.correction_source = CorrectionSource.USER_SKIP.value
        line_result.metadata['skip_reason'] = reason
        line_result.flagged = False
        
        # Update page state
        page_state.updated_at = datetime.now().isoformat()
        page_state = UserCorrectionService._check_completion(page_state)
        
        logger.info(f"Skipped correction for line {line_number}")
        return page_state
    
    @staticmethod
    def _check_completion(page_state: PageProcessingState) -> PageProcessingState:
        """Check if all flagged lines are handled, update stage if complete"""
        
        pending_corrections = [
            r for r in page_state.ocr_results
            if r.flagged and r.correction_status == CorrectionStatus.PENDING.value
        ]
        
        if not pending_corrections:
            page_state.processing_stage = ProcessingStage.READY_TO_FINALIZE.value
            logger.info("All flagged lines handled, moving to finalization stage")
        
        return page_state


# ============================================================================
# 6. REVIEW SERVICE
# ============================================================================

class ReviewService:
    """Prepares data for user review"""
    
    @staticmethod
    def get_review_view(
        page_state: PageProcessingState,
        original_image_url: str
    ) -> ReviewDTO:
        """
        Get formatted view for user review
        Shows original image + OCR results + correction status
        
        Args:
            page_state: Current page processing state
            original_image_url: URL to serve the original image
            
        Returns:
            ReviewDTO with formatted data for frontend
        """
        logger.info(f"Preparing review view for page {page_state.page_id}")
        
        review_lines = []
        
        for result in page_state.ocr_results:
            line_view = {
                'line_number': result.line_number,
                'original_text': result.original_text,
                'corrected_text': result.corrected_text,
                'displayed_text': result.get_display_text(),
                'confidence': float(result.confidence),
                'source_model': result.source_model,
                'flagged': result.flagged,
                'correction_status': result.correction_status,
                'needs_action': (
                    result.flagged and 
                    result.correction_status == CorrectionStatus.PENDING.value
                ),
                'bbox': result.bboxes[0] if result.bboxes else None,
                'character_confidences': [float(c) for c in result.character_confidences]
            }
            review_lines.append(line_view)
        
        review_dto = ReviewDTO(
            page_id=page_state.page_id,
            original_image_url=original_image_url,
            processing_stage=page_state.processing_stage,
            summary=page_state.confidence_summary,
            lines=review_lines
        )
        
        return review_dto


# ============================================================================
# 7. FINALIZATION SERVICE
# ============================================================================

class FinalizationService:
    """Handles finalization of processed pages"""
    
    @staticmethod
    def finalize_page(
        page_state: PageProcessingState,
        user_id: str
    ) -> FinalizedPageOutput:
        """
        Finalize page after user confirms corrections
        
        Flow:
        1. Validate all lines handled
        2. Build final text
        3. Create metadata
        4. Return finalized output
        
        Args:
            page_state: Current page processing state
            user_id: User finalizing the page
            
        Returns:
            FinalizedPageOutput ready for next pipeline stage
        """
        logger.info(f"Finalizing page {page_state.page_id}")
        
        # Validate completion
        pending_corrections = [
            r for r in page_state.ocr_results
            if r.flagged and r.correction_status == CorrectionStatus.PENDING.value
        ]
        
        if pending_corrections:
            raise ValueError(
                f"{len(pending_corrections)} lines still need correction"
            )
        
        # Build final text (use corrections where available)
        final_lines = []
        for result in page_state.ocr_results:
            final_text = result.get_display_text()
            final_lines.append(final_text)
        
        final_full_text = '\n'.join(final_lines)
        
        # Build metadata
        processing_duration = (
            datetime.fromisoformat(page_state.updated_at) - 
            datetime.fromisoformat(page_state.created_at)
        ).total_seconds()
        
        metadata = {
            'page_id': page_state.page_id,
            'processed_at': page_state.created_at,
            'finalized_at': datetime.now().isoformat(),
            'user_id': user_id,
            'original_ocr_confidence': float(page_state.confidence_summary.avg_confidence),
            'corrected_lines': page_state.confidence_summary.corrected_lines,
            'total_lines': page_state.confidence_summary.total_lines,
            'processing_duration_seconds': processing_duration
        }
        
        # Create finalized output
        finalized = FinalizedPageOutput(
            page_id=page_state.page_id,
            final_text=final_full_text,
            lines=page_state.ocr_results,
            metadata=metadata,
            original_image_path=page_state.image_path,
            original_image_bytes=page_state.original_image
        )
        
        # Mark page as finalized
        page_state.processing_stage = ProcessingStage.FINALIZED.value
        page_state.updated_at = datetime.now().isoformat()
        
        logger.info(f"Page {page_state.page_id} finalized successfully")
        return finalized


# ============================================================================
# 8. MAIN ORCHESTRATOR
# ============================================================================

class HandwritingPageProcessor:
    """Main orchestrator for the complete pipeline"""
    
    def __init__(self, gpu: bool = False):
        """
        Initialize the processor
        
        Args:
            gpu: Whether to use GPU for EasyOCR
        """
        self.ocr_service = OCRExtractionService(gpu=gpu)
        self.correction_service = UserCorrectionService()
        self.review_service = ReviewService()
        self.finalization_service = FinalizationService()
        self.page_states: Dict[str, PageProcessingState] = {}  # In-memory storage
        logger.info("HandwritingPageProcessor initialized")
    
    def process_page(
        self,
        image_path: str,
        user_id: Optional[str] = None
    ) -> PageProcessingState:
        """
        Step 1: Extract text from page
        
        Args:
            image_path: Path to journal page image
            user_id: Optional user identifier
            
        Returns:
            PageProcessingState with OCR results and flagged lines
        """
        logger.info(f"Processing page: {image_path}")
        
        # Run OCR extraction
        page_state = self.ocr_service.extract_text_from_page(image_path, user_id)
        
        # Store in memory
        self.page_states[page_state.page_id] = page_state
        
        return page_state
    
    def get_review(self, page_id: str) -> ReviewDTO:
        """
        Step 2: Get page ready for user review
        
        Args:
            page_id: Page identifier
            
        Returns:
            ReviewDTO with formatted data for user interface
        """
        page_state = self._get_page_state(page_id)
        
        # In production, serve from CDN/storage
        original_image_url = f"data:image/jpeg;base64,{page_state.original_image}"
        
        return self.review_service.get_review_view(page_state, original_image_url)
    
    def submit_correction(
        self,
        page_id: str,
        line_number: int,
        corrected_text: str,
        user_id: str
    ) -> PageProcessingState:
        """
        Step 3a: User submits single correction
        
        Args:
            page_id: Page identifier
            line_number: Line number to correct
            corrected_text: Corrected text
            user_id: User making the correction
            
        Returns:
            Updated PageProcessingState
        """
        page_state = self._get_page_state(page_id)
        
        page_state = self.correction_service.submit_correction(
            page_state, line_number, corrected_text, user_id
        )
        
        self.page_states[page_id] = page_state
        return page_state
    
    def submit_batch_corrections(
        self,
        page_id: str,
        corrections: List[Dict],
        user_id: str
    ) -> PageProcessingState:
        """
        Step 3b: User submits multiple corrections
        
        Args:
            page_id: Page identifier
            corrections: List of {'line_number': int, 'corrected_text': str}
            user_id: User making the corrections
            
        Returns:
            Updated PageProcessingState
        """
        page_state = self._get_page_state(page_id)
        
        page_state = self.correction_service.submit_batch_corrections(
            page_state, corrections, user_id
        )
        
        self.page_states[page_id] = page_state
        return page_state
    
    def skip_correction(
        self,
        page_id: str,
        line_number: int,
        reason: str,
        user_id: str
    ) -> PageProcessingState:
        """
        Step 3c: User skips correction
        
        Args:
            page_id: Page identifier
            line_number: Line to skip
            reason: Reason for skipping
            user_id: User making the decision
            
        Returns:
            Updated PageProcessingState
        """
        page_state = self._get_page_state(page_id)
        
        page_state = self.correction_service.skip_correction(
            page_state, line_number, reason, user_id
        )
        
        self.page_states[page_id] = page_state
        return page_state
    
    def finalize(
        self,
        page_id: str,
        user_id: str
    ) -> FinalizedPageOutput:
        """
        Step 4: Finalize page and prepare for next stage
        
        Args:
            page_id: Page identifier
            user_id: User finalizing the page
            
        Returns:
            FinalizedPageOutput ready for structuring module
        """
        page_state = self._get_page_state(page_id)
        
        finalized_output = self.finalization_service.finalize_page(page_state, user_id)
        
        # Update state
        page_state.processing_stage = ProcessingStage.FINALIZED.value
        self.page_states[page_id] = page_state
        
        return finalized_output
    
    def get_page_status(self, page_id: str) -> Dict:
        """Get current processing status of a page"""
        page_state = self._get_page_state(page_id)
        
        return {
            'page_id': page_id,
            'processing_stage': page_state.processing_stage,
            'summary': page_state.confidence_summary.to_dict(),
            'pending_corrections': [
                {
                    'line_number': r.line_number,
                    'original_text': r.original_text,
                    'confidence': float(r.confidence)
                }
                for r in page_state.ocr_results
                if r.flagged and r.correction_status == CorrectionStatus.PENDING.value
            ]
        }
    
    def _get_page_state(self, page_id: str) -> PageProcessingState:
        """Retrieve page state from storage"""
        if page_id not in self.page_states:
            raise ValueError(f"Page {page_id} not found")
        return self.page_states[page_id]
    
    def save_state_to_file(self, page_id: str, filepath: str) -> None:
        """Save page state to JSON file"""
        page_state = self._get_page_state(page_id)
        
        state_dict = page_state.to_dict()
        # Don't serialize binary image in JSON
        state_dict.pop('original_image', None)
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2, cls=NumpyJSONEncoder)
        
        logger.info(f"Page state saved to {filepath}")
    
    def load_state_from_file(
        self,
        filepath: str,
        original_image_path: Optional[str] = None
    ) -> PageProcessingState:
        """Load page state from JSON file"""
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        
        # Load original image if available
        original_image = b''
        if original_image_path:
            with open(original_image_path, 'rb') as f:
                original_image = f.read()
        
        page_state = PageProcessingState.from_dict(state_dict, original_image)
        self.page_states[page_state.page_id] = page_state
        
        logger.info(f"Page state loaded from {filepath}")
        return page_state


# ============================================================================
# 9. EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Example usage of the complete pipeline
    
    # Initialize processor (gpu=False for CPU, set True if CUDA available)
    processor = HandwritingPageProcessor(gpu=False)
    
    print("\n" + "="*60)
    print("Handwriting Page Processing System - Example Usage")
    print("="*60)
    
    # Note: Replace with actual image path
    image_path = "/Users/krishiv/Desktop/Projects/hybrid-journal/sample_journal_page.jpg.jpeg"
    
    try:
        # Step 1: Process page (OCR extraction)
        print("\n[Step 1] Processing page...")
        page_state = processor.process_page(image_path, user_id="user_123")
        print(f"✓ Page ID: {page_state.page_id}")
        print(f"✓ Total lines: {page_state.confidence_summary.total_lines}")
        print(f"✓ Flagged lines: {page_state.confidence_summary.flagged_lines}")
        print(f"✓ Average confidence: {page_state.confidence_summary.avg_confidence:.2%}")
        
        # Step 2: Get review view
        print("\n[Step 2] Preparing review view...")
        review = processor.get_review(page_state.page_id)
        print(f"✓ Review ready with {len(review.lines)} lines")
        
        # Example: Show flagged lines
        flagged_lines = [l for l in review.lines if l['needs_action']]
        if flagged_lines:
            print(f"\nFlagged lines needing review:")
            for line in flagged_lines[:3]:  # Show first 3
                print(f"  Line {line['line_number']}: '{line['original_text'][:50]}...'")
                print(f"    Confidence: {line['confidence']:.2%}")
        
        # Step 3a: Submit single correction
        if flagged_lines:
            print(f"\n[Step 3a] Submitting single correction...")
            first_flagged = flagged_lines[0]
            corrected_state = processor.submit_correction(
                page_id=page_state.page_id,
                line_number=first_flagged['line_number'],
                corrected_text="Corrected text for this line",
                user_id="user_123"
            )
            print(f"✓ Correction submitted")
            print(f"✓ Corrected lines: {corrected_state.confidence_summary.corrected_lines}")
        
        # Step 3b: Submit batch corrections (example)
        remaining_flagged = [l for l in review.lines if l['needs_action']][1:]
        if remaining_flagged:
            print(f"\n[Step 3b] Submitting batch corrections...")
            batch = [
                {
                    'line_number': l['line_number'],
                    'corrected_text': f"Batch corrected line {l['line_number']}"
                }
                for l in remaining_flagged[:2]
            ]
            batch_state = processor.submit_batch_corrections(
                page_id=page_state.page_id,
                corrections=batch,
                user_id="user_123"
            )
            print(f"✓ Batch corrections submitted for {len(batch)} lines")
        
        # Step 3c: Skip remaining corrections (if any)
        final_flagged = [l for l in review.lines if l['needs_action']]
        if final_flagged:
            print(f"\n[Step 3c] Skipping {len(final_flagged)} remaining lines...")
            skip_state = processor.skip_correction(
                page_id=page_state.page_id,
                line_number=final_flagged[0]['line_number'],
                reason="User confident in OCR result",
                user_id="user_123"
            )
            print(f"✓ Correction skipped")
        
        # Step 4: Finalize page
        print(f"\n[Step 4] Finalizing page...")
        status = processor.get_page_status(page_state.page_id)
        print(f"✓ Processing stage: {status['processing_stage']}")
        
        if status['processing_stage'] == ProcessingStage.READY_TO_FINALIZE.value:
            finalized = processor.finalize(page_state.page_id, user_id="user_123")
            print(f"✓ Page finalized!")
            print(f"✓ Final text preview:")
            print(f"\n{finalized.final_text[:200]}...")
            print(f"\n✓ Metadata: {finalized.metadata}")
        
        # Save state to file
        print(f"\n[Bonus] Saving page state to file...")
        processor.save_state_to_file(page_state.page_id, "/tmp/page_state.json")
        print(f"✓ State saved")
        
    except FileNotFoundError:
        print(f"\n⚠ Sample image not found at '{image_path}'")
        print("To run this example, provide a valid journal page image.")
        print("\nThe system is ready to process images. Usage:")
        print("  processor = HandwritingPageProcessor(gpu=False)")
        print("  page_state = processor.process_page('path/to/image.jpg')")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")