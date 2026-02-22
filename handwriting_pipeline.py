"""
Advanced Handwriting Recognition & Structuring Pipeline
========================================================
Implements CLAUDE_BUILD_INSTRUCTIONS.md

Pipeline:
  Raw Image
    → OpenCV Layout Processing (deskew, enhance, segment, detect drawings)
    → Line Crops
    → TrOCR (handwriting OCR + token confidences)
    → Structural Confidence Evaluation (entropy + anomaly detection)
    → Selective Vision-LLM Correction (Claude, only for conf < 0.60)
    → Markdown Structuring (paragraphs, strikethrough, drawings, annotations)
    → Structured Journal Entry
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from spellchecker import SpellChecker
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. ENUMS & CONSTANTS
# ============================================================================

class ProcessingStage(str, Enum):
    IMAGE_PREPROCESSING     = "image_preprocessing"
    LINE_EXTRACTION         = "line_extraction"
    TROCR_PROCESSING        = "trocr_processing"
    CONFIDENCE_EVALUATION   = "confidence_evaluation"
    VISION_LLM_CORRECTION   = "vision_llm_correction"
    MARKDOWN_GENERATION     = "markdown_generation"
    REVIEW_PENDING          = "review_pending"
    USER_CORRECTING         = "user_correcting"
    FINALIZED               = "finalized"


class CorrectionSource(str, Enum):
    TROCR      = "trocr"
    VISION_LLM = "vision_llm"
    USER       = "user"


# Confidence thresholds
CONF_ACCEPT          = 0.75   # Green: accept as-is
CONF_FLAG_OPTIONAL   = 0.50   # Yellow: optional review
CONF_FLAG_MANDATORY  = 0.50   # Below this → mandatory review
CONF_VISION_LLM      = 0.60   # Below this → call Vision-LLM

# Line segmentation
LINE_PAD_PX          = 10     # Pixels padding around each line crop
MIN_LINE_HEIGHT      = 10     # Ignore lines shorter than this
MIN_LINE_GAP         = 8      # Minimum gap (rows) between two lines

# Confidence signal weights
W_TROCR   = 0.40
W_ENTROPY = 0.30
W_ANOMALY = 0.30


# ============================================================================
# 2. DATA MODELS
# ============================================================================

@dataclass
class LineProcessingState:
    """State for a single handwritten line"""
    line_number:           int
    image_crop:            bytes            # JPEG bytes of cropped line image
    bbox:                  Tuple[int,int,int,int]  # (x1,y1,x2,y2) in page coords
    original_text:         str              # TrOCR output
    corrected_text:        Optional[str]    = None
    trocr_confidence:      float           = 0.0
    entropy_score:         float           = 0.0   # 0-1 (lower is better)
    anomaly_score:         float           = 0.0   # 0-1 (lower is better)
    structural_confidence: float           = 0.0   # combined signal
    final_confidence:      float           = 0.0
    token_confidences:     List[float]     = field(default_factory=list)
    char_confidences:      List[float]     = field(default_factory=list)
    needs_user_review:     bool            = False
    vision_llm_called:     bool            = False
    vision_llm_reasoning:  Optional[str]   = None
    correction_source:     str             = CorrectionSource.TROCR.value
    is_strikethrough:      bool            = False
    metadata:              Dict            = field(default_factory=dict)

    def display_text(self) -> str:
        return self.corrected_text if self.corrected_text else self.original_text

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Encode bytes as base64 so it round-trips through JSON
        d["image_crop"] = base64.b64encode(self.image_crop).decode() if self.image_crop else ""
        return d

    @staticmethod
    def from_dict(data: Dict) -> "LineProcessingState":
        data = dict(data)
        if data.get("image_crop"):
            data["image_crop"] = base64.b64decode(data["image_crop"])
        if "bbox" in data:
            data["bbox"] = tuple(data["bbox"])
        return LineProcessingState(**data)


@dataclass
class DrawingRegion:
    """Detected non-text (drawing / sketch) region on the page"""
    bbox:         Tuple[int,int,int,int]   # (x1,y1,x2,y2)
    after_line:   int                      # appears after this line number
    confidence:   float                    # confidence that region is non-text
    description:  str = "Drawing/sketch"


@dataclass
class PageProcessingState:
    """Full state for a journal page"""
    page_id:             str
    image_path:          str
    original_image:      bytes
    preprocessed_image:  bytes               = field(default_factory=bytes)
    lines:               List[LineProcessingState]  = field(default_factory=list)
    drawing_regions:     List[DrawingRegion] = field(default_factory=list)
    overall_confidence:  float               = 0.0
    processing_stage:    str                 = ProcessingStage.IMAGE_PREPROCESSING.value
    deskew_angle:        float               = 0.0
    markdown_output:     str                 = ""
    created_at:          str                 = field(default_factory=lambda: datetime.now().isoformat())
    completed_at:        Optional[str]       = None
    metadata:            Dict                = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "page_id":            self.page_id,
            "image_path":         self.image_path,
            "overall_confidence": self.overall_confidence,
            "processing_stage":   self.processing_stage,
            "deskew_angle":       self.deskew_angle,
            "markdown_output":    self.markdown_output,
            "created_at":         self.created_at,
            "completed_at":       self.completed_at,
            "metadata":           self.metadata,
            "lines":              [l.to_dict() for l in self.lines],
            "drawing_regions":    [asdict(d) for d in self.drawing_regions],
        }


# ============================================================================
# 3. OPENCV LAYOUT PROCESSING MODULE
# ============================================================================

class LayoutProcessor:
    """
    Handles image preprocessing and page layout analysis.

    Responsibilities:
      - Adaptive deskewing via Hough line detection
      - CLAHE contrast enhancement
      - Morphological noise removal
      - Line segmentation using horizontal projection profile
      - Drawing / non-text region detection
    """

    # ------------------------------------------------------------------ #
    # 3.1 Adaptive deskewing
    # ------------------------------------------------------------------ #
    @staticmethod
    def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Straighten the page using Hough line detection on text baselines.

        Returns:
            (deskewed_image, rotation_angle_degrees)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=100, minLineLength=100, maxLineGap=10
        )

        if lines is None:
            logger.debug("No Hough lines found; skipping deskew")
            return image, 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Keep only near-horizontal lines (text baselines)
            if abs(angle) < 30:
                angles.append(angle)

        if not angles:
            return image, 0.0

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:          # negligible skew
            return image, median_angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
        logger.info(f"Deskewed by {median_angle:.2f}°")
        return rotated, median_angle

    # ------------------------------------------------------------------ #
    # 3.2 CLAHE contrast enhancement
    # ------------------------------------------------------------------ #
    @staticmethod
    def enhance_contrast(gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE to a grayscale image."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    # ------------------------------------------------------------------ #
    # 3.3 Morphological noise removal
    # ------------------------------------------------------------------ #
    @staticmethod
    def remove_noise(image: np.ndarray) -> np.ndarray:
        """Gaussian blur + morphological close + open to remove noise specks."""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel, iterations=1)
        return opened

    # ------------------------------------------------------------------ #
    # 3.4 Line segmentation (horizontal projection profile)
    # ------------------------------------------------------------------ #
    @staticmethod
    def segment_lines(
        binary: np.ndarray,
        original_color: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[Tuple[int,int,int,int]]]:
        """
        Extract individual handwritten lines as image crops.

        Args:
            binary:         Binarised (Otsu) grayscale image (0=bg, 255=ink)
            original_color: The preprocessed BGR image to crop from

        Returns:
            (line_crops, bboxes)
            bboxes are (x1, y1, x2, y2) in the page coordinate space
        """
        h, w = binary.shape[:2]

        # Invert so ink = 1, background = 0
        inv = (binary < 128).astype(np.uint8)

        # Horizontal projection: sum of ink pixels per row
        proj = inv.sum(axis=1)          # shape (h,)

        # Moving-average to smooth projection
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        proj_smooth = np.convolve(proj, kernel, mode="same")

        # Find row ranges where ink density > threshold
        threshold = max(1, proj_smooth.max() * 0.02)
        ink_rows = proj_smooth > threshold

        # Group consecutive ink rows into line segments
        line_bboxes: List[Tuple[int,int,int,int]] = []
        in_line = False
        start_row = 0

        for row in range(h):
            if ink_rows[row] and not in_line:
                in_line = True
                start_row = row
            elif not ink_rows[row] and in_line:
                in_line = False
                end_row = row
                height = end_row - start_row
                if height >= MIN_LINE_HEIGHT:
                    # Check gap to previous line
                    if line_bboxes:
                        gap = start_row - line_bboxes[-1][3]
                        if gap < MIN_LINE_GAP:
                            # Merge with previous
                            prev = line_bboxes[-1]
                            line_bboxes[-1] = (prev[0], prev[1], w, end_row)
                            continue
                    line_bboxes.append((0, start_row, w, end_row))

        # Close any open line at bottom of image
        if in_line:
            height = h - start_row
            if height >= MIN_LINE_HEIGHT:
                line_bboxes.append((0, start_row, w, h))

        # Add padding and crop
        line_crops = []
        padded_bboxes = []
        for (x1, y1, x2, y2) in line_bboxes:
            y1p = max(0, y1 - LINE_PAD_PX)
            y2p = min(h, y2 + LINE_PAD_PX)
            crop = original_color[y1p:y2p, x1:x2]
            line_crops.append(crop)
            padded_bboxes.append((x1, y1p, x2, y2p))

        logger.info(f"Segmented {len(line_crops)} lines")
        return line_crops, padded_bboxes

    # ------------------------------------------------------------------ #
    # 3.5 Drawing / non-text detection
    # ------------------------------------------------------------------ #
    @staticmethod
    def detect_drawings(
        binary: np.ndarray,
        line_bboxes: List[Tuple[int,int,int,int]],
    ) -> List[DrawingRegion]:
        """
        Identify regions likely to be drawings rather than text.

        A contour is flagged as a drawing when:
          - Its area is large (>= 2000 px²)
          - Its circularity is low (< 0.3) — irregular shape
          - Its aspect ratio doesn't match typical word/line proportions
        """
        h, w = binary.shape[:2]

        # Work on inverted binary (ink=255)
        inv = cv2.bitwise_not(binary)

        contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        drawings: List[DrawingRegion] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * math.pi * area / (perimeter ** 2)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)

            # Heuristic: drawings are irregular and not extremely wide
            is_drawing = circularity < 0.3 and not (aspect > 5 and bh < 60)
            if not is_drawing:
                continue

            # Determine which line it appears after
            cx = by + bh // 2
            after_line = 0
            for i, (_, ly1, _, ly2) in enumerate(line_bboxes, 1):
                if cx >= ly1:
                    after_line = i

            conf = min(1.0, (1 - circularity) * 0.8 + (1 - min(aspect / 10, 1)) * 0.2)
            drawings.append(DrawingRegion(
                bbox=(bx, by, bx + bw, by + bh),
                after_line=after_line,
                confidence=round(conf, 3),
            ))

        logger.info(f"Detected {len(drawings)} drawing regions")
        return drawings

    # ------------------------------------------------------------------ #
    # 3.6 Strikethrough detection
    # ------------------------------------------------------------------ #
    @staticmethod
    def detect_strikethrough(
        line_crop: np.ndarray,
        line_height: int,
    ) -> bool:
        """
        Detect whether a line has a strikethrough mark.

        Method: look for a strong horizontal line passing through
        the vertical centre third of the crop.
        """
        gray = cv2.cvtColor(line_crop, cv2.COLOR_BGR2GRAY) if line_crop.ndim == 3 else line_crop
        h, w = gray.shape[:2]

        if h < 6:
            return False

        # Focus on central vertical band (33% – 67%)
        y_start = h // 3
        y_end = 2 * h // 3
        roi = gray[y_start:y_end, :]

        edges = cv2.Canny(roi, 30, 100, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=max(20, w // 4),
            minLineLength=w // 2,
            maxLineGap=10,
        )
        return lines is not None and len(lines) > 0

    # ------------------------------------------------------------------ #
    # 3.7 Full preprocessing pipeline
    # ------------------------------------------------------------------ #
    def preprocess(
        self,
        image_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Load and preprocess image.

        Returns:
            (color_preprocessed, binary_for_segmentation, deskew_angle)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Cannot load image: {image_path}")
        logger.info(f"Loaded image {image_path}: {image.shape}")

        # Deskew
        image, angle = self.deskew(image)

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        enhanced = self.enhance_contrast(gray)

        # Noise removal
        denoised = self.remove_noise(enhanced)

        # Binarise (Otsu threshold)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Reconstruct colour-looking image for line crops (use enhanced as BGR)
        color_prep = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        logger.info("Preprocessing complete")
        return color_prep, binary, angle


# ============================================================================
# 4. TrOCR SERVICE
# ============================================================================

class TrOCRService:
    """
    Runs microsoft/trocr-base-handwritten on individual line crops.

    Extracts:
      - Recognised text
      - Token-level confidence (softmax of logits)
      - Character-level confidence (interpolated from tokens)
    """

    MODEL_ID = "microsoft/trocr-base-handwritten"

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Loading TrOCR model on {self.device}…")
        self.processor = TrOCRProcessor.from_pretrained(self.MODEL_ID)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_ID)
        self.model.to(self.device)
        self.model.eval()
        logger.info("TrOCR model loaded")

    # ------------------------------------------------------------------ #
    # 4.1 Process a single line crop
    # ------------------------------------------------------------------ #
    def process_line(
        self,
        line_crop: np.ndarray,
        line_number: int,
    ) -> Dict[str, Any]:
        """
        Run TrOCR on one line image.

        Returns a dict matching the spec:
          line_number, text, token_confidences, character_confidences,
          mean_confidence, min_confidence, tokens
        """
        pil_img = Image.fromarray(cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)).convert("RGB")
        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=128,
            )

        sequences  = outputs.sequences          # (1, seq_len)
        scores     = outputs.scores             # tuple of (1, vocab_size) per step

        # Decode text
        text = self.processor.batch_decode(sequences, skip_special_tokens=True)[0]

        # ── Token confidences ────────────────────────────────────────────
        token_confidences: List[float] = []
        tokens: List[str] = []

        for step_scores in scores:
            probs = torch.softmax(step_scores[0], dim=-1)
            conf  = float(probs.max().item())
            token_confidences.append(conf)

        # Decode individual token strings
        generated_ids = sequences[0, 1:]   # skip BOS
        for tid in generated_ids:
            tok = self.processor.tokenizer.decode([int(tid)], skip_special_tokens=True)
            tokens.append(tok)

        # Align token_confidences length with tokens (may differ by 1 due to EOS)
        min_len = min(len(token_confidences), len(tokens))
        token_confidences = token_confidences[:min_len]
        tokens            = tokens[:min_len]

        # ── Character confidences ────────────────────────────────────────
        char_confidences = self._token_to_char_confidences(tokens, token_confidences, text)

        mean_conf = float(np.mean(token_confidences)) if token_confidences else 0.0
        min_conf  = float(np.min(token_confidences))  if token_confidences else 0.0

        return {
            "line_number":           line_number,
            "text":                  text,
            "token_confidences":     token_confidences,
            "character_confidences": char_confidences,
            "mean_confidence":       mean_conf,
            "min_confidence":        min_conf,
            "tokens":                tokens,
        }

    # ------------------------------------------------------------------ #
    # 4.2 Token → character confidence mapping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _token_to_char_confidences(
        tokens: List[str],
        token_confs: List[float],
        full_text: str,
    ) -> List[float]:
        """
        Map token-level confidence to individual characters by distributing
        each token's confidence across its constituent characters.
        """
        char_confs: List[float] = []
        for tok, conf in zip(tokens, token_confs):
            for _ in tok:
                char_confs.append(conf)
        # Pad or trim to match decoded text length
        target_len = len(full_text)
        if len(char_confs) < target_len:
            last = char_confs[-1] if char_confs else 0.0
            char_confs.extend([last] * (target_len - len(char_confs)))
        return char_confs[:target_len]

    # ------------------------------------------------------------------ #
    # 4.3 Process all lines on a page
    # ------------------------------------------------------------------ #
    def process_all_lines(
        self,
        line_crops: List[np.ndarray],
        line_bboxes: List[Tuple[int,int,int,int]],
    ) -> List[LineProcessingState]:
        """
        Process every line crop and return initial LineProcessingState objects.
        """
        states: List[LineProcessingState] = []
        total = len(line_crops)
        for i, (crop, bbox) in enumerate(zip(line_crops, line_bboxes), 1):
            logger.info(f"TrOCR: processing line {i}/{total}")
            result = self.process_line(crop, line_number=i)

            # Encode crop as JPEG bytes
            success, buf = cv2.imencode(".jpg", crop)
            crop_bytes = bytes(buf) if success else b""

            state = LineProcessingState(
                line_number=i,
                image_crop=crop_bytes,
                bbox=bbox,
                original_text=result["text"],
                trocr_confidence=result["mean_confidence"],
                token_confidences=result["token_confidences"],
                char_confidences=result["character_confidences"],
                correction_source=CorrectionSource.TROCR.value,
            )
            states.append(state)

        return states


# ============================================================================
# 5. STRUCTURAL CONFIDENCE EVALUATION MODULE
# ============================================================================

class ConfidenceEvaluator:
    """
    Calculates a multi-signal confidence score per line.

    Signals:
      1. TrOCR mean confidence         (weight 0.40)
      2. Character entropy              (weight 0.30, inverted)
      3. Word shape anomaly score       (weight 0.30, inverted)
    """

    def __init__(self):
        self._spell = SpellChecker()

    # ------------------------------------------------------------------ #
    # 5.1 Character entropy analysis
    # ------------------------------------------------------------------ #
    @staticmethod
    def _character_entropy(char_confidences: List[float]) -> float:
        """
        Derive a normalised entropy metric from character-level confidences.

        We treat each char confidence p as a Bernoulli(p) variable and
        compute its binary entropy H(p) = -p log p - (1-p) log(1-p).
        The mean across all characters is returned (0-1, lower = more certain).
        """
        if not char_confidences:
            return 0.5

        entropies = []
        for p in char_confidences:
            p = max(1e-9, min(1 - 1e-9, p))
            h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
            entropies.append(h)

        return float(np.mean(entropies))  # 0 = perfect confidence, 1 = max uncertainty

    # ------------------------------------------------------------------ #
    # 5.2 Word shape anomaly detection
    # ------------------------------------------------------------------ #
    def _anomaly_score(self, text: str) -> Tuple[float, List[str]]:
        """
        Fraction of words not recognised by spell-checker (with edit-distance >2
        to the nearest known word).

        Returns (score 0-1, list of anomalous words).
        """
        words = re.findall(r"[a-zA-Z']+", text)
        if not words:
            return 0.0, []

        anomalous: List[str] = []
        for word in words:
            if word.lower() in self._spell:
                continue
            # Check edit distance to best candidate
            candidates = self._spell.candidates(word)
            if candidates:
                best = min(candidates, key=lambda c: self._edit_distance(word.lower(), c))
                if self._edit_distance(word.lower(), best) > 2:
                    anomalous.append(word)
            else:
                anomalous.append(word)

        score = len(anomalous) / len(words) if words else 0.0
        return round(score, 4), anomalous

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """Standard Levenshtein edit distance."""
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]

    # ------------------------------------------------------------------ #
    # 5.3 Combined confidence + flagging decision
    # ------------------------------------------------------------------ #
    def evaluate(self, line: LineProcessingState) -> LineProcessingState:
        """
        Compute structural_confidence and final_confidence; set needs_user_review.
        Mutates and returns the line state.
        """
        # Signal 1 – TrOCR
        trocr_conf = line.trocr_confidence

        # Signal 2 – Entropy (0-1, lower is better → invert for confidence)
        entropy_raw = self._character_entropy(line.char_confidences)
        entropy_conf = 1.0 - entropy_raw           # high value = low entropy = good

        # Signal 3 – Anomaly (0-1, lower is better → invert)
        anomaly_raw, anomalous_words = self._anomaly_score(line.original_text)
        anomaly_conf = 1.0 - anomaly_raw

        structural_conf = (
            W_TROCR   * trocr_conf +
            W_ENTROPY * entropy_conf +
            W_ANOMALY * anomaly_conf
        )
        structural_conf = round(max(0.0, min(1.0, structural_conf)), 4)

        line.entropy_score         = round(entropy_raw,  4)
        line.anomaly_score         = round(anomaly_raw,  4)
        line.structural_confidence = structural_conf
        line.final_confidence      = structural_conf
        line.needs_user_review     = structural_conf < CONF_ACCEPT

        if anomalous_words:
            line.metadata["anomalous_words"] = anomalous_words

        return line

    def evaluate_all(
        self, lines: List[LineProcessingState]
    ) -> List[LineProcessingState]:
        for line in lines:
            self.evaluate(line)
        logger.info(
            f"Confidence eval done. "
            f"Flagged: {sum(1 for l in lines if l.needs_user_review)}/{len(lines)}"
        )
        return lines


# ============================================================================
# 6. VISION-LLM CORRECTION MODULE (Claude)
# ============================================================================

class VisionLLMCorrector:
    """
    Calls Google Gemini vision API (free tier) for lines with confidence < CONF_VISION_LLM.

    Free tier: gemini-1.5-flash — 15 req/min, 1,500 req/day
    Get key free at: https://aistudio.google.com
    Set env var: GEMINI_API_KEY

    Conservative correction strategy:
      - Only applies when Vision-LLM confidence > 0.80
      - Only accepts short changes (edit-distance ≤ 3)
      - Never makes semantic rewrites
    """

    PROMPT_TEMPLATE = (
        "You are helping correct handwritten text recognition. "
        "You see both the original handwritten line image and the OCR-generated text. "
        "Your task is to:\n"
        "1. Verify if the OCR text is correct.\n"
        "2. If incorrect, provide the corrected text.\n"
        "3. Be conservative – only correct obvious character-level errors.\n"
        "4. Preserve ambiguity if the text is genuinely unclear.\n"
        "5. Do NOT rewrite, paraphrase, or improve the text.\n"
        "Respond ONLY with valid JSON (no markdown fences):\n"
        '{"corrected_text": "<text>", "is_different": <bool>, '
        '"confidence": <0.0-1.0>, "reasoning": "<brief>"}\n\n'
        "OCR Result: '{ocr_text}'\n"
        "Is this correct based on the handwritten line image?"
    )

    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise RuntimeError(
                "google-genai package not installed. "
                "Run: pip install google-genai"
            )
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var "
                "or pass api_key argument. "
                "Get a free key at https://aistudio.google.com"
            )
        self.client = google_genai.Client(api_key=api_key)

    # ------------------------------------------------------------------ #
    # 6.1 Call Vision-LLM for a single line
    # ------------------------------------------------------------------ #
    def correct_line(self, line: LineProcessingState) -> LineProcessingState:
        """
        Submit a low-confidence line to Gemini for correction.
        Updates line in-place; returns it.
        """
        if not line.image_crop:
            logger.warning(f"Line {line.line_number}: no image crop, skipping VLM")
            return line

        try:
            pil_img = Image.open(BytesIO(line.image_crop)).convert("RGB")
            prompt  = self.PROMPT_TEMPLATE.format(ocr_text=line.original_text)

            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, pil_img],
            )
            raw = response.text.strip()
            # Strip markdown fences if model adds them anyway
            raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
            parsed = json.loads(raw)

            vlm_conf = float(parsed.get("confidence", 0.0))
            is_diff  = bool(parsed.get("is_different", False))
            corrected = parsed.get("corrected_text", line.original_text)

            line.vision_llm_called    = True
            line.vision_llm_reasoning = parsed.get("reasoning", "")

            # Apply correction only if it passes conservative checks
            if is_diff and self._is_acceptable_correction(
                line.original_text, corrected, vlm_conf
            ):
                line.corrected_text    = corrected
                line.correction_source = CorrectionSource.VISION_LLM.value
                line.final_confidence  = vlm_conf
                logger.info(
                    f"Line {line.line_number} corrected by VLM: "
                    f"'{line.original_text}' → '{corrected}'"
                )
            else:
                line.final_confidence = max(line.final_confidence, vlm_conf * 0.5)

        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning(f"Line {line.line_number} VLM parse error: {exc}")
        except Exception as exc:
            logger.error(f"Line {line.line_number} VLM error: {exc}")

        return line

    # ------------------------------------------------------------------ #
    # 6.2 Conservative correction gating
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_acceptable_correction(
        original: str,
        corrected: str,
        vlm_confidence: float,
    ) -> bool:
        """
        Returns True only when all conservative criteria are met.
        """
        if vlm_confidence < 0.80:
            return False
        edit_dist = sum(a != b for a, b in zip(original, corrected))
        edit_dist += abs(len(original) - len(corrected))
        if edit_dist > 3:
            return False
        # Reject purely semantic or punctuation-normalisation changes
        if original.lower().replace("'", "") == corrected.lower().replace("'", ""):
            return False
        return True

    # ------------------------------------------------------------------ #
    # 6.3 Process all flagged lines (selective)
    # ------------------------------------------------------------------ #
    def correct_flagged_lines(
        self,
        lines: List[LineProcessingState],
    ) -> Tuple[List[LineProcessingState], int]:
        """
        Call Vision-LLM only for lines with final_confidence < CONF_VISION_LLM.

        Returns (updated_lines, number_of_vlm_calls).
        """
        calls = 0
        for line in lines:
            if line.final_confidence < CONF_VISION_LLM:
                self.correct_line(line)
                calls += 1
        logger.info(f"Vision-LLM corrections: {calls} API calls made")
        return lines, calls


# ============================================================================
# 7. MARKDOWN STRUCTURING MODULE
# ============================================================================

class MarkdownStructurer:
    """
    Converts per-line OCR results into a structured Markdown journal entry.

    Features:
      - YAML frontmatter with processing metadata
      - Line confidence annotations (✓ / ⚠️ / ❌ / 🔧)
      - Paragraph grouping by vertical gap
      - ~~Strikethrough~~ formatting
      - Drawing region markers
    """

    # ------------------------------------------------------------------ #
    # 7.1 Paragraph detection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_paragraphs(
        lines: List[LineProcessingState],
    ) -> List[List[LineProcessingState]]:
        """
        Group consecutive lines into paragraphs.

        A new paragraph starts when the vertical gap between consecutive
        line bboxes exceeds 1.5× the typical inter-line spacing.
        """
        if not lines:
            return []

        # Calculate inter-line gaps
        gaps: List[int] = []
        for i in range(1, len(lines)):
            prev_y2 = lines[i - 1].bbox[3]
            curr_y1 = lines[i].bbox[1]
            gaps.append(max(0, curr_y1 - prev_y2))

        if not gaps:
            return [lines]

        median_gap = float(np.median(gaps)) if gaps else 0
        para_threshold = max(20, median_gap * 1.5)

        paragraphs: List[List[LineProcessingState]] = []
        current_para: List[LineProcessingState] = [lines[0]]

        for i, gap in enumerate(gaps):
            line = lines[i + 1]
            if gap > para_threshold:
                paragraphs.append(current_para)
                current_para = [line]
            else:
                current_para.append(line)

        paragraphs.append(current_para)
        return paragraphs

    # ------------------------------------------------------------------ #
    # 7.2 Confidence indicator
    # ------------------------------------------------------------------ #
    @staticmethod
    def _conf_indicator(line: LineProcessingState) -> str:
        """Return visual indicator + score for a line."""
        conf = line.final_confidence
        if line.correction_source == CorrectionSource.VISION_LLM.value:
            icon = "🔧"
            return f"{icon} [{conf:.2f}*]"
        if conf >= CONF_ACCEPT:
            return f"✓ [{conf:.2f}]"
        if conf >= CONF_FLAG_OPTIONAL:
            return f"⚠️ [{conf:.2f}]"
        return f"❌ [{conf:.2f}]"

    # ------------------------------------------------------------------ #
    # 7.3 Format a single line of text
    # ------------------------------------------------------------------ #
    def _format_line(self, line: LineProcessingState) -> str:
        text = line.display_text() or "*[unreadable]*"
        if line.is_strikethrough:
            text = f"~~{text}~~"
        indicator = self._conf_indicator(line)
        parts = [text, indicator]

        # Inline comment for very low confidence
        if line.final_confidence < CONF_FLAG_MANDATORY:
            note_parts = []
            if line.metadata.get("anomalous_words"):
                note_parts.append(
                    "possible misreads: "
                    + ", ".join(line.metadata["anomalous_words"][:3])
                )
            if line.vision_llm_reasoning:
                note_parts.append(line.vision_llm_reasoning[:80])
            if note_parts:
                parts.append(f"<!-- OCR uncertain: {'; '.join(note_parts)} -->")

        return "  ".join(parts)

    # ------------------------------------------------------------------ #
    # 7.4 Generate full markdown
    # ------------------------------------------------------------------ #
    def generate(
        self,
        page: PageProcessingState,
        vision_llm_calls: int,
    ) -> str:
        lines      = page.lines
        drawings   = page.drawing_regions
        total      = len(lines)
        flagged    = sum(1 for l in lines if l.needs_user_review)
        avg_conf   = float(np.mean([l.final_confidence for l in lines])) if lines else 0.0
        vlm_corrected = sum(
            1 for l in lines if l.correction_source == CorrectionSource.VISION_LLM.value
        )

        # ── Frontmatter ──────────────────────────────────────────────────
        frontmatter = {
            "date":               datetime.now().strftime("%Y-%m-%d"),
            "deskew_angle":       round(page.deskew_angle, 2),
            "average_confidence": round(avg_conf, 4),
            "lines_flagged":      f"{flagged}/{total}",
            "processing_timestamp": datetime.now().isoformat(),
            "original_image":     page.image_path,
            "processing_metadata": {
                "trocr_model":               TrOCRService.MODEL_ID,
                "vision_llm_calls":          vision_llm_calls,
                "lines_corrected_by_vision": vlm_corrected,
            },
        }
        fm_text = "---\n" + yaml.dump(frontmatter, default_flow_style=False) + "---\n"

        # ── Journal body ─────────────────────────────────────────────────
        paragraphs = self._detect_paragraphs(lines)

        body_parts: List[str] = ["# Journal Entry\n"]

        # Build a lookup: after which line_number → drawing descriptions
        drawing_map: Dict[int, List[DrawingRegion]] = {}
        for dr in drawings:
            drawing_map.setdefault(dr.after_line, []).append(dr)

        def _render_drawings_after(line_no: int) -> List[str]:
            result = []
            for dr in drawing_map.get(line_no, []):
                result.append(
                    f"\n> *[{dr.description} — "
                    f"region ({dr.bbox[0]},{dr.bbox[1]})→"
                    f"({dr.bbox[2]},{dr.bbox[3]}), "
                    f"confidence: {dr.confidence:.0%}]*\n"
                )
            return result

        for para in paragraphs:
            para_lines: List[str] = []
            for line in para:
                para_lines.append(self._format_line(line))
                para_lines.extend(_render_drawings_after(line.line_number))
            body_parts.append("\n".join(para_lines))
            body_parts.append("")           # blank line between paragraphs

        # ── Footer metadata ───────────────────────────────────────────────
        body_parts += [
            "---",
            "## Metadata",
            f"- Lines processed: {total}",
            f"- Average confidence: {avg_conf:.1%}",
            f"- Flagged for review: {flagged} lines ({flagged/max(total,1):.0%})",
            f"- Vision-LLM corrections: {vlm_corrected}",
            "",
            "## Original Image",
            f"![]({page.image_path})",
        ]

        return fm_text + "\n" + "\n".join(body_parts)


# ============================================================================
# 8. MAIN ORCHESTRATOR
# ============================================================================

class HandwritingPipeline:
    """
    End-to-end orchestrator.

    Usage:
        pipeline = HandwritingPipeline(gemini_api_key="AIza...")
        result = pipeline.process("journal_page.jpg")
        print(result.markdown_output)
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        use_vision_llm:    bool = True,
        device:            Optional[str] = None,
    ):
        self.layout_processor = LayoutProcessor()
        self.trocr_service     = TrOCRService(device=device)
        self.conf_evaluator    = ConfidenceEvaluator()
        self.md_structurer     = MarkdownStructurer()

        self.vision_llm: Optional[VisionLLMCorrector] = None
        if use_vision_llm and GEMINI_AVAILABLE:
            key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
            if key:
                self.vision_llm = VisionLLMCorrector(api_key=key)
                logger.info("Vision-LLM corrector enabled (Gemini free tier)")
            else:
                logger.warning(
                    "GEMINI_API_KEY not set – Vision-LLM correction disabled. "
                    "Get a free key at https://aistudio.google.com"
                )
        elif not GEMINI_AVAILABLE:
            logger.warning("google-genai package not installed – Vision-LLM disabled")

        # In-memory page store
        self._pages: Dict[str, PageProcessingState] = {}

    # ------------------------------------------------------------------ #
    # Main process entry point
    # ------------------------------------------------------------------ #
    def process(
        self,
        image_path: str,
        save_markdown: Optional[str] = None,
        save_state:    Optional[str] = None,
    ) -> PageProcessingState:
        """
        Full pipeline: image → structured markdown.

        Args:
            image_path:     Path to the journal page image.
            save_markdown:  If provided, write markdown to this path.
            save_state:     If provided, write JSON state to this path.

        Returns:
            Completed PageProcessingState with markdown_output populated.
        """
        page = PageProcessingState(
            page_id=str(uuid.uuid4()),
            image_path=image_path,
            original_image=Path(image_path).read_bytes(),
        )
        self._pages[page.page_id] = page

        # ── Stage 1: Preprocessing ───────────────────────────────────────
        logger.info("Stage 1/6 – Image preprocessing")
        page.processing_stage = ProcessingStage.IMAGE_PREPROCESSING.value
        color_prep, binary, angle = self.layout_processor.preprocess(image_path)
        page.deskew_angle = angle

        ok, buf = cv2.imencode(".jpg", color_prep)
        page.preprocessed_image = bytes(buf) if ok else b""

        # ── Stage 2: Line extraction ─────────────────────────────────────
        logger.info("Stage 2/6 – Line segmentation")
        page.processing_stage = ProcessingStage.LINE_EXTRACTION.value
        line_crops, line_bboxes = self.layout_processor.segment_lines(binary, color_prep)

        # Detect drawings
        page.drawing_regions = self.layout_processor.detect_drawings(binary, line_bboxes)

        # ── Stage 3: TrOCR ───────────────────────────────────────────────
        logger.info("Stage 3/6 – TrOCR inference")
        page.processing_stage = ProcessingStage.TROCR_PROCESSING.value
        page.lines = self.trocr_service.process_all_lines(line_crops, line_bboxes)

        # Detect strikethrough on each line
        for line, crop in zip(page.lines, line_crops):
            line_h = line.bbox[3] - line.bbox[1]
            line.is_strikethrough = self.layout_processor.detect_strikethrough(crop, line_h)

        # ── Stage 4: Confidence evaluation ───────────────────────────────
        logger.info("Stage 4/6 – Confidence evaluation")
        page.processing_stage = ProcessingStage.CONFIDENCE_EVALUATION.value
        page.lines = self.conf_evaluator.evaluate_all(page.lines)
        page.overall_confidence = float(
            np.mean([l.final_confidence for l in page.lines])
        ) if page.lines else 0.0

        # ── Stage 5: Vision-LLM correction ───────────────────────────────
        logger.info("Stage 5/6 – Vision-LLM correction")
        page.processing_stage = ProcessingStage.VISION_LLM_CORRECTION.value
        vlm_calls = 0
        if self.vision_llm:
            page.lines, vlm_calls = self.vision_llm.correct_flagged_lines(page.lines)
        else:
            logger.info("Vision-LLM skipped (not configured)")

        # ── Stage 6: Markdown generation ─────────────────────────────────
        logger.info("Stage 6/6 – Markdown generation")
        page.processing_stage = ProcessingStage.MARKDOWN_GENERATION.value
        page.markdown_output = self.md_structurer.generate(page, vlm_calls)

        page.processing_stage = ProcessingStage.REVIEW_PENDING.value
        page.completed_at     = datetime.now().isoformat()

        logger.info(
            f"Pipeline complete. "
            f"Page ID: {page.page_id}, "
            f"Lines: {len(page.lines)}, "
            f"Avg conf: {page.overall_confidence:.2%}"
        )

        # Optional saves
        if save_markdown:
            Path(save_markdown).write_text(page.markdown_output, encoding="utf-8")
            logger.info(f"Markdown saved to {save_markdown}")

        if save_state:
            self.save_state(page.page_id, save_state)

        return page

    # ------------------------------------------------------------------ #
    # State persistence
    # ------------------------------------------------------------------ #
    def save_state(self, page_id: str, filepath: str) -> None:
        page = self._get(page_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(page.to_dict(), f, indent=2)
        logger.info(f"State saved to {filepath}")

    def _get(self, page_id: str) -> PageProcessingState:
        if page_id not in self._pages:
            raise ValueError(f"Page {page_id} not found")
        return self._pages[page_id]


# ============================================================================
# 9. CLI / EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys

    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Users/krishiv/Desktop/Projects/hybrid-journal/sample_journal_page.jpg.jpeg"
    )

    print("\n" + "=" * 64)
    print("  Handwriting Recognition & Structuring Pipeline")
    print("=" * 64)

    # Gemini API key from env (free tier – get at aistudio.google.com).
    # Vision-LLM silently disabled if GEMINI_API_KEY not set.
    pipeline = HandwritingPipeline(use_vision_llm=True)

    try:
        result = pipeline.process(
            image_path=image_path,
            save_markdown="/tmp/journal_output.md",
            save_state="/tmp/page_state.json",
        )

        print(f"\n{'─'*64}")
        print(f"  Page ID      : {result.page_id}")
        print(f"  Lines        : {len(result.lines)}")
        print(f"  Drawings     : {len(result.drawing_regions)}")
        print(f"  Avg Conf     : {result.overall_confidence:.2%}")
        print(f"  Deskew angle : {result.deskew_angle:.2f}°")

        flagged = [l for l in result.lines if l.needs_user_review]
        print(f"  Flagged      : {len(flagged)}/{len(result.lines)}")

        print(f"\n  Markdown saved → /tmp/journal_output.md")
        print(f"  State saved    → /tmp/page_state.json")

        print(f"\n{'─'*64}")
        print("  Markdown preview (first 600 chars):")
        print("─" * 64)
        print(result.markdown_output[:600])
        print("  …")

    except FileNotFoundError:
        print(f"\n⚠  Image not found: '{image_path}'")
        print("   Pass the image path as the first argument:")
        print("   python handwriting_pipeline.py path/to/page.jpg")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
