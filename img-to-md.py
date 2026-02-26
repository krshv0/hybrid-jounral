"""
Hybrid Journal System - Image to Markdown Converter
Converts handwritten journal pages to structured markdown files
Using Google Gemini via LangChain
"""

import os
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    
    # Paths
    VAULT_DIR = Path("/Users/krishiv/Downloads/Stuff/journal_vault")
    ATTACHMENTS_DIR = VAULT_DIR / "attachments"
    PROCESSED_DIR = VAULT_DIR / "entries"
    
    # Gemini models
    VISION_MODEL = "gemini-2.5-flash"       # Multimodal vision + text
    REFINEMENT_MODEL = "gemini-2.5-flash"  # Text processing

    # Gemini API key (set GEMINI_API_KEY in env or .env file)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Processing
    MAX_IMAGE_SIZE = (2048, 2048)
    ATTACHMENT_FORMAT = "png"
    
    @classmethod
    def setup_dirs(cls):
        """Create necessary directories"""
        cls.VAULT_DIR.mkdir(parents=True, exist_ok=True)
        cls.ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 2. IMAGE PREPROCESSING
# ============================================================================

class ImagePreprocessor:
    """Handle image preprocessing"""
    
    @staticmethod
    def preprocess_image(image_path: str) -> Tuple[np.ndarray, str]:
        """
        Load and preprocess journal page image
        
        Args:
            image_path: Path to journal page image
            
        Returns:
            - Preprocessed image (numpy array)
            - Base64 encoded image (for LLM)
        """
        
        # Load image
        image = cv2.imread(image_path)
        
        # Auto-rotate if needed
        angle = ImagePreprocessor._detect_rotation(image)
        if angle != 0:
            image = ImagePreprocessor._rotate_image(image, angle)
        
        # Enhance contrast (CLAHE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Resize if too large
        height, width = enhanced_gray.shape
        if height > Config.MAX_IMAGE_SIZE[0] or width > Config.MAX_IMAGE_SIZE[1]:
            scale = min(Config.MAX_IMAGE_SIZE[0] / height, Config.MAX_IMAGE_SIZE[1] / width)
            enhanced_gray = cv2.resize(enhanced_gray, None, fx=scale, fy=scale)
        
        # Convert to base64
        pil_image = Image.fromarray(enhanced_gray)
        base64_image = ImagePreprocessor._encode_to_base64(pil_image)
        
        return enhanced_gray, base64_image
    
    @staticmethod
    def _detect_rotation(image: np.ndarray) -> float:
        """Detect and return rotation angle using edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        
        if lines is None:
            return 0
        
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            angles.append(angle)
        
        median_angle = np.median(angles) if angles else 0
        return median_angle if abs(median_angle) < 45 else 0
    
    @staticmethod
    def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    @staticmethod
    def _encode_to_base64(image: Image) -> str:
        """Convert PIL image to base64 string"""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()


# ============================================================================
# 3. GEMINI MULTIMODAL EXTRACTION
# ============================================================================

class MultimodalExtractor:
    """Extract text, tags and title from journal page using Gemini"""
    
    def __init__(self):
        """Initialize Gemini vision model"""
        if not Config.GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file or environment.")
        self.model = ChatGoogleGenerativeAI(
            model=Config.VISION_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.3
        )
    
    def extract_multimodal(self, base64_images: List[str]) -> Dict:
        """
        Send one or more page images to Gemini for analysis.

        Args:
            base64_images: List of base64-encoded page images

        Returns:
            {
                'text': 'full transcription',
                'tags': ['tag1', 'tag2', ...],
                'title': 'Short Title',
                'page_notes': '...'
            }
        """

        system_prompt = """You are a journal digitization assistant. Your job is to faithfully digitise handwritten journal pages.

YOUR TASK:
1. TRANSCRIBE all handwritten text as accurately as possible.
   - Preserve the writer's exact words, spelling, punctuation and line breaks.
   - Only correct an obvious slip (e.g. a duplicated word) where the intended meaning is unambiguous.
   - Do NOT paraphrase, summarise or rewrite anything.
2. You MAY use context to:
   - Resolve ambiguous handwriting (pick the most contextually plausible word).
   - Infer paragraph breaks or list structure where the layout makes it clear.
   - Add light markdown formatting (blank lines between paragraphs, `- ` for bullet lists) to aid readability, but only where the original layout clearly implies it.
3. Generate 3–6 concise, lowercase tags that reflect the main themes, emotions or topics of the entry.
4. Generate a short title (2–4 words) that captures the essence of the entry. This will be used as the file name — make it specific, not generic.
5. Write a 3–5 sentence AI summary of the entry's key ideas and emotional tone. This is a synthesis, not a transcription — write it in third person.

RESPOND ONLY WITH VALID JSON IN THIS EXACT FORMAT (no markdown fences):
{
    "text": "full transcription here",
    "summary": "3-5 sentence synthesis of the entry here",
    "tags": ["tag1", "tag2", "tag3"],
    "title": "Short Specific Title"
}"""
        
        # Build content list: all images first, then the instruction
        content = []
        for b64 in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
        content.append({"type": "text", "text": system_prompt})

        message = HumanMessage(content=content)

        # Single LLM call
        response = self.model.invoke([message])

        # Strip markdown fences Gemini sometimes adds
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            extracted_data = json.loads(raw)
        except json.JSONDecodeError:
            extracted_data = {
                "text": raw,
                "summary": "",
                "tags": ["journal"],
                "title": "Untitled Entry",
            }

        return extracted_data


# ============================================================================
# 4. MARKDOWN BUILDER
# ============================================================================

class MarkdownBuilder:
    """Build structured markdown from extracted data (no LLM calls)"""

    def build_markdown(
        self,
        extracted_data: Dict,
        page_id: str,
        source_filenames: List[str],
    ) -> str:
        """
        Build complete markdown file from extracted data.

        Args:
            extracted_data: Output from MultimodalExtractor
            page_id: Unique page identifier
            source_filenames: Original image filenames (one or more pages)

        Returns:
            Markdown string ready to save
        """

        text = extracted_data.get('text', '')
        summary = extracted_data.get('summary', '')
        tags = extracted_data.get('tags', ['journal'])
        title = extracted_data.get('title', 'Untitled Entry')

        # Wrap transcribed text as blockquote
        blockquote = "\n".join(
            f"> {line}" if line.strip() else ">"
            for line in text.splitlines()
        )

        frontmatter = self._build_frontmatter(page_id, title, tags, source_filenames)
        tags_line = " ".join(f"#{t.replace(' ', '-')}" for t in tags)
        images_section = self._build_images_section(source_filenames)
        metadata_section = self._build_metadata_section(text)
        summary_section = f"## 🧠 AI Summary\n\n{summary}\n" if summary else ""

        markdown = f"""{frontmatter}

# {title}

{tags_line}

---

{summary_section}
---

{images_section}

---

## Entry

{blockquote}

---

{metadata_section}
"""
        return markdown

    @staticmethod
    def _build_frontmatter(page_id: str, title: str, tags: List[str], source_filenames: List[str]) -> str:
        """Build YAML frontmatter"""
        date = datetime.now().isoformat()
        sources = ", ".join(f'"[[{f}]]"' for f in source_filenames)
        return f"""---
id: {page_id}
title: "{title}"
date: {date}
tags: {json.dumps(tags)}
source_pages: [{sources}]
---"""

    @staticmethod
    def _build_images_section(source_filenames: List[str]) -> str:
        """Embed source page images at the top of the entry"""
        return "\n".join(f"![[{filename}]]" for filename in source_filenames)

    @staticmethod
    def _build_metadata_section(text: str) -> str:
        """Build metadata section"""
        word_count = len(text.split())
        return f"""## Metadata

- **Word Count:** {word_count}
- **Processing Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""


# ============================================================================
# 5. MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class JournalToPipelineMarkdown:
    """Main orchestrator for image-to-markdown conversion"""

    def __init__(self):
        Config.setup_dirs()
        self.preprocessor = ImagePreprocessor()
        self.extractor = MultimodalExtractor()
        self.markdown_builder = MarkdownBuilder()

    def process_journal_page(self, image_paths: List[str]) -> Dict:
        """
        Main pipeline: One or more images → single Markdown entry.

        Flow:
        1. Preprocess all images
        2. Single LLM call — extract text, tags and title
        3. Build structured markdown with embedded source images
        4. Save to vault

        Args:
            image_paths: List of image paths (one per journal page)

        Returns:
            {'success', 'markdown_path', 'page_id', 'word_count'}
        """

        paths = [Path(p) for p in image_paths]

        print(f"\n{'='*60}")
        print(f"Processing: {', '.join(p.name for p in paths)}")
        print(f"{'='*60}\n")

        # STEP 1: Preprocess all pages
        print(f"[1/4] Preprocessing {len(paths)} image(s)...")
        base64_images = []
        for p in paths:
            _, b64 = self.preprocessor.preprocess_image(str(p))
            base64_images.append(b64)

        # STEP 2: Single LLM call
        print("[2/4] Extracting text, tags and title via Gemini...")
        extracted_data = self.extractor.extract_multimodal(base64_images)
        print(f"     ✓ Text extracted ({len(extracted_data.get('text', '').split())} words)")
        print(f"     ✓ Title: {extracted_data.get('title', '?')}")
        print(f"     ✓ Tags: {extracted_data.get('tags', [])}")

        # STEP 3: Build markdown
        print("[3/4] Building structured markdown...")
        page_id = self._generate_page_id(extracted_data.get('title', ''))

        # Copy & rename source images into attachments folder
        import shutil
        saved_image_names = []
        for idx, p in enumerate(paths):
            ext = p.suffix  # preserve original extension
            if len(paths) == 1:
                dest_name = f"{page_id}{ext}"
            else:
                dest_name = f"{page_id} - p{idx + 1}{ext}"
            dest_path = Config.ATTACHMENTS_DIR / dest_name
            shutil.copy2(str(p), str(dest_path))
            saved_image_names.append(dest_name)
        print(f"     ✓ Copied {len(saved_image_names)} image(s) to attachments")

        markdown_content = self.markdown_builder.build_markdown(
            extracted_data=extracted_data,
            page_id=page_id,
            source_filenames=saved_image_names,
        )
        print("     ✓ Markdown built successfully")

        # STEP 4: Save to vault
        print("[4/4] Saving to vault...")
        markdown_path = self._save_to_vault(markdown_content, page_id)
        print(f"     ✓ Saved to: {markdown_path}\n")

        print(f"{'='*60}")
        print("✓ Processing complete!")
        print(f"{'='*60}\n")

        return {
            'success': True,
            'markdown_path': str(markdown_path),
            'page_id': page_id,
            'word_count': len(extracted_data.get('text', '').split()),
        }

    @staticmethod
    def _generate_page_id(title: str) -> str:
        """Generate filename: MMMDD - Title"""
        date_str = datetime.now().strftime('%b%d').upper()   # e.g. FEB22
        clean_title = title.strip() if title.strip() else 'Entry'
        return f"{date_str} - {clean_title}"
    
    @staticmethod
    def _save_to_vault(markdown_content: str, page_id: str) -> Path:
        """Save markdown file to vault"""
        
        filepath = Config.PROCESSED_DIR / f"{page_id}.md"
        
        filepath.write_text(markdown_content, encoding='utf-8')
        
        return filepath


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    import sys

    pipeline = JournalToPipelineMarkdown()

    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
        missing = [p for p in image_paths if not Path(p).exists()]
        if missing:
            print(f"Error: File(s) not found: {', '.join(missing)}")
            return
        result = pipeline.process_journal_page(image_paths)
    else:
        # Default demo path
        image_path = "sample_journal_page.jpg"
        if not Path(image_path).exists():
            print(f"Error: Image not found at {image_path}")
            print("Usage: python img-to-md.py <page1.jpg> [page2.jpg ...]")
            return
        result = pipeline.process_journal_page([image_path])

    if result['success']:
        print(f"\n✓ Successfully processed!")
        print(f"  - Markdown: {result['markdown_path']}")
        print(f"  - Words:    {result['word_count']}")


if __name__ == "__main__":
    main()