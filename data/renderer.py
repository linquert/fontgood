"""
Font Renderer with Multi-Character Sequence Support
Renders 1, 2, or 3 character sequences to study kerning and composition
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceRenderer:
    """
    Render character sequences (1-3 chars) in various fonts
    
    For mech interp, we want to understand:
    - Single characters: basic features
    - Two characters: kerning, spacing
    - Three characters: composition, rhythm
    """
    
    def __init__(self,
                 image_size: int = 128,
                 font_size: int = 48,
                 output_dir: str = "./data/rendered"):
        self.image_size = image_size
        self.font_size = font_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def render_sequence(self,
                       sequence: str,
                       font_path: Path) -> Optional[np.ndarray]:
        """
        Render a character sequence (1-3 chars)
        
        Returns:
            numpy array (H, W) with values in [0, 1]
        """
        try:
            # Create blank image (white background)
            img = Image.new('L', (self.image_size, self.image_size), color=255)
            draw = ImageDraw.Draw(img)
            
            # Load font
            font = ImageFont.truetype(str(font_path), size=self.font_size)
            
            # Get bounding box for centering
            bbox = draw.textbbox((0, 0), sequence, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center the text
            x = (self.image_size - text_width) // 2 - bbox[0]
            y = (self.image_size - text_height) // 2 - bbox[1]
            
            # Draw text (black on white)
            draw.text((x, y), sequence, fill=0, font=font)
            
            # Convert to numpy and normalize
            arr = np.array(img).astype(np.float32) / 255.0
            arr = 1.0 - arr  # Invert: character=1, background=0
            
            return arr
            
        except Exception as e:
            logger.debug(f"Failed to render '{sequence}' in {font_path.name}: {e}")
            return None
    
    def generate_sequences(self,
                          characters: str,
                          max_length: int = 3,
                          common_bigrams: Optional[List[str]] = None,
                          common_trigrams: Optional[List[str]] = None) -> List[str]:
        """
        Generate sequence list
        
        For mech interp, we want:
        - All single characters (52)
        - Common bigrams (for kerning analysis)
        - Common trigrams (for composition)
        
        Total: ~52 + 20 bigrams + 20 trigrams = ~100 sequences per font
        """
        sequences = []
        
        # 1. Single characters
        sequences.extend(list(characters))
        
        # 2. Bigrams (if enabled)
        if max_length >= 2:
            if common_bigrams:
                sequences.extend(common_bigrams)
            else:
                # Generate some systematic bigrams
                # Uppercase + lowercase combinations
                for uc in "ABCDEFGHIJ":
                    for lc in "abcde":
                        sequences.append(uc + lc)
                        if len(sequences) >= 52 + 30:
                            break
                    if len(sequences) >= 52 + 30:
                        break
        
        # 3. Trigrams (if enabled)
        if max_length >= 3:
            if common_trigrams:
                sequences.extend(common_trigrams)
            else:
                # Add some systematic trigrams
                test_trigrams = ["The", "ABC", "xyz", "Typ", "aaa", "iii"]
                sequences.extend(test_trigrams)
        
        return sequences
    
    def render_font_dataset(self,
                           font_info: Dict,
                           sequences: List[str],
                           save_images: bool = False) -> Dict:
        """
        Render all sequences for a single font
        
        Args:
            font_info: Dict with font metadata
            sequences: List of character sequences to render
            save_images: Whether to save individual PNG files
        
        Returns:
            Dict with rendered data
        """
        font_name = font_info['name']
        font_path = font_info['files'][0]
        attributes = font_info['attributes']
        
        rendered_data = {
            'font_name': font_name,
            'font_path': str(font_path),
            'attributes': attributes,
            'sequences': {},
        }
        
        if save_images:
            font_dir = self.output_dir / font_name
            font_dir.mkdir(exist_ok=True)
        
        success_count = 0
        for seq in sequences:
            arr = self.render_sequence(seq, font_path)
            
            if arr is not None:
                rendered_data['sequences'][seq] = arr
                success_count += 1
                
                if save_images:
                    # Save as PNG
                    img = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
                    # Use safe filename
                    safe_name = "".join(c if c.isalnum() else "_" for c in seq)
                    img.save(font_dir / f"{safe_name}.png")
        
        logger.debug(f"Rendered {success_count}/{len(sequences)} sequences for {font_name}")
        
        return rendered_data
    
    def render_dataset(self,
                      fonts: List[Dict],
                      sequences: List[str],
                      split_name: str = "train") -> List[Dict]:
        """
        Render complete dataset for a font split
        
        Returns:
            List of rendered font dicts
        """
        rendered_fonts = []
        
        logger.info(f"Rendering {len(fonts)} fonts ({len(sequences)} sequences each) for {split_name}...")
        
        save_images = (split_name == "train")  # Only save train images for inspection
        
        for font_info in tqdm(fonts, desc=f"Rendering {split_name}"):
            rendered = self.render_font_dataset(font_info, sequences, save_images)
            
            # Only include if at least 80% of sequences rendered successfully
            if len(rendered['sequences']) >= len(sequences) * 0.8:
                rendered_fonts.append(rendered)
            else:
                logger.debug(f"Skipping {font_info['name']} - too few sequences rendered")
        
        # Save cache
        cache_file = self.output_dir / f"{split_name}_dataset.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(rendered_fonts, f)
        
        logger.info(f"Saved {len(rendered_fonts)} fonts to {cache_file}")
        
        return rendered_fonts
    
    def load_cached_dataset(self, split_name: str) -> Optional[List[Dict]]:
        """Load pre-rendered dataset from cache"""
        cache_file = self.output_dir / f"{split_name}_dataset.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        return None
    
    def compute_kerning_metrics(self,
                                rendered_data: Dict,
                                bigram: str) -> Dict:
        """
        Compute approximate kerning metrics for a bigram
        
        For mech interp: compare spacing in bigram vs individual chars
        """
        if bigram not in rendered_data['sequences']:
            return {}
        
        char1, char2 = bigram[0], bigram[1]
        if char1 not in rendered_data['sequences'] or char2 not in rendered_data['sequences']:
            return {}
        
        bigram_img = rendered_data['sequences'][bigram]
        char1_img = rendered_data['sequences'][char1]
        char2_img = rendered_data['sequences'][char2]
        
        # Compute horizontal center of mass for each
        def center_of_mass(img):
            y, x = np.where(img > 0.5)
            if len(x) == 0:
                return self.image_size // 2
            return int(np.mean(x))
        
        bigram_com = center_of_mass(bigram_img)
        char1_com = center_of_mass(char1_img)
        char2_com = center_of_mass(char2_img)
        
        # Estimate if kerning is present (bigram is more compact than expected)
        expected_width = abs(char2_com - char1_com)
        
        # Compute actual extent in bigram
        bigram_extent = np.where(bigram_img.sum(axis=0) > 0.1)[0]
        if len(bigram_extent) > 0:
            actual_width = bigram_extent[-1] - bigram_extent[0]
        else:
            actual_width = expected_width
        
        kerning_amount = expected_width - actual_width
        
        return {
            'has_kerning': kerning_amount > 2,  # Pixels
            'kerning_amount': int(kerning_amount),
            'bigram_width': int(actual_width),
        }


if __name__ == "__main__":
    from font_metadata import FontMetadataLoader
    
    # Load fonts
    loader = FontMetadataLoader()
    fonts = loader.load_all_fonts(min_fonts=10, max_fonts=10)
    
    if fonts:
        # Create renderer
        renderer = SequenceRenderer(image_size=128, font_size=48)
        
        # Generate sequences
        characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        bigrams = ["Th", "he", "in", "AV", "WA", "To", "We"]
        trigrams = ["The", "and", "WAV"]
        
        sequences = renderer.generate_sequences(
            characters,
            max_length=3,
            common_bigrams=bigrams,
            common_trigrams=trigrams
        )
        
        print(f"\nRendering {len(sequences)} sequences:")
        print(f"  Single chars: {sum(1 for s in sequences if len(s) == 1)}")
        print(f"  Bigrams: {sum(1 for s in sequences if len(s) == 2)}")
        print(f"  Trigrams: {sum(1 for s in sequences if len(s) == 3)}")
        
        # Test render one font
        rendered = renderer.render_font_dataset(fonts[0], sequences[:10], save_images=True)
        print(f"\nRendered {len(rendered['sequences'])} sequences for {fonts[0]['name']}")
