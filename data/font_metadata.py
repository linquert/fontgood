"""
Font Metadata Loader
Loads 500-2000+ fonts from Google Fonts with comprehensive metadata
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import logging
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FontMetadataLoader:
    """Load and parse comprehensive font metadata from Google Fonts"""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fonts_repo_path = self.cache_dir / "fonts"
        
    def clone_google_fonts(self) -> bool:
        """Clone or update Google Fonts repository"""
        if self.fonts_repo_path.exists():
            logger.info("Google Fonts repo exists, updating...")
            try:
                subprocess.run(
                    ["git", "pull"],
                    cwd=self.fonts_repo_path,
                    check=True,
                    capture_output=True,
                    timeout=60
                )
                return True
            except:
                logger.warning("Update failed, using cached version")
                return True
        else:
            logger.info("Cloning Google Fonts repository (this may take a minute)...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1",
                     "https://github.com/google/fonts.git",
                     str(self.fonts_repo_path)],
                    check=True,
                    capture_output=True,
                    timeout=300
                )
                return True
            except Exception as e:
                logger.error(f"Failed to clone repository: {e}")
                return False
    
    def parse_metadata(self, font_path: Path) -> Optional[Dict]:
        """Parse METADATA.pb file with comprehensive attributes"""
        metadata_file = font_path / "METADATA.pb"
        
        if not metadata_file.exists():
            return None
        
        metadata = {
            'name': font_path.name,
            'category': 'SANS_SERIF',
            'weights': [],
            'styles': [],
        }
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Parse category
                category_match = re.search(r'category:\s*"([^"]+)"', content)
                if category_match:
                    metadata['category'] = category_match.group(1)
                
                # Parse all font entries
                font_blocks = re.findall(r'fonts \{[^}]*\}', content, re.DOTALL)
                
                for block in font_blocks:
                    # Extract weight
                    weight_match = re.search(r'weight:\s*(\d+)', block)
                    if weight_match:
                        weight = int(weight_match.group(1))
                        if weight not in metadata['weights']:
                            metadata['weights'].append(weight)
                    
                    # Extract style
                    style_match = re.search(r'style:\s*"([^"]+)"', block)
                    if style_match:
                        style = style_match.group(1)
                        if style not in metadata['styles']:
                            metadata['styles'].append(style)
                
                # Sort weights
                metadata['weights'].sort()
                
        except Exception as e:
            logger.debug(f"Parse error for {font_path.name}: {e}")
            return None
        
        return metadata
    
    def extract_font_attributes(self, metadata: Dict, font_name: str) -> Dict:
        """
        Extract 8-dimensional font attributes for auxiliary supervision
        
        Attributes (all normalized to ~[0, 1]):
        1. serif_score: 0=sans, 0.5=slab, 1=serif
        2. weight: 0=thin(100), 0.5=regular(400), 1=black(900)
        3. width: 0=condensed, 0.5=normal, 1=extended
        4. slant: 0=upright, 1=italic
        5. contrast: 0=low (sans), 1=high (didone)
        6. x_height: 0=low, 1=high (estimated)
        7. stroke_ending: 0=blunt, 1=rounded
        8. formality: 0=casual/display, 1=formal/text
        """
        attr = {
            'serif_score': 0.0,
            'weight': 0.5,
            'width': 0.5,
            'slant': 0.0,
            'contrast': 0.3,
            'x_height': 0.5,
            'stroke_ending': 0.5,
            'formality': 0.5,
        }
        
        category = metadata['category']
        name_lower = font_name.lower()
        
        # 1. Serif score
        if category == 'SERIF':
            # Check if it's slab serif
            if any(x in name_lower for x in ['slab', 'rockwell', 'courier', 'typewriter']):
                attr['serif_score'] = 0.5
            else:
                attr['serif_score'] = 1.0
        elif category == 'SANS_SERIF':
            attr['serif_score'] = 0.0
        elif category == 'DISPLAY':
            attr['serif_score'] = 0.3  # Display fonts vary
        elif category == 'HANDWRITING':
            attr['serif_score'] = 0.2
        
        # 2. Weight (from metadata, use median if multiple)
        weights = metadata.get('weights', [400])
        if weights:
            median_weight = sorted(weights)[len(weights) // 2]
            attr['weight'] = (median_weight - 100) / 800
            attr['weight'] = max(0.0, min(1.0, attr['weight']))
        
        # 3. Width (from name heuristics)
        if any(x in name_lower for x in ['condensed', 'narrow', 'compressed', 'compact']):
            attr['width'] = 0.2
        elif any(x in name_lower for x in ['extended', 'expanded', 'wide', 'broad']):
            attr['width'] = 0.8
        
        # 4. Slant
        styles = metadata.get('styles', [])
        if 'italic' in styles or 'Italic' in str(styles):
            attr['slant'] = 1.0
        elif any(x in name_lower for x in ['italic', 'oblique', 'slanted']):
            attr['slant'] = 1.0
        
        # 5. Contrast (stroke thickness variation)
        if category == 'SERIF':
            # Traditional serifs have high contrast
            if any(x in name_lower for x in ['didot', 'bodoni', 'modern']):
                attr['contrast'] = 1.0
            else:
                attr['contrast'] = 0.7
        elif category == 'SANS_SERIF':
            # Sans serifs typically low contrast
            attr['contrast'] = 0.2
        elif any(x in name_lower for x in ['mono', 'typewriter', 'courier']):
            attr['contrast'] = 0.1  # Monospace very uniform
        
        # 6. X-height (ratio of lowercase to uppercase)
        # Estimate from name patterns
        if any(x in name_lower for x in ['grotesque', 'grotesk', 'news']):
            attr['x_height'] = 0.6  # Typically large x-height
        elif any(x in name_lower for x in ['old', 'venetian', 'renaissance']):
            attr['x_height'] = 0.4  # Smaller x-height
        
        # 7. Stroke ending
        if category == 'SANS_SERIF':
            if any(x in name_lower for x in ['round', 'circular']):
                attr['stroke_ending'] = 1.0
            else:
                attr['stroke_ending'] = 0.3  # Most sans are blunt
        elif category == 'SERIF':
            attr['stroke_ending'] = 0.6  # Serifs vary
        
        # 8. Formality
        if category == 'DISPLAY' or category == 'HANDWRITING':
            attr['formality'] = 0.2
        elif any(x in name_lower for x in ['casual', 'comic', 'marker', 'brush']):
            attr['formality'] = 0.1
        elif any(x in name_lower for x in ['text', 'book', 'times', 'garamond']):
            attr['formality'] = 0.9
        elif category == 'SERIF':
            attr['formality'] = 0.7
        elif category == 'SANS_SERIF':
            attr['formality'] = 0.6
        
        return attr
    
    def get_font_files(self, font_path: Path) -> List[Path]:
        """Get .ttf files, preferring Regular weight"""
        all_files = list(font_path.glob("*.ttf"))
        
        if not all_files:
            return []
        
        # Prefer Regular or normal weight
        for f in all_files:
            fname = f.name.lower()
            if 'regular' in fname or '-regular' in fname:
                return [f]
        
        # Fallback to first file
        return [all_files[0]]
    
    def load_all_fonts(self, 
                      min_fonts: int = 200,
                      max_fonts: Optional[int] = None) -> List[Dict]:
        """
        Load metadata for all available fonts
        
        Args:
            min_fonts: Minimum fonts required
            max_fonts: Maximum fonts to load (None = all)
        
        Returns:
            List of font dictionaries
        """
        if not self.clone_google_fonts():
            logger.error("Failed to access Google Fonts")
            return []
        
        fonts = []
        
        # Scan font directories
        for license_dir in ['ofl', 'apache', 'ufl']:
            license_path = self.fonts_repo_path / license_dir
            if not license_path.exists():
                continue
            
            for font_dir in sorted(license_path.iterdir()):
                if not font_dir.is_dir():
                    continue
                
                # Get font files
                font_files = self.get_font_files(font_dir)
                if not font_files:
                    continue
                
                # Parse metadata
                metadata = self.parse_metadata(font_dir)
                if metadata is None:
                    metadata = {
                        'name': font_dir.name,
                        'category': 'SANS_SERIF',
                        'weights': [400],
                        'styles': ['normal']
                    }
                
                # Extract attributes
                attributes = self.extract_font_attributes(metadata, font_dir.name)
                
                fonts.append({
                    'name': font_dir.name,
                    'path': font_dir,
                    'files': font_files,
                    'metadata': metadata,
                    'attributes': attributes,
                })
                
                if max_fonts and len(fonts) >= max_fonts:
                    break
            
            if max_fonts and len(fonts) >= max_fonts:
                break
        
        logger.info(f"Loaded {len(fonts)} fonts")
        
        if len(fonts) < min_fonts:
            logger.warning(f"Only found {len(fonts)} fonts (minimum {min_fonts})")
        
        return fonts
    
    def create_splits(self, 
                     fonts: List[Dict],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     continual_ratio: float = 0.1) -> Dict[str, List[Dict]]:
        """
        Create balanced splits by category
        """
        import random
        random.seed(42)
        
        # Group by category
        by_category = defaultdict(list)
        for font in fonts:
            category = font['metadata']['category']
            by_category[category].append(font)
        
        splits = {
            'train': [],
            'val': [],
            'test': [],
            'continual': []
        }
        
        # Split each category proportionally
        for category, cat_fonts in by_category.items():
            random.shuffle(cat_fonts)
            n = len(cat_fonts)
            
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            n_test = int(n * test_ratio)
            
            splits['train'].extend(cat_fonts[:n_train])
            splits['val'].extend(cat_fonts[n_train:n_train+n_val])
            splits['test'].extend(cat_fonts[n_train+n_val:n_train+n_val+n_test])
            splits['continual'].extend(cat_fonts[n_train+n_val+n_test:])
        
        logger.info(f"Splits - Train: {len(splits['train'])}, Val: {len(splits['val'])}, "
                   f"Test: {len(splits['test'])}, Continual: {len(splits['continual'])}")
        
        return splits


if __name__ == "__main__":
    loader = FontMetadataLoader()
    fonts = loader.load_all_fonts(min_fonts=200, max_fonts=500)
    
    print(f"\nLoaded {len(fonts)} fonts")
    print("\nCategory distribution:")
    
    from collections import Counter
    categories = Counter(f['metadata']['category'] for f in fonts)
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")
    
    print("\nExample fonts:")
    for i, font in enumerate(fonts[:5]):
        print(f"\n{i+1}. {font['name']}")
        print(f"   Category: {font['metadata']['category']}")
        print(f"   Weights: {font['metadata']['weights']}")
        print(f"   Attributes: {font['attributes']}")
