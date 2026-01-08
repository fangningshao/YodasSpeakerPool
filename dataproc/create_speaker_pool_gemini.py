# Copyright (c) 2026 (authors: Fangning Shao)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Create Speaker Pool with Gemini Audio Understanding.

This script creates an isolated speaker pool by:
1. Sampling one speech per speaker with proper duration (4-15s)
2. Copying the speech wav into a speaker pool folder
3. Using Gemini 2.5 Flash to analyze audio characteristics
4. Storing comprehensive speaker metadata in JSON format

Example Usage:
    python create_speaker_pool_gemini.py --input_dir "Emilia_Yodas/ZH-*" --output_dir data/speaker_pool --min_duration 4.0 --max_duration 15.0
"""

import os
import sys
import json
import codecs
import shutil
import argparse
import ssl
import urllib3
import glob
from pathlib import Path
from collections import defaultdict
import random
import torchaudio
import google.generativeai as genai
from tqdm import tqdm

# Disable SSL warnings and verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# System prompt for Gemini audio understanding
AUDIO_UNDERSTANDING_PROMPT = """You are an expert audio analyst. Please listen to the provided audio file carefully and analyze its characteristics in detail.

You should output a JSON object with the following fields:

1. "asr": The transcribed text from the audio (what the speaker is saying)
2. "noise_level": Integer 0-3 (0 = clear recording without noise, 1 = slight background noise, 2 = moderate noise, 3 = loud noise)
3. "naturalness": Integer 0-3 (0 = mechanical/robotic voice, 1 = somewhat natural, 2 = natural, 3 = very natural human voice)
4. "language": Language code (e.g., "en-US", "zh-CN", "zh-TW", "ja-JP", "fr-FR", etc.)
5. "dialect": Specific dialect or accent (e.g., "American English", "British English", "普通话", "粤语", "上海话", "四川话", "天津话", "台湾口音", "其他南方口音", "其他北方口音", etc.)
6. "style": Speaking style (e.g., "chatting", "news", "podcast", "storytelling", "lecture", "presentation", "reading", "singing", etc.)
7. "emotion": Primary emotion (e.g., "neutral", "happy", "sad", "angry", "surprised", "fearful", "excited", "calm", "passionate", etc.)
8. "maybe_ai": indicating if the voice sounds like AI-generated or synthetic voice (1 = possibly AI/synthetic, or remind you of an AI assistant; 0 = certainly human). Note that many high-quality synthetic voices can sound very natural, so use your best judgment.
8. "speaker_description": A detailed 200-500 word description in English covering:
   - Speaker's gender and estimated age range
   - Voice characteristics (pitch, tone, timber, pace, clarity)
   - Speaking style and emotion
   - Language and dialect details
   - Audio quality (noise level, reverb, recording quality)
   - Guessed profession or background based on voice
   - What the speaker reminds you of (celebrities, character types, etc.)
   - Topics or contexts this voice would be suitable for
   - Any unique or distinguishing vocal features

**IMPORTANT**: Output ONLY a valid JSON object, no additional text or explanation.

Here are two examples of the expected output format:

**Example 1 (Professional studio recording):**
{
  "asr": "Hello everyone, welcome to today's podcast. We're going to discuss some fascinating topics about artificial intelligence and its impact on our daily lives.",
  "noise_level": 0,
  "naturalness": 3,
  "language": "en-US",
  "dialect": "American English",
  "style": "podcast",
  "emotion": "enthusiastic",
  "maybe_ai": 0,
  "speaker_description": "This is a male speaker, approximately 30-40 years old, with a warm and engaging voice. His vocal tone is medium-pitched with excellent clarity and a professional broadcast quality. The speaker demonstrates natural conversational flow with well-modulated pacing, neither too fast nor too slow. His American English accent is clear and neutral, suggesting someone from the West Coast or a media professional trained in standard broadcast English. The recording quality is exceptional with no background noise or reverb, indicating a professional studio setup. The speaker's enthusiastic and welcoming tone suggests experience in public speaking or media work, possibly a podcast host, radio presenter, or educator. His voice has a friendly, approachable quality that reminds one of popular tech podcast hosts like Lex Fridman or professional audiobook narrators. The timber is smooth and consistent throughout, with good breath control and natural inflections that keep the listener engaged. This voice would be ideal for educational content, technology podcasts, audiobooks, professional presentations, or any content requiring a trustworthy and engaging male narrator. The speaker seems comfortable discussing intellectual topics and would likely excel at explaining complex subjects in an accessible way. His voice conveys confidence without arrogance, making him suitable for thought leadership content, interviews, or documentary narration."
}

**Example 2 (Phone recording with background noise):**
{
  "asr": "我昨天去了那个新开的火锅店，巴适的板！服务员态度也特别好，下次我们一起去吧。",
  "noise_level": 2,
  "naturalness": 3,
  "language": "zh-CN",
  "dialect": "四川话",
  "style": "chatting",
  "emotion": "excited",
  "maybe_ai": 0,
  "speaker_description": "This is a young female speaker, approximately 20-28 years old, with an animated and expressive voice. Her pitch is naturally high with a bright, energetic quality that conveys genuine excitement. She speaks with a noticeable Sichuan dialect, featuring characteristic tonal variations and local pronunciations that add authenticity and regional flavor to her speech. The recording appears to be captured on a mobile phone, with moderate background noise including distant traffic sounds and slight environmental echo, suggesting an outdoor or casual indoor setting. Despite the imperfect recording conditions, her voice cuts through clearly with natural human warmth. Her speaking pace is rapid and enthusiastic, typical of casual conversation among friends, with natural pauses and informal speech patterns. The speaker's energetic delivery and colloquial language suggest she is likely a college student or young professional discussing everyday life topics. Her voice has a youthful, bubbly quality reminiscent of lifestyle vloggers or social media content creators who focus on food, travel, and daily experiences. The natural expressiveness and genuine emotion in her voice make it particularly suitable for conversational content, casual vlogs, friend-to-friend recommendations, social media stories, or any content targeting young Chinese audiences interested in lifestyle, food culture, or regional experiences. Her authentic Sichuan accent would resonate strongly with audiences from southwestern China and adds cultural authenticity to content about regional cuisine, travel, or local culture. The slight imperfections in recording quality actually enhance the relatable, authentic feel of the audio, making it perfect for user-generated content, testimonials, or casual conversational scenarios where professional polish might feel too distant or formal."
}

Now, please analyze the following audio file and provide your analysis in the exact JSON format shown above:"""


class SpeakerPoolCreator:
    """Creates a curated speaker pool with Gemini audio understanding."""
    
    def __init__(self, input_dir, output_dir, min_duration=5.0, max_duration=20.0, gemini_api_key=None):
        """
        Args:
            input_dir: Directory containing training data with JSON and WAV files
            output_dir: Output directory for speaker pool
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            gemini_api_key: Gemini API key (if None, reads from gemini-key.txt)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wavs_dir = self.output_dir / "wavs"
        self.wavs_dir.mkdir(exist_ok=True)
        self.json_dir = self.output_dir / "metadata"
        self.json_dir.mkdir(exist_ok=True)
        
        # Initialize Gemini
        if gemini_api_key is None:
            try:
                with open("gemini-key.txt", "r") as f:
                    gemini_api_key = f.read().strip()
            except FileNotFoundError:
                print("Error: gemini-key.txt not found. Please create this file with your Gemini API key.")
                sys.exit(1)
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        
        print(f"Initialized SpeakerPoolCreator:")
        print(f"  Input: {self.input_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"  Duration range: {self.min_duration}s - {self.max_duration}s")
    
    def get_audio_duration(self, wav_path):
        """Get audio duration in seconds."""
        try:
            info = torchaudio.info(str(wav_path))
            duration = info.num_frames / info.sample_rate
            return duration
        except Exception as e:
            print(f"Error reading {wav_path}: {e}")
            return None
    
    def collect_speakers(self):
        """Collect all speakers and their audio files from input directory (with glob support)."""
        print("\n" + "="*80)
        print("Step 1: Collecting speakers and audio files")
        print("="*80)
        
        # Expand glob patterns if present
        input_dirs = []
        input_pattern = str(self.input_dir)
        
        if '*' in input_pattern or '?' in input_pattern:
            # Expand glob pattern
            matched_dirs = glob.glob(input_pattern)
            if matched_dirs:
                print(f"Glob pattern '{input_pattern}' matched {len(matched_dirs)} directories")
                input_dirs = [Path(d) for d in matched_dirs]
            else:
                print(f"WARNING: Glob pattern '{input_pattern}' matched no directories")
                input_dirs = []
        else:
            # Direct path
            input_dirs = [self.input_dir]
        
        if not input_dirs:
            print("ERROR: No input directories found!")
            return {}
        
        speakers = defaultdict(list)
        speaker_seen = set()  # Track which speakers we've already found a valid audio for
        
        total_files = 0
        skipped_duplicates = 0
        skipped_no_wav = 0
        skipped_bad_duration = 0
        added_count = 0
        
        # Process each directory
        for input_dir in input_dirs:
            print(f"\nProcessing directory: {input_dir}")
            
            # Find all JSON files in this directory
            json_files = list(input_dir.rglob("*.json"))
            print(f"  Found {len(json_files)} JSON files")
            total_files += len(json_files)
            
            for json_file in tqdm(json_files, desc=f"  Scanning {input_dir.name}", leave=False):
                try:
                    # Extract speaker ID from filename FIRST (before loading metadata)
                    # Expected format: <dataset>_<speaker_id>_<utterance>.json
                    # e.g., "EN-B000001_S00123_xyz_U00456.json" -> speaker_id = "S00123_xyz"
                    filename = json_file.stem
                    
                    speaker_id = filename
                    parts = filename.split('_', 1)
                    if len(parts) == 2:
                        speaker_id = parts[1].rsplit('_', 1)[0]
                    
                    # Skip if we already have a valid audio for this speaker - NO FILE I/O!
                    if speaker_id in speaker_seen:
                        skipped_duplicates += 1
                        continue
                    
                    # Find corresponding WAV file (check before loading JSON)
                    file_extension = '.wav'
                    wav_file = json_file.with_suffix('.wav')
                    if not wav_file.exists():
                        # Also try .mp3 extension
                        wav_file = json_file.with_suffix('.mp3')
                        if not wav_file.exists():
                            skipped_no_wav += 1
                            continue
                        else:
                            file_extension = '.mp3'
                    
                    # Check duration BEFORE loading JSON metadata
                    duration = self.get_audio_duration(wav_file)
                    if duration is None:
                        skipped_bad_duration += 1
                        continue
                    
                    # Check if duration is valid
                    if not (self.min_duration <= duration <= self.max_duration):
                        skipped_bad_duration += 1
                        continue
                    
                    # Only NOW load JSON metadata (we found a valid audio!)
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Store this as THE candidate for this speaker (first valid one wins)
                    speakers[speaker_id] = [{
                        'json_path': json_file,
                        'wav_path': wav_file,
                        'duration': duration,
                        'data': data
                    }]
                    
                    # Mark speaker as seen - skip all future files from this speaker
                    speaker_seen.add(speaker_id)
                    added_count += 1
                
                except Exception as e:
                    print(f"\nError processing {json_file}: {e}")
                    continue
        
        print(f"\n{'='*80}")
        print(f"Collection complete:")
        print(f"  Total JSON files scanned: {total_files}")
        print(f"  Unique speakers found: {len(speakers)}")
        print(f"  Added to pool: {added_count}")
        print(f"  Skipped (duplicate speaker): {skipped_duplicates}")
        print(f"  Skipped (no WAV file): {skipped_no_wav}")
        print(f"  Skipped (bad duration): {skipped_bad_duration}")
        print(f"{'='*80}")
        
        return speakers
    
    def sample_one_per_speaker(self, speakers):
        """Sample one audio file per speaker with proper duration."""
        print("\n" + "="*80)
        print("Step 2: Sampling one audio per speaker")
        print("="*80)
        
        sampled = {}
        
        for speaker_id, candidates in tqdm(speakers.items(), desc="Sampling speakers"):
            # Filter by duration
            valid_candidates = [
                c for c in candidates 
                if self.min_duration <= c['duration'] <= self.max_duration
            ]
            
            if not valid_candidates:
                # No valid duration, skip
                continue
            
            # Randomly sample one
            selected = random.choice(valid_candidates)
            sampled[speaker_id] = selected
        
        print(f"\nSampled {len(sampled)} speakers with valid duration")
        return sampled
    
    def copy_all_wavs(self, sampled_speakers):
        """Copy all WAV files first before calling Gemini (Step 2.5)."""
        print("\n" + "="*80)
        print("Step 2.5: Copying all WAV files")
        print("="*80)
        
        copied_count = 0
        skipped_count = 0
        
        for speaker_id, data in tqdm(sampled_speakers.items(), desc="Copying WAV files"):
            output_wav = self.wavs_dir / f"{speaker_id}.wav"
            if data['wav_path'].suffix.lower() == '.mp3':
                output_wav = self.wavs_dir / f"{speaker_id}.mp3"
            
            # Skip if WAV already exists (resume support)
            if output_wav.exists():
                skipped_count += 1
                continue
            
            try:
                shutil.copy2(data['wav_path'], output_wav)
                copied_count += 1
            except Exception as e:
                print(f"\nError copying {data['wav_path']} to {output_wav}: {e}")
                continue
        
        print(f"\nWAV copying complete:")
        print(f"  Copied: {copied_count}")
        print(f"  Skipped (already exists): {skipped_count}")
        print(f"  Total: {len(sampled_speakers)}")
        
        return copied_count, skipped_count
    
    def analyze_with_gemini(self, wav_path):
        """Analyze audio with Gemini and return structured results."""
        try:
            print(f"  Uploading: {wav_path.name}")
            audio_file = genai.upload_file(path=str(wav_path))
            
            print(f"  Analyzing...")
            response = self.model.generate_content([AUDIO_UNDERSTANDING_PROMPT, audio_file])
            
            if response and response.text:
                # Parse JSON from response
                response_text = response.text.strip()
                response_text =  response_text.replace('\",\n}', '\"\n}')  # Fix common JSON issue returned by Gemini

                # Remove markdown code blocks if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if '```' in response_text:
                    response_text = response_text.split('```')[0]
                
                response_text = response_text.strip()
                
                # Parse JSON
                try:
                    gemini_result = json.loads(response_text)
                except json.JSONDecodeError as e:
                    print("  JSON parse error in Gemini response:", response_text)
                    return {"error": "Gemini response failure"}
                return gemini_result
            else:
                return {"error": "Gemini response failure"}
        
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            print(f"  Response text: {response.text[:200]}...")
            return {"error": f"JSON parse error: {str(e)}", "raw_response": response.text}
        
        except Exception as e:
            print(f"  Error: {e}")
            return {"error": str(e)}
    
    def process_speakers(self, sampled_speakers):
        """Process all sampled speakers with Gemini analysis (with resume support)."""
        print("\n" + "="*80)
        print("Step 3: Processing speakers with Gemini audio understanding")
        print("="*80)
        
        results = []
        failed = []
        skipped_count = 0
        processed_count = 0
        
        # Open TSV file for logging speaker descriptions
        descriptions_tsv = self.output_dir / "speaker_descriptions.tsv"
        tsv_mode = 'a' if descriptions_tsv.exists() else 'w'
        
        with open(descriptions_tsv, tsv_mode, encoding='utf-8') as tsv_file:
            # Write header if new file
            if tsv_mode == 'w':
                tsv_file.write("speaker_id\tmaybe_ai\tspeaker_description\n")
            
            for speaker_id, data in tqdm(list(sampled_speakers.items()), desc="Analyzing speakers"):
                output_json = self.json_dir / f"{speaker_id}.json"
                output_wav = self.wavs_dir / f"{speaker_id}.wav"
                
                # Resume support: Skip if JSON already exists
                if output_json.exists():
                    skipped_count += 1
                    
                    # Load existing result for summary
                    try:
                        with open(output_json, 'r', encoding='utf-8') as f:
                            existing_json = json.load(f)
                            gemini_result = existing_json.get('gemini_res', {})
                            
                            if 'error' not in gemini_result:
                                results.append({
                                    'speaker_id': speaker_id,
                                    'wav_path': output_wav,
                                    'json_path': output_json,
                                    'gemini_res': gemini_result
                                })
                            else:
                                failed.append({
                                    'speaker_id': speaker_id,
                                    'error': gemini_result.get('error')
                                })
                    except Exception as e:
                        print(f"\nError loading existing JSON for {speaker_id}: {e}")
                    
                    continue
                
                print(f"\nProcessing speaker: {speaker_id}")
                
                # Copy WAV file (one at a time, right before processing)
                if not output_wav.exists():
                    try:
                        print(f"  Copying WAV: {data['wav_path'].name}")
                        shutil.copy2(data['wav_path'], output_wav)
                    except Exception as e:
                        print(f"  Error copying WAV: {e}")
                        failed.append({
                            'speaker_id': speaker_id,
                            'error': f"Failed to copy WAV: {str(e)}"
                        })
                        continue
                
                # Analyze with Gemini
                gemini_result = self.analyze_with_gemini(output_wav)
                
                # Create final JSON with original data + Gemini results
                final_json = data['data'].copy()
                final_json['speaker_id'] = speaker_id
                final_json['duration'] = data['duration']
                final_json['original_wav_path'] = str(data['wav_path'])
                final_json['pool_wav_path'] = str(output_wav)
                final_json['gemini_res'] = gemini_result
                
                # Save JSON
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(final_json, f, ensure_ascii=False, indent=2)
                
                print(f"  Saved JSON: {output_json.name}")
                processed_count += 1
                
                # Log to TSV if successful
                if 'error' not in gemini_result:
                    speaker_desc = gemini_result.get('speaker_description', '')
                    # Escape tabs and newlines in description
                    speaker_desc_escaped = speaker_desc.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                    maybe_ai = gemini_result.get('maybe_ai', -1)
                    tsv_file.write(f"{speaker_id}\t{maybe_ai}\t{speaker_desc_escaped}\n")
                    tsv_file.flush()
                    
                    results.append({
                        'speaker_id': speaker_id,
                        'wav_path': output_wav,
                        'json_path': output_json,
                        'gemini_res': gemini_result
                    })
                else:
                    failed.append({
                        'speaker_id': speaker_id,
                        'error': gemini_result.get('error')
                    })
        
        print(f"\n{'='*80}")
        print(f"Processing complete:")
        print(f"  Processed: {processed_count}")
        print(f"  Skipped (already done): {skipped_count}")
        print(f"  Success: {len(results)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Descriptions saved to: {descriptions_tsv}")
        print(f"{'='*80}")
        
        return results, failed
   
    def create_summary(self, results, failed):
        """Create summary statistics and save to file."""
        print("\n" + "="*80)
        print("Step 4: Creating summary")
        print("="*80)
        
        summary = {
            'total_speakers': len(results) + len(failed),
            'successful': len(results),
            'failed': len(failed),
            'statistics': {
                'languages': defaultdict(int),
                'dialects': defaultdict(int),
                'styles': defaultdict(int),
                'emotions': defaultdict(int),
                'noise_levels': defaultdict(int),
                'naturalness': defaultdict(int)
            },
            'failed_speakers': failed
        }
        
        # Collect statistics
        for result in results:
            gemini_res = result['gemini_res']
            summary['statistics']['languages'][gemini_res.get('language', 'unknown')] += 1
            summary['statistics']['dialects'][gemini_res.get('dialect', 'unknown')] += 1
            summary['statistics']['styles'][gemini_res.get('style', 'unknown')] += 1
            summary['statistics']['emotions'][gemini_res.get('emotion', 'unknown')] += 1
            summary['statistics']['noise_levels'][str(gemini_res.get('noise_level', -1))] += 1
            summary['statistics']['naturalness'][str(gemini_res.get('naturalness', -1))] += 1
        
        # Convert defaultdict to dict for JSON serialization
        summary['statistics'] = {k: dict(v) for k, v in summary['statistics'].items()}
        
        # Save summary
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Summary saved to: {summary_file}")
        print(f"\nStatistics:")
        print(f"  Total speakers: {summary['total_speakers']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"\nLanguage distribution:")
        for lang, count in sorted(summary['statistics']['languages'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {lang}: {count}")
        
        return summary
    
    def run(self):
        """Run the complete speaker pool creation pipeline."""
        print("="*80)
        print("SPEAKER POOL CREATOR WITH GEMINI AUDIO UNDERSTANDING")
        print("="*80)
        
        # Step 1: Collect speakers
        speakers = self.collect_speakers()
        
        # Step 2: Sample one per speaker
        sampled = self.sample_one_per_speaker(speakers)
        
        # Step 3: Process with Gemini (WAVs copied one-by-one during processing)
        results, failed = self.process_speakers(sampled)
        
        # Step 4: Create summary
        summary = self.create_summary(results, failed)
        
        print("\n" + "="*80)
        print("SPEAKER POOL CREATION COMPLETE!")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"  - WAV files: {self.wavs_dir}")
        print(f"  - Metadata: {self.json_dir}")
        print(f"  - Summary: {self.output_dir / 'summary.json'}")
        print("="*80)
        
        return results, failed, summary


def main():
    parser = argparse.ArgumentParser(
        description='Create speaker pool with Gemini audio understanding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python create_speaker_pool_gemini.py --input_dir data/train --output_dir data/speaker_pool
  
  # Custom duration range
  python create_speaker_pool_gemini.py --input_dir data/train --output_dir data/speaker_pool --min_duration 5.0 --max_duration 12.0
  
  # Specify Gemini API key
  python create_speaker_pool_gemini.py --input_dir data/train --output_dir data/speaker_pool --api_key YOUR_API_KEY
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing JSON and WAV/MP3 files, such as Emilia-YODAS/ZH-*')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for speaker pool')
    parser.add_argument('--min_duration', type=float, default=4.0,
                       help='Minimum audio duration in seconds (default: 4.0)')
    parser.add_argument('--max_duration', type=float, default=15.0,
                       help='Maximum audio duration in seconds (default: 15.0)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='Gemini API key (if not provided, reads from gemini-key.txt)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create speaker pool
    creator = SpeakerPoolCreator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        gemini_api_key=args.api_key
    )
    
    results, failed, summary = creator.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

