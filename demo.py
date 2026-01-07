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

"""A Gradio-based UI for exploring and annotating a speaker pool dataset."""

import gradio as gr
import json
import pandas as pd
from pathlib import Path
import random
import os
import glob
import shutil

# --- Configuration ---
DEFAULT_DATA_DIR = "data/emilia_yodas_speaker_pool"
MARKED_FILE = "speaker_marks.json"

class SpeakerPoolExplorer:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.df = pd.DataFrame()
        self.marks = self.load_marks()
        self.load_data()

    def load_marks(self):
        start_path = Path(MARKED_FILE)
        if start_path.exists():
            try:
                with open(start_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_marks(self):
        with open(MARKED_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.marks, f, indent=2)

    def load_data(self):
        """Scans the directory structure for JSON metadata and WAV files."""
        data_list = []
        
        # Support flexible directory structure: look for metadata folders recursively
        # Expected: root/{lang}/metadata/*.json OR root/metadata/*.json
        json_files = list(self.root_dir.rglob("metadata/*.json"))
        
        print(f"Scanning {self.root_dir}...")
        print(f"Found {len(json_files)} metadata files.")

        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                # Extract Gemini results
                gemini = meta.get('gemini_res', {})
                if 'error' in gemini:
                    continue

                # Locate audio file
                # Strategy: Look in ../wavs/ relative to metadata folder
                wav_dir = json_path.parent.parent / "wavs"
                wav_path = wav_dir / f"{json_path.stem}.wav"
                
                if not wav_path.exists():
                    # Fallback: try finding it relative to root if structure differs
                    # or check if path is absolute in json
                    if 'pool_wav_path' in meta and os.path.exists(meta['pool_wav_path']):
                         wav_path = Path(meta['pool_wav_path'])
                    else:
                        continue # Skip if no audio found

                speaker_id = meta.get('speaker_id', json_path.stem)
                
                # Flatten the dictionary for the DataFrame
                entry = {
                    'speaker_id': speaker_id,
                    'audio_path': str(wav_path),
                    'language_tag': meta.get('language', 'unknown'), # original dataset lang
                    'gemini_lang': gemini.get('language', 'unknown'),
                    'gemini_dialect': gemini.get('dialect', 'unknown'),
                    'gemini_style': gemini.get('style', 'unknown'),
                    'gemini_emotion': gemini.get('emotion', 'unknown'),
                    'gemini_gender': 'unknown', # extracted via heuristic below
                    'noise_level': gemini.get('noise_level', -1),
                    'naturalness': gemini.get('naturalness', -1),
                    'maybe_ai': gemini.get('maybe_ai', 0),
                    'asr': gemini.get('asr', ''),
                    'description': gemini.get('speaker_description', ''),
                    'mark': self.marks.get(speaker_id, 'Unmarked'),
                    'duration': meta.get('duration', 0.0)
                }
                
                # Simple heuristic for gender from description if not explicit fields
                desc_lower = entry['description'].lower()
                if 'female' in desc_lower and 'male' not in desc_lower:
                    entry['gemini_gender'] = 'Female'
                elif ' male' in desc_lower or 'male ' in desc_lower:
                    entry['gemini_gender'] = 'Male'
                
                data_list.append(entry)

            except Exception as e:
                print(f"Error loading {json_path}: {e}")

        self.df = pd.DataFrame(data_list)
        if not self.df.empty:
            # Sort by recent (assuming file creating usually correlates, or just random/scan order)
            # here we just leave it as scan order, effectively "random"
            pass
        
        print(f"Loaded {len(self.df)} valid entries.")
        return f"Loaded {len(self.df)} speakers."

    def get_filter_choices(self, column):
        if self.df.empty:
            return []
        return sorted(list(self.df[column].unique()))

    def filter_data(self, 
                    lang_filter, 
                    dialect_filter, 
                    style_filter, 
                    emotion_filter, 
                    noise_max, 
                    naturalness_min, 
                    ai_filter,
                    mark_filter,
                    search_query,
                    limit=10, 
                    randomize=False):
        
        if self.df.empty:
            return []

        # Start with full dataframe
        mask = pd.Series([True] * len(self.df))

        # Apply filters
        if lang_filter:
            mask &= self.df['gemini_lang'].isin(lang_filter)
        if dialect_filter:
            mask &= self.df['gemini_dialect'].isin(dialect_filter)
        if style_filter:
            mask &= self.df['gemini_style'].isin(style_filter)
        if emotion_filter:
            mask &= self.df['gemini_emotion'].isin(emotion_filter)
        
        mask &= (self.df['noise_level'] <= noise_max)
        mask &= (self.df['naturalness'] >= naturalness_min)
        
        if ai_filter != "All":
            val = 1 if ai_filter == "Likely AI" else 0
            mask &= (self.df['maybe_ai'] == val)

        if mark_filter != "All":
             mask &= (self.df['mark'] == mark_filter)

        if search_query:
            # Search in ASR, Description, and Speaker ID
            q = search_query.lower()
            search_mask = (
                self.df['asr'].str.lower().str.contains(q) | 
                self.df['description'].str.lower().str.contains(q) | 
                self.df['speaker_id'].str.lower().str.contains(q)
            )
            mask &= search_mask

        filtered_df = self.df[mask]

        if filtered_df.empty:
            return []

        # Randomize or just take head (latest scanned)
        if randomize:
            filtered_df = filtered_df.sample(n=min(len(filtered_df), limit))
        else:
            # Default behavior: show top N
            filtered_df = filtered_df.head(limit)

        return filtered_df.to_dict('records')

    def update_mark(self, speaker_id, mark_status):
        self.marks[speaker_id] = mark_status
        self.save_marks()
        
        # Update dataframe in memory
        if not self.df.empty:
            self.df.loc[self.df['speaker_id'] == speaker_id, 'mark'] = mark_status
        
        return f"Marked {speaker_id} as {mark_status}"

# --- Gradio UI Construction ---

def create_demo():
    explorer = SpeakerPoolExplorer(DEFAULT_DATA_DIR)

    with gr.Blocks(title="Speaker Pool Explorer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ Speaker Pool Explorer")
        
        # State to store current visible data
        current_data_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Filters")
                
                with gr.Group():
                    refresh_btn = gr.Button("🔄 Refresh Data Source", variant="secondary")
                    data_status = gr.Textbox(label="Status", value=f"Loaded {len(explorer.df)} speakers", interactive=False)

                search_box = gr.Textbox(label="Search (ID, Desc, ASR)", placeholder="Type to search...")
                
                with gr.Row():
                    lang_dd = gr.Dropdown(label="Language", choices=explorer.get_filter_choices('gemini_lang'), multiselect=True)
                    dialect_dd = gr.Dropdown(label="Dialect", choices=explorer.get_filter_choices('gemini_dialect'), multiselect=True)
                
                with gr.Row():
                    style_dd = gr.Dropdown(label="Style", choices=explorer.get_filter_choices('gemini_style'), multiselect=True)
                    emotion_dd = gr.Dropdown(label="Emotion", choices=explorer.get_filter_choices('gemini_emotion'), multiselect=True)

                with gr.Row():
                    noise_sld = gr.Slider(0, 3, value=1, step=1, label="Max Noise Level (0=Clean)")
                    nat_sld = gr.Slider(0, 3, value=2, step=1, label="Min Naturalness (3=Best)")
                
                with gr.Row():
                    ai_radio = gr.Radio(["All", "Human Only", "Likely AI"], value="Human Only", label="AI Detection")
                    mark_radio = gr.Radio(["All", "Unmarked", "Good", "Bad"], value="All", label="Mark Filter")

                with gr.Row():
                    apply_btn = gr.Button("🔎 Apply Filters", variant="primary")
                    random_btn = gr.Button("🎲 Random 10", variant="secondary")

            with gr.Column(scale=3):
                gr.Markdown("### 🎧 Speaker List")
                gallery = gr.Group() # Container for dynamically generated rows
                
                # We interpret "Gallery" as a list of components, but Gradio create_components is static.
                # We will use a Dataset component or HTML/Audio list. 
                # For rich interaction (individual play + mark buttons), a Dataframe is limited.
                # BETTER APPROACH: Use a gr.Dataframe to show the list, and a "Detail View" below/side.
                
                results_table = gr.Dataframe(
                    headers=["Mark", "ID", "Lang", "Dialect", "Style", "Emotion", "ASR"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    label="Filtered Results (Select row to listen)"
                )
                
                gr.Markdown("### 🔊 Player & Details")
                with gr.Group():
                    with gr.Row():
                        audio_player = gr.Audio(label="Audio", type="filepath", autoplay=True)
                        with gr.Column():
                            curr_id_lbl = gr.Label(label="Speaker ID")
                            mark_good_btn = gr.Button("👍 Mark Good", size="sm")
                            mark_bad_btn = gr.Button("👎 Mark Bad", size="sm")
                            mark_status_lbl = gr.Label(label="Current Status")
                    
                    desc_box = gr.Markdown(label="Full Description")
                    asr_box = gr.Textbox(label="Full ASR Text", interactive=False)
                    tech_box = gr.JSON(label="Technical Metadata")

        # --- Event Handlers ---

        def refresh_dataset():
            msg = explorer.load_data()
            # Update choices
            return (
                msg, 
                gr.update(choices=explorer.get_filter_choices('gemini_lang')),
                gr.update(choices=explorer.get_filter_choices('gemini_dialect')),
                gr.update(choices=explorer.get_filter_choices('gemini_style')),
                gr.update(choices=explorer.get_filter_choices('gemini_emotion'))
            )

        def get_table_data(
                lang, dia, sty, emo, noise, nat, ai_flt, mark_flt, search, is_random
            ):
            limit = 10 if is_random else 50
            results = explorer.filter_data(lang, dia, sty, emo, noise, nat, ai_flt, mark_flt, search, limit, is_random)
            
            # Transform for Dataframe
            table_data = []
            for r in results:
                table_data.append([
                    r['mark'], r['speaker_id'], r['gemini_lang'], 
                    r['gemini_dialect'], r['gemini_style'], 
                    r['gemini_emotion'], r['asr']
                ])
            
            return table_data, results # Return raw results to store in State

        def on_select_row(evt: gr.SelectData, state_results):
            # evt.index is [row, col]
            row_idx = evt.index[0]
            if not state_results or row_idx >= len(state_results):
                return None, "", "", "", {}, "Unknown", "Unmarked"
            
            data = state_results[row_idx]
            
            # Format description markdown
            desc_md = f"**Description:** \n\n{data['description']}"
            
            tech_data = {
                "noise": data['noise_level'],
                "naturalness": data['naturalness'],
                "maybe_ai": data['maybe_ai'],
                "duration": data['duration']
            }

            return (
                data['audio_path'], # Audio
                data['speaker_id'], # ID Label
                desc_md,           # Description
                data['asr'],       # ASR
                tech_data,         # Tech JSON
                data['mark']       # Status
            )

        def mark_current(speaker_id, status):
            if not speaker_id or speaker_id == "None":
                return "No speaker selected"
            msg = explorer.update_mark(speaker_id.strip(), status)
            return status # Return new status for the label

        # --- Wiring ---
        
        refresh_btn.click(
            refresh_dataset, 
            inputs=[], 
            outputs=[data_status, lang_dd, dialect_dd, style_dd, emotion_dd]
        )

        filter_inputs = [
            lang_dd, dialect_dd, style_dd, emotion_dd, 
            noise_sld, nat_sld, ai_radio, mark_radio, search_box
        ]

        apply_btn.click(
            fn=lambda *args: get_table_data(*args, False),
            inputs=filter_inputs,
            outputs=[results_table, current_data_state]
        )

        random_btn.click(
            fn=lambda *args: get_table_data(*args, True),
            inputs=filter_inputs,
            outputs=[results_table, current_data_state]
        )
        
        # Select row -> Populate details
        results_table.select(
            on_select_row,
            inputs=[current_data_state],
            outputs=[audio_player, curr_id_lbl, desc_box, asr_box, tech_box, mark_status_lbl]
        )

        # Marking buttons
        mark_good_btn.click(lambda s: mark_current(s, "Good"), inputs=[curr_id_lbl], outputs=[mark_status_lbl])
        mark_bad_btn.click(lambda s: mark_current(s, "Bad"), inputs=[curr_id_lbl], outputs=[mark_status_lbl])
        
        # Trigger initial load (simulated click)
        # demo.load can also be used but click is easier to repurpose
        demo.load(
             fn=lambda: get_table_data([],[],[],[],1,2,"Human Only","All","", True),
             inputs=[],
             outputs=[results_table, current_data_state]
        )

    return demo


if __name__ == "__main__":
    demo_app = create_demo()
    # allow_flagging="never" prevents gradio from creating a log folder for user feedback
    demo_app.launch(server_name="127.0.0.1", server_port=7879, inbrowser=True, allowed_paths=[DEFAULT_DATA_DIR])