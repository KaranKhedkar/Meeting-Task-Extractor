import zipfile
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter

ZIP_PATH = Path(__file__).resolve().parent / "data_raw" / "ami_public_manual_1.6.2.zip"
BASE_DATASET = Path(__file__).resolve().parent / "dataset.json"
OUTPUT_DATASET = Path(__file__).resolve().parent / "dataset_large.json"

DA_ASPECT_MAP = {
    'ami_da_1': 'Backchannel',
    'ami_da_2': 'Stall',
    'ami_da_3': 'Fragment',
    'ami_da_4': 'Inform',
    'ami_da_5': 'Elicit-Inform',
    'ami_da_6': 'Suggest',
    'ami_da_7': 'Offer', # Action commitment
    'ami_da_8': 'Elicit-Offer-Or-Suggestion', # Action request
    'ami_da_9': 'Assess',
    'ami_da_11': 'Elicit-Assessment',
    'ami_da_12': 'Comment-About-Understanding',
    'ami_da_13': 'Elicit-Comment-Understanding',
    'ami_da_14': 'Be-Positive',
    'ami_da_15': 'Be-Negative',
    'ami_da_16': 'Other',
}

COMPLETED_REGEX = re.compile(
    r'\b(already|finished|completed|deployed|fixed|implemented|tested|verified|restored|published)\b',
    re.IGNORECASE
)


PAST_PRONOUNS = re.compile(r'\b(i(\'ve|\s+have|\s+already)?|we(\'ve|\s+have|\s+already)?|the\s+\w+|it\s+was|they)\b', re.IGNORECASE)


def parse_id_range(ch_href: str):
    m = re.search(r'id\(([^)]+)\)(\.\.id\(([^)]+)\))?', ch_href)
    if m:
        start_w = m.group(1)
        end_w = m.group(3) or start_w
        return start_w, end_w
    return None, None


def build_large_corpus(max_meetings=80):
    print(f"Opening AMI archive: {ZIP_PATH}")
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH) as z:
        all_files = set(z.namelist())
        da_files = [n for n in all_files if n.startswith('dialogueActs/') and n.endswith('.dialog-act.xml')]

        meeting_to_da_files = defaultdict(list)
        for f in da_files:
            m_id = f.split('/')[1].split('.')[0]
            meeting_to_da_files[m_id].append(f)

        sorted_meeting_ids = sorted(meeting_to_da_files.keys())[:max_meetings]
        print(f"Selected {len(sorted_meeting_ids)} AMI meetings for compilation.")

        n_train = int(len(sorted_meeting_ids) * 0.75)
        train_m_ids = set(sorted_meeting_ids[:n_train])
        test_m_ids = set(sorted_meeting_ids[n_train:])

        compiled_meetings = []
        overall_stats = Counter()

        for m_id in sorted_meeting_ids:
            split = "train" if m_id in train_m_ids else "test"
            meeting_utterances = []
            files_for_meeting = meeting_to_da_files[m_id]

            for da_file in files_for_meeting:
                parts = da_file.split('/')[1].split('.')
                speaker = parts[1] if len(parts) > 1 else "Unknown"

                words_file = f"words/{m_id}.{speaker}.words.xml"
                if words_file not in all_files:
                    continue

                w_tree = ET.fromstring(z.read(words_file))
                ordered_words = []
                word_id_to_idx = {}
                for elem in w_tree.iter():
                    if elem.tag.endswith('w') and elem.text:
                        wid = elem.attrib.get('{http://nite.sourceforge.net/}id', '')
                        word_id_to_idx[wid] = len(ordered_words)
                        ordered_words.append((wid, elem.text, elem.attrib.get('punc') == 'true'))

                da_tree = ET.fromstring(z.read(da_file))
                for dact in da_tree.findall('{http://nite.sourceforge.net/}root/dact') or da_tree.findall('dact'):
                    ptr = dact.find('{http://nite.sourceforge.net/}pointer')
                    da_type = 'Unknown'
                    if ptr is not None:
                        href = ptr.attrib.get('href', '')
                        da_code = href.split('#id(')[-1].rstrip(')')
                        da_type = DA_ASPECT_MAP.get(da_code, da_code)

                    child = dact.find('{http://nite.sourceforge.net/}child')
                    if child is None:
                        continue

                    start_w, end_w = parse_id_range(child.attrib.get('href', ''))
                    if not start_w or start_w not in word_id_to_idx or end_w not in word_id_to_idx:
                        continue

                    i1 = word_id_to_idx[start_w]
                    i2 = word_id_to_idx[end_w]
                    if i2 < i1:
                        continue

                    tokens = [w for _, w, _ in ordered_words[i1:i2+1]]
                    text = ' '.join(tokens).strip()
                    text = re.sub(r'\s+([,.:;?!])', r'\1', text)

                    # Filter disfluencies and short fragments
                    if len(text.split()) < 3:
                        continue
                    if re.match(r'^(uh|um|mm|yeah|okay|right|ah|mhm)\b', text, re.I) and len(text.split()) <= 4:
                        continue

                    lower_text = text.lower()

                    # Class assignment
                    if da_type in ('Offer', 'Elicit-Offer-Or-Suggestion'):
                        label = 'ACTION_ITEM'
                    elif da_type == 'Suggest':
                        label = 'SUGGESTION'
                    elif (da_type in ('Inform', 'Assess')) and bool(COMPLETED_REGEX.search(lower_text)) and bool(PAST_PRONOUNS.search(lower_text)):
                        label = 'COMPLETED_ACTION'
                    else:
                        label = 'NON_ACTIONABLE'

                    assignee = None
                    if label == 'ACTION_ITEM':
                        if da_type == 'Offer':
                            assignee = f"Speaker_{speaker}"
                        else:
                            m_voc = re.match(r'^([A-Z][a-z]+)[,\s]', text)
                            if m_voc and m_voc.group(1).lower() not in {"so", "well", "and", "yeah", "okay", "alright", "then", "just"}:
                                assignee = m_voc.group(1)

                    deadline_text = None
                    m_temp = re.search(r'\b(by\s+[A-Za-z]+|before\s+[A-Za-z]+|tomorrow|next\s+week|tonight|this\s+afternoon|by\s+end\s+of\s+day|within\s+\w+\s+days)\b', lower_text)
                    if m_temp:
                        deadline_text = m_temp.group(0)

                    meeting_utterances.append({
                        "speaker": f"Speaker_{speaker}",
                        "text": text,
                        "label": label,
                        "original_da": da_type,
                        "assignee": assignee,
                        "deadline_text": deadline_text,
                    })

            # Subsample non-actionable utterances per meeting to maintain a balanced academic distribution
            action_items = [u for u in meeting_utterances if u["label"] == "ACTION_ITEM"]
            suggestions = [u for u in meeting_utterances if u["label"] == "SUGGESTION"]
            completed = [u for u in meeting_utterances if u["label"] == "COMPLETED_ACTION"]
            non_actionable = [u for u in meeting_utterances if u["label"] == "NON_ACTIONABLE"]

            # Balance representation: keep all action items, suggestions, completed, and balanced non-actionable
            cap_na = max(10, (len(action_items) + len(suggestions) + len(completed)) * 2)
            filtered_utterances = action_items + suggestions + completed + non_actionable[:cap_na]

            for u in filtered_utterances:
                overall_stats[f"{split}:{u['label']}"] += 1

            if filtered_utterances:
                compiled_meetings.append({
                    "meeting_id": m_id,
                    "split": split,
                    "description": f"AMI Meeting Session {m_id}",
                    "utterances": filtered_utterances
                })

        # Also incorporate the base dataset curated meetings
        if BASE_DATASET.exists():
            with open(BASE_DATASET, "r", encoding="utf-8") as bf:
                base_data = json.load(bf)
                for bm in base_data.get("meetings", []):
                    b_split = bm.get("split", "train")
                    for u in bm.get("utterances", []):
                        overall_stats[f"{b_split}:{u['label']}"] += 1
                    compiled_meetings.append(bm)

        dataset_obj = {
            "metadata": {
                "name": "Meeting Task Extraction & Dialogue Act Corpus (AMI + Curated Benchmark)",
                "version": "2.0.0",
                "total_meetings": len(compiled_meetings),
                "train_meetings": sum(1 for m in compiled_meetings if m.get("split") == "train"),
                "test_meetings": sum(1 for m in compiled_meetings if m.get("split") == "test"),
                "source": "AMI Meeting Corpus (1.6.2) & Curated Enterprise Dialogues",
            },
            "meetings": compiled_meetings
        }

        with open(OUTPUT_DATASET, "w", encoding="utf-8") as out_f:
            json.dump(dataset_obj, out_f, indent=2)

        print("\n" + "="*50)
        print(f"Large Dataset Compiled: {OUTPUT_DATASET}")
        print(f"Total Meetings: {len(compiled_meetings)}")
        total_utts = sum(len(m['utterances']) for m in compiled_meetings)
        print(f"Total Utterances: {total_utts}")
        print("Class distributions in final dataset:")
        for k, v in sorted(overall_stats.items()):
            print(f"  {k:25s}: {v}")
        print("="*50)


if __name__ == "__main__":
    build_large_corpus(max_meetings=80)
