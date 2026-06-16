"""
Audio + Text exploration script for the Campus Chatbot project.
- Transcribes voice query MP3s using Whisper
- Loads the synthetic text corpus
- Tokenizes, removes stopwords, and explores intent distribution
"""

import os
import json
import string
import matplotlib.pyplot as plt

AUDIO_DIR = "data/audio"
TEXT_CORPUS_PATH = "data/knowledge_base/text_corpus.json"
VOICE_QUERIES_PATH = "data/knowledge_base/voice_queries.json"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Minimal stopword list (avoids extra NLTK download requirement)
STOPWORDS = set("""
a an the is are was were be been being to of in on at for with by from
this that these those it its as and or but if then than so do does did
have has had can could will would should may might must i you he she we they
my your his her our their what where when why how
""".split())


def transcribe_audio_files(audio_dir=AUDIO_DIR, model_size="base"):
    """Transcribe all mp3 files in audio_dir using Whisper. Returns dict {filename: transcript}."""
    import whisper

    if not os.path.isdir(audio_dir):
        print(f"Audio directory '{audio_dir}' not found.")
        return {}

    files = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith((".mp3", ".wav", ".m4a"))])
    if not files:
        print(f"No audio files found in '{audio_dir}'.")
        return {}

    print(f"Loading Whisper '{model_size}' model (this may take a moment)...")
    model = whisper.load_model(model_size)

    transcripts = {}
    for fname in files:
        path = os.path.join(audio_dir, fname)
        print(f"Transcribing {fname}...")
        result = model.transcribe(path)
        transcripts[fname] = result["text"].strip()
        print(f"  -> {transcripts[fname]}")

    # Save transcripts
    out_path = os.path.join(OUTPUT_DIR, "voice_query_transcripts.json")
    with open(out_path, "w") as f:
        json.dump(transcripts, f, indent=2)
    print(f"Saved transcripts to {out_path}")

    return transcripts


def word_error_rate(reference, hypothesis):
    """Compute WER using Levenshtein distance at word level."""
    ref_words = reference.lower().translate(str.maketrans("", "", string.punctuation)).split()
    hyp_words = hypothesis.lower().translate(str.maketrans("", "", string.punctuation)).split()

    # Dynamic programming edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    if len(ref_words) == 0:
        return 0.0
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def evaluate_transcripts(transcripts, voice_queries_path=VOICE_QUERIES_PATH):
    """Compare Whisper transcripts against ground-truth original_text and compute WER."""
    if not os.path.isfile(voice_queries_path):
        print(f"Voice queries file not found at {voice_queries_path}")
        return

    with open(voice_queries_path, "r") as f:
        voice_queries = json.load(f)

    print("\n=== Whisper Transcription Evaluation (WER) ===")
    total_wer = 0.0
    count = 0
    results = []

    for vq in voice_queries:
        fname = os.path.basename(vq["audio_file"])
        reference = vq["original_text"]
        hypothesis = transcripts.get(fname, "")

        if not hypothesis:
            continue

        wer = word_error_rate(reference, hypothesis)
        total_wer += wer
        count += 1

        results.append({
            "file": fname,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": round(wer, 3),
        })

        print(f"  {fname}")
        print(f"    Reference : {reference}")
        print(f"    Hypothesis: {hypothesis}")
        print(f"    WER       : {wer:.3f}")

    if count > 0:
        avg_wer = total_wer / count
        print(f"\nAverage WER across {count} samples: {avg_wer:.3f}")

        out_path = os.path.join(OUTPUT_DIR, "whisper_wer_results.json")
        with open(out_path, "w") as f:
            json.dump({"average_wer": round(avg_wer, 3), "results": results}, f, indent=2)
        print(f"Saved WER results to {out_path}")


def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]


def explore_text_corpus(path=TEXT_CORPUS_PATH):
    if not os.path.isfile(path):
        print(f"Text corpus not found at {path}")
        return

    with open(path, "r") as f:
        corpus = json.load(f)

    print(f"Loaded {len(corpus)} text samples.")

    # Tokenization example
    print("\nSample tokenization:")
    for sample in corpus[:3]:
        tokens = tokenize(sample["text"])
        print(f"  Original: {sample['text']}")
        print(f"  Tokens (lowercased, no stopwords): {tokens}\n")

    # Intent distribution
    intent_counts = {}
    for sample in corpus:
        intent_counts[sample["intent"]] = intent_counts.get(sample["intent"], 0) + 1

    print("Intent distribution:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count}")

    plt.figure(figsize=(6, 4))
    plt.bar(intent_counts.keys(), intent_counts.values(), color="seagreen")
    plt.title("Intent Class Distribution")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "intent_distribution.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved intent distribution plot to {save_path}")

    # Vocabulary size
    vocab = set()
    for sample in corpus:
        vocab.update(tokenize(sample["text"]))
    print(f"\nVocabulary size (after stopword removal): {len(vocab)}")
    print(f"Sample vocabulary: {sorted(list(vocab))[:20]}")


if __name__ == "__main__":
    print("=== Transcribing voice queries ===")
    transcripts = transcribe_audio_files(model_size="base")

    if transcripts:
        evaluate_transcripts(transcripts)

    print("\n=== Exploring text corpus ===")
    explore_text_corpus()