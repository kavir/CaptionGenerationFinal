# =====================================================
# EVALUATE (TEST) YOUR IMAGE CAPTION MODEL (NO TRAINING)
# - Creates a random test split by IMAGE
# - Generates captions
# - Computes BLEU-1..BLEU-4
# - Prints some qualitative examples
# =====================================================

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

# -----------------------------
# CPU only
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")

# -----------------------------
# PATHS (update if needed)
# -----------------------------
IMAGE_DIR = "/home/kavir/image_project/Images"
CAPTIONS_FILE = "/home/kavir/image_project/captions.txt"
MODEL_PATH = "/home/kavir/image_project/model_checkpoints/best_caption_model.keras"

# -----------------------------
# MUST MATCH TRAINING
# -----------------------------
IMAGE_SIZE = (224, 224)
SEQ_LENGTH = 20
VOCAB_SIZE = 5000
EMBED_DIM = 256

# -----------------------------
# Split settings
# -----------------------------
TEST_RATIO = 0.10          # 10% images for test
EVAL_MAX_IMAGES = 500      # evaluate on at most 500 test images (speed control)
RANDOM_SEED = 42

print("TensorFlow:", tf.__version__)
print("GPUs visible:", tf.config.list_physical_devices("GPU"))

# =====================================================
# 1) LOAD CAPTIONS (image -> list of captions)
# =====================================================
captions_dict = {}
all_caption_texts = []

with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("image,caption"):
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        img = parts[0].strip()
        cap = parts[1].strip()
        if not img or not cap:
            continue
        captions_dict.setdefault(img, []).append(cap)
        all_caption_texts.append("sos " + cap + " eos")

print(f"Images with captions: {len(captions_dict)}")
print(f"Total captions: {len(all_caption_texts)}")

# =====================================================
# 2) TOKENIZER (same settings as training)
# =====================================================
tokenizer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=SEQ_LENGTH,
    standardize="lower_and_strip_punctuation"
)
tokenizer.adapt(tf.data.Dataset.from_tensor_slices(all_caption_texts))
vocab = tokenizer.get_vocabulary()
vocab_set = set(vocab)

FALLBACK_WORD = "a" if "a" in vocab_set else vocab[1]
print("Tokenizer vocab size:", len(vocab))

# =====================================================
# 3) IMAGE LOADER (jpg/png safe)
# =====================================================
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

# =====================================================
# 4) CUSTOM LAYERS (must match training)
# =====================================================
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads=4, ff_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        x = self.norm1(inputs + attn_output)
        return self.norm2(x + self.ffn(x))

class TransformerDecoder(layers.Layer):
    def __init__(self, vocab_size, embed_dim, num_heads=4, ff_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.embed = layers.Embedding(vocab_size, embed_dim)
        self.att1 = layers.MultiHeadAttention(num_heads, embed_dim)
        self.att2 = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()
        self.out = layers.Dense(vocab_size)

    def call(self, x, enc_output):
        x = self.embed(x)
        x = self.norm1(x + self.att1(x, x))
        x = self.norm2(x + self.att2(x, enc_output))
        x = self.norm3(x + self.ffn(x))
        return self.out(x)

# =====================================================
# 5) LOAD MODEL
# =====================================================
print("\nLoading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"TransformerEncoder": TransformerEncoder, "TransformerDecoder": TransformerDecoder},
    compile=False
)
print("✓ Model loaded")

# =====================================================
# 6) GENERATE CAPTION (greedy decoding)
# =====================================================
def generate_caption(image_path, max_length=20, min_length=3):
    img = load_image(image_path)
    img = tf.expand_dims(img, 0)

    caption = "sos"

    for i in range(max_length):
        seq = tokenizer([caption])
        preds = model([img, seq], training=False)

        next_id = int(tf.argmax(preds[0, -1]).numpy())
        next_word = vocab[next_id]

        # avoid stopping too early
        if i < min_length and next_word in ("eos", "[UNK]"):
            next_word = FALLBACK_WORD

        if next_word == "eos":
            break

        if next_word == "[UNK]":
            next_word = FALLBACK_WORD

        caption += " " + next_word

    result = caption.replace("sos ", "").strip()
    return result if result else "(no caption)"

# =====================================================
# 7) MAKE TEST SPLIT (by image)
# =====================================================
random.seed(RANDOM_SEED)
all_images = list(captions_dict.keys())
random.shuffle(all_images)

test_size = int(len(all_images) * TEST_RATIO)
test_images = all_images[:test_size]

# limit for speed
if EVAL_MAX_IMAGES and len(test_images) > EVAL_MAX_IMAGES:
    test_images = test_images[:EVAL_MAX_IMAGES]

print(f"\nTest images: {len(test_images)} (ratio={TEST_RATIO})")

# =====================================================
# 8) BLEU EVALUATION
# =====================================================
try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method4
except ImportError:
    raise SystemExit("NLTK not installed. Run: pip install nltk")

references = []
predictions = []

skipped_missing = 0

for img_name in test_images:
    img_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.exists(img_path):
        skipped_missing += 1
        continue

    pred = generate_caption(img_path)
    predictions.append(pred.split())

    # Reference captions for that image
    refs = [c.lower().split() for c in captions_dict[img_name]]
    references.append(refs)

print("Skipped missing files:", skipped_missing)

bleu1 = corpus_bleu(references, predictions, weights=(1, 0, 0, 0), smoothing_function=smoothie)
bleu2 = corpus_bleu(references, predictions, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
bleu3 = corpus_bleu(references, predictions, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
bleu4 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

print("\n==================== RESULTS ====================")
print(f"BLEU-1: {bleu1:.4f}")
print(f"BLEU-2: {bleu2:.4f}")
print(f"BLEU-3: {bleu3:.4f}")
print(f"BLEU-4: {bleu4:.4f}")
print("=================================================\n")

# =====================================================
# 9) QUALITATIVE SAMPLES
# =====================================================
print("Sample predictions (with references):")
for img_name in random.sample(test_images, min(5, len(test_images))):
    img_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.exists(img_path):
        continue
    pred = generate_caption(img_path)
    print("\nImage:", img_name)
    print("Pred :", pred)
    for i, ref in enumerate(captions_dict[img_name][:5], 1):
        print(f"Ref {i}:", ref)
