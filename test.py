import tensorflow as tf
import numpy as np
import json
import re
import argparse
import os
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow import keras

# ────────────────────────────────────────────────
# Paths — adjust only if your folder structure is different
# ────────────────────────────────────────────────
MODEL_CONFIG_PATH   = "save_train_dir/config_train.json"
MODEL_WEIGHTS_PATH  = "save_train_dir/model_weights_coco.h5"
TEXT_DATA_JSON_PATH = "COCO_dataset/text_data.json"          # must exist!

# Desired image dimensions (must match training)
IMAGE_SIZE = (299, 299)

# Your custom text cleaning function (exact copy from training)
strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# ────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────
def get_cnn_model():
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, 1280))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = layers.Dense(embed_dim, activation="relu")
        self.layernorm_1 = layers.LayerNormalization()

    def call(self, inputs, training, mask=None):
        inputs = self.dense_proj(inputs)
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=None
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        return proj_input


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=embed_dim, sequence_length=25, vocab_size=self.vocab_size
        )
        self.out = layers.Dense(self.vocab_size)
        self.dropout_1 = layers.Dropout(0.1)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)
        inputs = self.dropout_1(inputs, training=training)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        else:
            combined_mask = None
            padding_mask = None

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=combined_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        proj_out = self.layernorm_3(out_2 + proj_output)
        proj_out = self.dropout_2(proj_out, training=training)

        preds = self.out(proj_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


class ImageCaptioningModel(keras.Model):
    def __init__(self, cnn_model, encoder, decoder, num_captions_per_image=5):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.num_captions_per_image = num_captions_per_image

    def call(self, inputs):
        x = self.cnn_model(inputs[0])
        x = self.encoder(x, False)
        x = self.decoder(inputs[2], x, training=inputs[1], mask=None)
        return x


def read_image_inf(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img


def get_inference_model(model_config_path):
    with open(model_config_path) as json_file:
        model_config = json.load(json_file)

    EMBED_DIM = model_config["EMBED_DIM"]
    FF_DIM = model_config["FF_DIM"]
    NUM_HEADS = model_config["NUM_HEADS"]
    VOCAB_SIZE = model_config["VOCAB_SIZE"]

    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(
        embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS
    )
    decoder = TransformerDecoderBlock(
        embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE
    )
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder
    )

    # Force model initialization
    cnn_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    training = False
    decoder_input = tf.keras.layers.Input(shape=(None,))
    caption_model([cnn_input, training, decoder_input])

    return caption_model


def generate_caption(image_path, caption_model, tokenizer, seq_length):
    vocab = tokenizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = seq_length - 1

    img = read_image_inf(image_path)
    img_embed = caption_model.cnn_model(img)
    encoded_img = caption_model.encoder(img_embed, training=False)

    decoded_caption = "sos"
    for i in range(max_decoded_sentence_length):
        tokenized_caption = tokenizer([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup.get(sampled_token_index, "[UNK]")
        if sampled_token == "eos":
            break
        decoded_caption += " " + sampled_token

    return decoded_caption.replace("sos ", "").strip()


# ────────────────────────────────────────────────
# Main inference logic
# ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning Inference")
    parser.add_argument('--image', required=True, help="Path to input image")
    args = parser.parse_args()
    image_path = args.image

    if not os.path.exists(image_path):
        print(f"Error: Image file not found → {image_path}")
        exit(1)

    # ─── Tokenizer reconstruction & adaptation ───────────────────────
    print("Re-creating tokenizer and loading vocabulary from text_data.json...")

    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=2000000,
        standardize=custom_standardization,
        output_mode="int",
        output_sequence_length=25
    )

    if not os.path.exists(TEXT_DATA_JSON_PATH):
        print(f"ERROR: {TEXT_DATA_JSON_PATH} not found!")
        print("You need this file to restore the vocabulary.")
        exit(1)

    with open(TEXT_DATA_JSON_PATH, "r", encoding="utf-8") as f:
        text_data = json.load(f)

    print(f"Type of loaded data: {type(text_data).__name__}")
    print(f"Length of loaded data: {len(text_data):,}")

    # Collect all captions (handle different possible formats)
    all_captions = []

    if isinstance(text_data, list):
        print("Detected: top-level list")
        # Flat list of strings
        if all(isinstance(item, str) for item in text_data[:200]):
            all_captions = text_data
            print("→ Using flat list of captions directly")
        # List of dicts
        elif isinstance(text_data[0], dict) if text_data else False:
            print("→ Detected list of dictionaries")
            for item in text_data:
                if "caption" in item:
                    all_captions.append(item["caption"])
                elif "captions" in item and isinstance(item["captions"], list):
                    all_captions.extend(item["captions"])
                elif "text" in item:
                    all_captions.append(item["text"])
        else:
            print("Warning: Could not extract captions from list items")

    elif isinstance(text_data, dict):
        print("Detected: top-level dictionary")
        for value in text_data.values():
            if isinstance(value, list):
                all_captions.extend(value)
            elif isinstance(value, str):
                all_captions.append(value)

    else:
        print("ERROR: Unsupported format in text_data.json")
        exit(1)

    if not all_captions:
        print("ERROR: No captions could be extracted from the file")
        exit(1)

    print(f"Extracted {len(all_captions):,} captions for tokenizer adaptation")

    # Adapt tokenizer
    text_ds = tf.data.Dataset.from_tensor_slices(all_captions).batch(2048).prefetch(tf.data.AUTOTUNE)
    tokenizer.adapt(text_ds)

    vocab_size = len(tokenizer.get_vocabulary())
    print(f"Tokenizer vocabulary built — size: {vocab_size:,}")
    print("First 15 tokens:", tokenizer.get_vocabulary()[:15])

    # ─── Load model ────────────────────────────────────────────────
    print("\nLoading model configuration and weights...")
    model = get_inference_model(MODEL_CONFIG_PATH)
    model.load_weights(MODEL_WEIGHTS_PATH)
    print("Model weights loaded successfully.")

    with open(MODEL_CONFIG_PATH) as f:
        config = json.load(f)
    seq_length = config["SEQ_LENGTH"]

    # ─── Generate caption ──────────────────────────────────────────
    print("\nGenerating caption for:", os.path.basename(image_path))
    try:
        caption = generate_caption(image_path, model, tokenizer, seq_length)
        print(f"\nPREDICTED CAPTION:\n→ {caption}")
    except Exception as e:
        print("Error during caption generation:")
        print(str(e))