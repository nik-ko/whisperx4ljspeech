import argparse
import torch
import whisperx
import os
import gc
import tempfile
from pydub import AudioSegment
import csv

# Audio formats supported by WhisperX
WHISPER_FORMATS = ['wav', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'webm']

# Additional audio formats
ADDIDTIONAL_FORMATS = ['flac', 'ogg', 'aac', 'wma']


def convert_to_temp_wav(input_path):
    """
    Converts an audio file to a temporary WAV file if it's not already in a supported format.
    Returns the path to the temporary WAV file.
    """
    file_extension = input_path.split('.')[-1].lower()

    if file_extension in WHISPER_FORMATS:
        return input_path, None  # No conversion needed, return original path and no temp file

    # If not supported, convert the file to a temporary WAV
    print(f"Converting {input_path} to temporary WAV format...")
    audio = AudioSegment.from_file(input_path)
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav.name, format="wav")

    return temp_wav.name, temp_wav


def extract_audio_chunk(audio_segment, start_time, end_time):
    start_ms = start_time * 1000  # Convert seconds to milliseconds
    end_ms = end_time * 1000  # Convert seconds to milliseconds

    return audio_segment[start_ms:end_ms]


def process_audio_files(model_name, device, input_dir, output_dir, language="en", gpu_id=0, compute_type="float16"):
    # Create output directories if they don't exist
    audio_dir = os.path.join(output_dir, "audio")
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    metadata = []

    # Load WhisperX model
    model = whisperx.load_model(model_name, device=device, language=language, compute_type=compute_type,
                                device_index=gpu_id)

    # Process each audio file in the input directory
    for i, audio_file in enumerate(sorted(os.listdir(input_dir))):
        input_path = os.path.join(input_dir, audio_file)

        if not any(audio_file.endswith(ext) for ext in WHISPER_FORMATS + ADDIDTIONAL_FORMATS):
            print(f"Skipping non-audio file: {audio_file}")
            continue

        print(f"Processing {input_path}...")

        # Convert unsupported formats to a temporary WAV before processing
        converted_audio_path, temp_wav_file = convert_to_temp_wav(input_path)

        try:
            audio = whisperx.load_audio(converted_audio_path)
            result = model.transcribe(audio)

            # Load alignment model for precise word-level timestamps
            model_a, metadata_align = whisperx.load_align_model(language_code=result["language"], device=device)
            aligned_result = whisperx.align(result["segments"], model_a, metadata_align, audio, device=device)

            full_audio_segment = AudioSegment.from_file(converted_audio_path)

            num_segments = len(aligned_result["segments"])
            print(f"Splitting audio file into {num_segments} chunks...")

            # Process each segment detected by WhisperX
            for j, segment in enumerate(aligned_result["segments"]):
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()

                # Extract the segmented chunk
                chunk_audio_segment = extract_audio_chunk(full_audio_segment, start_time, end_time)

                sentence_id = f"{i + 1:06d}_{j + 1:06d}"
                sentence_path = os.path.join(audio_dir, f"{sentence_id}.wav")

                chunk_audio_segment.export(sentence_path, format="wav")

                metadata.append({
                    "id": sentence_id,
                    "text": text,
                    "text_cleaned": text.lower()
                })

        finally:
            if temp_wav_file:
                temp_wav_file.close()
                os.remove(temp_wav_file.name)

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Create metadata.csv file using the csv module
    metadata_csv_path = os.path.join(output_dir, "metadata.csv")

    with open(metadata_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='|')

        # Write header row (optional)
        csv_writer.writerow(["id", "text", "text_cleaned"])

        # Write data rows (id | text | text_cleaned)
        for entry in metadata:
            csv_writer.writerow([entry["id"], entry["text"], entry["text_cleaned"]])

    print(f"Processed {len(metadata)} sentences.")
    print(f"CSV file saved to {metadata_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Create dataset in LJ Speech format using WhisperX.")
    parser.add_argument('--model', type=str, default="large-v3", help="WhisperX model to use (default: large-v3)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--gpu', type=int, help="GPU ID to use (e.g., 0)")
    group.add_argument('--cpu', action='store_true', help="Use CPU for processing")
    parser.add_argument('--language', type=str, default=None,
                        help="Language (default: detect automatically)")
    parser.add_argument('--compute_type', type=str, default="float16",
                        help="Compute type (default: float16), change to \"int8\" if low on GPU mem (may reduce accuracy)")
    parser.add_argument('--input', type=str, required=True,
                        help="Directory containing input audio files")
    parser.add_argument('--output', type=str, required=True,
                        help="Directory to save processed output")

    args = parser.parse_args()

    if args.cpu:
        device = "cpu"

    else:
        device = "cuda"  # f"cuda:{args.gpu}"

    process_audio_files(args.model, device, args.input, args.output,
                        args.language,
                        args.gpu,
                        args.compute_type)


if __name__ == "__main__":
    main()
