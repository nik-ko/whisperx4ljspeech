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


def get_wav(input_path):
    """
    Converts an audio file to a temporary WAV file if not in a supported format
    """
    file_extension = input_path.split('.')[-1].lower()

    if file_extension in WHISPER_FORMATS:
        return input_path, None

    print(f"Converting {input_path} to temporary WAV file...")
    audio = AudioSegment.from_file(input_path)
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav.name, format="wav")

    return temp_wav.name, temp_wav


def extract_audio(audio_segment, start_time, end_time):
    start_ms = start_time * 1000
    end_ms = end_time * 1000

    return audio_segment[start_ms:end_ms]


def process_audio_files(model_name, device, input_dir, output_dir, language=None, gpu_id=0, compute_type="float16"):
    audio_dir = os.path.join(output_dir, "wav")
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    metadata = []

    model = whisperx.load_model(model_name, device=device, language=language, compute_type=compute_type,
                                device_index=gpu_id)

    for i, audio_file in enumerate(sorted(os.listdir(input_dir))):
        input_path = os.path.join(input_dir, audio_file)

        if not any(audio_file.endswith(ext) for ext in WHISPER_FORMATS + ADDIDTIONAL_FORMATS):
            print(f"Skipping non-audio file: {audio_file}")
            continue

        print(f"Processing {input_path}...")

        converted_audio_path, temp_wav_file = get_wav(input_path)

        try:
            audio = whisperx.load_audio(converted_audio_path)
            result = model.transcribe(audio, chunk_size=10)

            segments = result["segments"]

            full_audio_segment = AudioSegment.from_file(converted_audio_path)

            num_segments = len(segments)
            print(f"Splitting audio file into {num_segments} chunks...")

            # Process each segment detected by WhisperX
            for j, segment in enumerate(segments):
                start_time = segment['start']
                end_time = segment['end']

                audio_segment = extract_audio(full_audio_segment, start_time, end_time)

                sentence_id = f"{i + 1:06d}_{j + 1:06d}"
                sentence_path = os.path.join(audio_dir, f"{sentence_id}.wav")

                audio_segment.export(sentence_path, format="wav")

                text = segment["text"]

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

    metadata_csv_path = os.path.join(output_dir, "metadata.csv")

    with open(metadata_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='|')

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
