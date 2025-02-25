#!/usr/bin/env python3
import os
import argparse
import yt_dlp
from audio_separator.separator import Separator

def downloader(url):
    """Download the audio from a video URL using yt_dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'ytdl/%(title)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        info_dict = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info_dict).rsplit('.', 1)[0] + '.wav'
        print("Download Complete!")
        return file_path

def main():
    parser = argparse.ArgumentParser(
        description="Separate audio into vocals and instrumental using a UVR model."
    )
    parser.add_argument(
        "--video_url",
        type=str,
        default="https://youtu.be/UW547UGoM_g?si=VaAB_0BgoGq9x4Q6",
        help="URL of the video/audio to download."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/content/Vocales",
        help="Folder where the separated audio will be saved."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        help="Name of the model to use for separation."
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="wav",
        choices=["wav", "flac", "mp3", "ogg", "opus", "m4a", "aiff", "ac3"],
        help="Audio format for the output files."
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=8,
        help="Amount of overlap between prediction windows."
    )
    # By default, we want to use autocast (recommended for GPU inference). 
    # Use --no_autocast to disable it.
    parser.add_argument(
        "--no_autocast",
        action="store_true",
        help="Disable PyTorch autocast (default: enabled)."
    )
    
    args = parser.parse_args()
    use_autocast = not args.no_autocast

    print("Starting audio separation using UVR5!...")
    print(f"Video URL: {args.video_url}")
    print(f"Output Folder: {args.output_folder}")
    print(f"Model Used: {args.model}")
    print(f"Output Format: {args.output_format}")
    print(f"Overlap: {args.overlap}")
    
    # Download the audio
    audio_input = downloader(args.video_url)
    
    # Create the separator instance (additional parameters like overlap/output_format
    # could be passed here if supported by your Separator implementation)
    separator = Separator(
        output_dir=args.output_folder,
        use_autocast=use_autocast
    )
    
    # Load the chosen model
    separator.load_model(args.model)
    
    # Define output file name mapping
    output_names = {
        "Vocals": "vocals_output",
        "Instrumental": "instrumental_output",
    }
    
    # Perform the separation
    output_files = separator.separate(
        audio_input,
        output_names
    )
    
    print(f"Separation complete! Output file(s): {' '.join(output_files)}")

if __name__ == "__main__":
    main()
