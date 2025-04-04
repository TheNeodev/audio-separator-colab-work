import argparse
import os
import yt_dlp
import torch
import shutil
import logging
from audio_separator.separator import Separator

# Model lists
ROFORMER_MODELS = {
    'BS-Roformer-Viperx-1297.ckpt': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
    'BS-Roformer-Viperx-1296.ckpt': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
    'BS-Roformer-Viperx-1053.ckpt': 'model_bs_roformer_ep_937_sdr_10.5309.ckpt',
    'BS-Roformer-De-Reverb.ckpt': 'deverb_bs_roformer_8_384dim_10depth.ckpt',
    'Mel-Roformer-Viperx-1143.ckpt': 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt',
    'Mel-Roformer-Crowd-Aufr33-Viperx.ckpt': 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
    'Mel-Roformer-Karaoke-Aufr33-Viperx.ckpt': 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
    'Mel-Roformer-Denoise-Aufr33': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
    'Mel-Roformer-Denoise-Aufr33-Aggr': 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
    'MelBand Roformer Kim | Inst V1 by Unwa': 'melband_roformer_inst_v1.ckpt',
    'MelBand Roformer Kim | Inst V2 by Unwa': 'melband_roformer_inst_v2.ckpt',
    'MelBand Roformer Kim | InstVoc Duality V1 by Unwa': 'melband_roformer_instvoc_duality_v1.ckpt',
    'MelBand Roformer Kim | InstVoc Duality V2 by Unwa': 'melband_roformer_instvox_duality_v2.ckpt',
}
MDX23C_MODELS = [
    'MDX23C_D1581.ckpt',
    'MDX23C-8KFFT-InstVoc_HQ.ckpt',
    'MDX23C-8KFFT-InstVoc_HQ_2.ckpt',
]
MDXNET_MODELS = [
    'UVR-MDX-NET-Inst_full_292.onnx',
    'UVR-MDX-NET_Inst_187_beta.onnx',
    'UVR-MDX-NET_Inst_82_beta.onnx',
    'UVR-MDX-NET_Inst_90_beta.onnx',
    'UVR-MDX-NET_Main_340.onnx',
    'UVR-MDX-NET_Main_390.onnx',
    'UVR-MDX-NET_Main_406.onnx',
    'UVR-MDX-NET_Main_427.onnx',
    'UVR-MDX-NET_Main_438.onnx',
    'UVR-MDX-NET-Inst_HQ_1.onnx',
    'UVR-MDX-NET-Inst_HQ_2.onnx',
    'UVR-MDX-NET-Inst_HQ_3.onnx',
    'UVR-MDX-NET-Inst_HQ_4.onnx',
    'UVR_MDXNET_Main.onnx',
    'UVR-MDX-NET-Inst_Main.onnx',
    'UVR_MDXNET_1_9703.onnx',
    'UVR_MDXNET_2_9682.onnx',
    'UVR_MDXNET_3_9662.onnx',
    'UVR-MDX-NET-Inst_1.onnx',
    'UVR-MDX-NET-Inst_2.onnx',
    'UVR-MDX-NET-Inst_3.onnx',
    'UVR_MDXNET_KARA.onnx',
    'UVR_MDXNET_KARA_2.onnx',
    'UVR_MDXNET_9482.onnx',
    'UVR-MDX-NET-Voc_FT.onnx',
    'Kim_Vocal_1.onnx',
    'Kim_Vocal_2.onnx',
    'Kim_Inst.onnx',
    'Reverb_HQ_By_FoxJoy.onnx',
    'UVR-MDX-NET_Crowd_HQ_1.onnx',
    'kuielab_a_vocals.onnx',
    'kuielab_a_other.onnx',
    'kuielab_a_bass.onnx',
    'kuielab_a_drums.onnx',
    'kuielab_b_vocals.onnx',
    'kuielab_b_other.onnx',
    'kuielab_b_bass.onnx',
    'kuielab_b_drums.onnx',
]
VR_ARCH_MODELS = [
    '1_HP-UVR.pth',
    '2_HP-UVR.pth',
    '3_HP-Vocal-UVR.pth',
    '4_HP-Vocal-UVR.pth',
    '5_HP-Karaoke-UVR.pth',
    '6_HP-Karaoke-UVR.pth',
    '7_HP2-UVR.pth',
    '8_HP2-UVR.pth',
    '9_HP2-UVR.pth',
    '10_SP-UVR-2B-32000-1.pth',
    '11_SP-UVR-2B-32000-2.pth',
    '12_SP-UVR-3B-44100.pth',
    '13_SP-UVR-4B-44100-1.pth',
    '14_SP-UVR-4B-44100-2.pth',
    '15_SP-UVR-MID-44100-1.pth',
    '16_SP-UVR-MID-44100-2.pth',
    '17_HP-Wind_Inst-UVR.pth',
    'UVR-DeEcho-DeReverb.pth',
    'UVR-De-Echo-Normal.pth',
    'UVR-De-Echo-Aggressive.pth',
    'UVR-DeNoise.pth',
    'UVR-DeNoise-Lite.pth',
    'UVR-BVE-4B_SN-44100-1.pth',
    'MGM_HIGHEND_v4.pth',
    'MGM_LOWEND_A_v4.pth',
    'MGM_LOWEND_B_v4.pth',
    'MGM_MAIN_v4.pth',
]
DEMUCS_MODELS = [
    'htdemucs_ft.yaml',
    'htdemucs_6s.yaml',
    'htdemucs.yaml',
    'hdemucs_mmi.yaml',
]

def downloader(url, output_dir="ytdl"):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '32',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessor_args': ['-acodec', 'pcm_f32le'],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info['title']
            ydl.download([url])
            file_path = os.path.join(output_dir, f"{video_title}.wav")
            return os.path.abspath(file_path) if os.path.exists(file_path) else None
    except Exception as e:
        raise RuntimeError(f"Download failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description=f"CLI for audio separation using various models, by [NeoDev](https://github.com/TheNeodev)")
    subparsers = parser.add_subparsers(dest='model_type', required=True)

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--input', required=True, help="Input file path or YouTube URL")
    common_parser.add_argument('--output-dir', default='output', help="Output directory")
    common_parser.add_argument('--model-dir', default='models', help="Directory containing models")
    common_parser.add_argument('--output-format', default='WAV', help="Output audio format")
    common_parser.add_argument('--norm-thresh', type=float, default=0.9, help="Normalization threshold")
    common_parser.add_argument('--amp-thresh', type=float, default=0.005, help="Amplification threshold")
    common_parser.add_argument('--batch-size', type=int, default=1, help="Batch size")

    # Roformer subcommand
    roformer_parser = subparsers.add_parser('roformer', parents=[common_parser])
    roformer_parser.add_argument('--model-name', choices=ROFORMER_MODELS.keys(), required=True)
    roformer_parser.add_argument('--seg-size', type=int, default=256)
    roformer_parser.add_argument('--override-seg-size', action='store_true')
    roformer_parser.add_argument('--overlap', type=float, default=0.25)
    roformer_parser.add_argument('--pitch-shift', type=int, default=0)

    # MDX23C subcommand
    mdx23c_parser = subparsers.add_parser('mdx23c', parents=[common_parser])
    mdx23c_parser.add_argument('--model-name', choices=MDX23C_MODELS, required=True)
    mdx23c_parser.add_argument('--seg-size', type=int, default=256)
    mdx23c_parser.add_argument('--override-seg-size', action='store_true')
    mdx23c_parser.add_argument('--overlap', type=float, default=0.25)
    mdx23c_parser.add_argument('--pitch-shift', type=int, default=0)

    # MDX subcommand
    mdx_parser = subparsers.add_parser('mdx', parents=[common_parser])
    mdx_parser.add_argument('--model-name', choices=MDXNET_MODELS, required=True)
    mdx_parser.add_argument('--hop-length', type=int, default=1024)
    mdx_parser.add_argument('--seg-size', type=int, default=256)
    mdx_parser.add_argument('--overlap', type=float, default=0.25)
    mdx_parser.add_argument('--denoise', action='store_true')

    # VR subcommand
    vr_parser = subparsers.add_parser('vr', parents=[common_parser])
    vr_parser.add_argument('--model-name', choices=VR_ARCH_MODELS, required=True)
    vr_parser.add_argument('--window-size', type=int, default=512)
    vr_parser.add_argument('--aggression', type=int, default=5)
    vr_parser.add_argument('--tta', action='store_true')
    vr_parser.add_argument('--post-process', action='store_true')
    vr_parser.add_argument('--post-process-threshold', type=float, default=0.2)
    vr_parser.add_argument('--high-end-process', action='store_true')

    # Demucs subcommand
    demucs_parser = subparsers.add_parser('demucs', parents=[common_parser])
    demucs_parser.add_argument('--model-name', choices=DEMUCS_MODELS, required=True)
    demucs_parser.add_argument('--seg-size', type=int, default=256)
    demucs_parser.add_argument('--shifts', type=int, default=1)
    demucs_parser.add_argument('--overlap', type=float, default=0.25)
    demucs_parser.add_argument('--segments-enabled', action='store_true')

    args = parser.parse_args()

    # Handle input
    if args.input.startswith('http'):
        input_file = downloader(args.input)
        if not input_file:
            raise ValueError("Failed to download audio from URL")
    else:
        input_file = args.input
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")

    # Prepare output directory
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    out_dir = os.path.join(args.output_dir, base_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Initialize separator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=args.model_dir,
        output_dir=out_dir,
        output_format=args.output_format,
        normalization_threshold=args.norm_thresh,
        amplification_threshold=args.amp_thresh,
        use_autocast=(device == "cuda")
    )

    # Model-specific parameters
    model_filename = None
    if args.model_type == 'roformer':
        model_filename = ROFORMER_MODELS[args.model_name]
        separator.mdxc_params = {
            "segment_size": args.seg_size,
            "override_model_segment_size": args.override_seg_size,
            "batch_size": args.batch_size,
            "overlap": args.overlap,
            "pitch_shift": args.pitch_shift
        }
    elif args.model_type == 'mdx23c':
        model_filename = args.model_name
        separator.mdxc_params = {
            "segment_size": args.seg_size,
            "override_model_segment_size": args.override_seg_size,
            "batch_size": args.batch_size,
            "overlap": args.overlap,
            "pitch_shift": args.pitch_shift
        }
    elif args.model_type == 'mdx':
        model_filename = args.model_name
        separator.mdx_params = {
            "hop_length": args.hop_length,
            "segment_size": args.seg_size,
            "overlap": args.overlap,
            "batch_size": args.batch_size,
            "enable_denoise": args.denoise
        }
    elif args.model_type == 'vr':
        model_filename = args.model_name
        separator.vr_params = {
            "window_size": args.window_size,
            "aggression": args.aggression,
            "enable_tta": args.tta,
            "enable_post_process": args.post_process,
            "post_process_threshold": args.post_process_threshold,
            "high_end_process": args.high_end_process,
            "batch_size": args.batch_size
        }
    elif args.model_type == 'demucs':
        model_filename = args.model_name
        separator.demucs_params = {
            "segment_size": args.seg_size,
            "shifts": args.shifts,
            "overlap": args.overlap,
            "segments_enabled": args.segments_enabled
        }

    # Process separation
    try:
        separator.load_model(model_filename)
        separator.separate(input_file)
        print(f"✅ Separation complete! Results saved to: {out_dir}")
    except Exception as e:
        print(f"❌ Error during separation: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
