"""
Runner unificado de transcrição.

Modos de uso:
  python run.py                                      # batch em data/input/ com todos os defaults
  python run.py --input data/input/aula.mp4          # arquivo único
  python run.py --input data/recordings/             # batch em diretório alternativo

  # Sobrescrevendo defaults:
  python run.py --output exports/
  python run.py --input aula.mp4 --language en       # força idioma
  python run.py --input aula.mp4                     # detecta automaticamente
  python run.py --formats srt json
  python run.py --input aula.mp4 --output exports/ --language en --formats srt vtt

  # Configuração do modelo:
  python run.py --model_size large-v3
  python run.py --model_size medium --device cuda --compute_type float16
"""

import argparse
from pathlib import Path

from app.services.transcription_service import TranscriptionService
from app.services.transcription_exporter import TranscriptionExporter


DEFAULT_INPUT_DIR    = "data/input"
DEFAULT_OUTPUT_DIR   = "data/output"
DEFAULT_LANGUAGE     = None
DEFAULT_FORMATS      = ["srt", "vtt", "json", "docx", "txt"]
DEFAULT_MODEL_SIZE   = "base"
DEFAULT_DEVICE       = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"


def _export_result(exporter, file_name, result, output_dir, formats):
    """Imprime o cabeçalho do arquivo e exporta os formatos. Usado por single e batch."""
    print(f"[ {file_name} ]")

    if not result["success"]:
        print(f"  Transcrição falhou: {result['error']}\n")
        return

    all_out = exporter.save_all(result, formats=formats, output_dir=output_dir)

    for fmt, info in all_out["results"].items():
        status = "OK" if info["success"] else f"ERRO — {info['error']}"
        print(f"  {fmt:>5}: {status}")
        if info["success"]:
            print(f"         {info['file_path']}")

    print()


def run_single(service, exporter, input_path, output_dir, language, formats):
    result = service.transcribe(input_path, language=language)
    _export_result(exporter, Path(input_path).name, result, output_dir, formats)


def run_batch(service, exporter, input_dir, output_dir, language, formats):
    batch = service.transcribe_batch(input_dir, language=language)

    if not batch["success"]:
        print("Erro no batch:", batch["error"])
        return

    summary = batch["summary"]
    print(f"\nArquivos encontrados: {summary['total_files']}")
    print(f"Transcritos com sucesso: {summary['success_count']}")
    print(f"Erros de transcrição: {summary['error_count']}\n")

    for item in batch["results"]:
        _export_result(exporter, item["file"], item["result"], output_dir, formats)


def build_parser():
    parser = argparse.ArgumentParser(description="Runner de transcrição de áudio/vídeo.")
    parser.add_argument("--input",        default=None,                 help="Caminho do arquivo ou diretório de entrada.")
    parser.add_argument("--output",       default=DEFAULT_OUTPUT_DIR,   help="Diretório de saída (padrão: data/output).")
    parser.add_argument("--language",     default=DEFAULT_LANGUAGE,     help="Idioma da transcrição. Se não informado detecta automaticamente (exemplo: pt, en).")
    parser.add_argument("--formats",      default=None, nargs="+",      help="Formatos de exportação (padrão: srt vtt json docx).")
    parser.add_argument("--model_size",   default=DEFAULT_MODEL_SIZE,   help="Tamanho do modelo Whisper (padrão: base). Opções: tiny base small medium large-v1 large-v2 large-v3.")
    parser.add_argument("--device",       default=DEFAULT_DEVICE,       help="Device de inferência (padrão: cpu). Opções: cpu cuda.")
    parser.add_argument("--compute_type", default=DEFAULT_COMPUTE_TYPE, help="Tipo de computação (padrão: int8). Opções: int8 float16 int8_float16.")
    return parser


def main():
    args = build_parser().parse_args()

    formats = args.formats or DEFAULT_FORMATS

    service  = TranscriptionService(model_size=args.model_size, device=args.device, compute_type=args.compute_type)
    exporter = TranscriptionExporter()

    if args.input is None:
        run_batch(service, exporter, DEFAULT_INPUT_DIR, args.output, args.language, formats)
        return

    input_path = Path(args.input)

    if input_path.is_dir():
        run_batch(service, exporter, str(input_path), args.output, args.language, formats)
    else:
        run_single(service, exporter, str(input_path), args.output, args.language, formats)


if __name__ == "__main__":
    main()