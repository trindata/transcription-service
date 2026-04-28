"""
Testes para o CLI run.py.

Cobertura:
  - Parsing de todos os argumentos
  - Defaults corretos
  - Roteamento: sem args → batch, --input dir → batch, --input file → single
  - Integração ponta a ponta com mocks (sem modelo real)
"""

import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


# Stub do TranscriptionService para os testes de integração
def make_service_stub(batch_success=True, transcribe_success=True):
    stub = MagicMock()

    stub.transcribe.return_value = {
        "success": transcribe_success,
        "text": "Transcrição de teste.",
        "segments": [{"start": 0.0, "end": 2.0, "text": "Transcrição de teste."}],
        "error": None if transcribe_success else "erro simulado",
        "metadata": {
            "file_name": "audio.mp3",
            "file_path": "data/input/audio.mp3",
            "language": "pt",
            "language_probability": 0.99,
            "duration": 2.0,
        },
    }

    stub.transcribe_batch.return_value = {
        "success": batch_success,
        "results": [
            {
                "file": "audio.mp3",
                "result": stub.transcribe.return_value,
            }
        ],
        "summary": {"total_files": 1, "success_count": 1, "error_count": 0},
        "error": None if batch_success else "diretório inválido",
    }

    return stub


def make_exporter_stub():
    stub = MagicMock()
    stub.save_all.return_value = {
        "success": True,
        "results": {
            "txt":  {"success": True, "file_path": "data/output/audio.txt",  "error": None},
            "json": {"success": True, "file_path": "data/output/audio.json", "error": None},
        },
        "errors": None,
    }
    return stub


# ==================================================================
# Parsing de argumentos
# ==================================================================

class TestArgParsing:

    def _parse(self, argv):
        """Importa main e intercepta o parse sem executar nada."""
        import importlib, sys
        sys.argv = ["run.py"] + argv

        with patch("app.services.transcription_service.TranscriptionService"), \
             patch("app.services.transcription_exporter.TranscriptionExporter"):
            import run
            importlib.reload(run)

            parser = run.build_parser()
            return parser.parse_args(argv)

    def test_defaults(self, tmp_path):
        import importlib, sys
        sys.argv = ["run.py"]
        with patch("app.services.transcription_service.TranscriptionService"), \
             patch("app.services.transcription_exporter.TranscriptionExporter"):
            import run
            importlib.reload(run)
            args = run.build_parser().parse_args([])

        assert args.input is None
        assert args.output == "data/output"
        assert args.language == "pt"
        assert args.formats is None
        assert args.model_size == "base"
        assert args.device == "cpu"
        assert args.compute_type == "int8"

    def test_todos_os_argumentos(self):
        with patch("app.services.transcription_service.TranscriptionService"), \
             patch("app.services.transcription_exporter.TranscriptionExporter"):
            import run
            args = run.build_parser().parse_args([
                "--input",        "data/input/aula.mp4",
                "--output",       "exports/",
                "--language",     "en",
                "--formats",      "srt", "vtt",
                "--model_size",   "large-v3",
                "--device",       "cuda",
                "--compute_type", "float16",
            ])

        assert args.input == "data/input/aula.mp4"
        assert args.output == "exports/"
        assert args.language == "en"
        assert args.formats == ["srt", "vtt"]
        assert args.model_size == "large-v3"
        assert args.device == "cuda"
        assert args.compute_type == "float16"


# ==================================================================
# Roteamento
# ==================================================================

class TestRoteamento:

    def test_sem_input_chama_batch_no_dir_padrao(self, tmp_path):
        service  = make_service_stub()
        exporter = make_exporter_stub()

        with patch("run.TranscriptionService", return_value=service), \
             patch("run.TranscriptionExporter", return_value=exporter):
            import run, sys
            sys.argv = ["run.py"]
            run.main()

        service.transcribe_batch.assert_called_once()
        call_args = service.transcribe_batch.call_args
        assert call_args[0][0] == "data/input"

    def test_input_arquivo_chama_single(self, tmp_path):
        audio = tmp_path / "aula.mp4"
        audio.write_bytes(b"fake")

        service  = make_service_stub()
        exporter = make_exporter_stub()

        with patch("run.TranscriptionService", return_value=service), \
             patch("run.TranscriptionExporter", return_value=exporter):
            import run, sys
            sys.argv = ["run.py", "--input", str(audio)]
            run.main()

        service.transcribe.assert_called_once_with(str(audio), language="pt")
        service.transcribe_batch.assert_not_called()

    def test_input_diretorio_chama_batch(self, tmp_path):
        service  = make_service_stub()
        exporter = make_exporter_stub()

        with patch("run.TranscriptionService", return_value=service), \
             patch("run.TranscriptionExporter", return_value=exporter):
            import run, sys
            sys.argv = ["run.py", "--input", str(tmp_path)]
            run.main()

        service.transcribe_batch.assert_called_once_with(str(tmp_path), language="pt")
        service.transcribe.assert_not_called()

    def test_language_repassada_ao_service(self, tmp_path):
        audio = tmp_path / "aula.mp4"
        audio.write_bytes(b"fake")

        service  = make_service_stub()
        exporter = make_exporter_stub()

        with patch("run.TranscriptionService", return_value=service), \
             patch("run.TranscriptionExporter", return_value=exporter):
            import run, sys
            sys.argv = ["run.py", "--input", str(audio), "--language", "en"]
            run.main()

        service.transcribe.assert_called_once_with(str(audio), language="en")

    def test_batch_error_imprime_e_retorna(self, tmp_path, capsys):
        service  = make_service_stub(batch_success=False)
        exporter = make_exporter_stub()

        with patch("run.TranscriptionService", return_value=service), \
             patch("run.TranscriptionExporter", return_value=exporter):
            import run, sys
            sys.argv = ["run.py"]
            run.main()

        captured = capsys.readouterr()
        assert "Erro no batch" in captured.out
        exporter.save_all.assert_not_called()

    def test_transcricao_falha_pula_exportacao(self, tmp_path, capsys):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake")

        service  = make_service_stub(transcribe_success=False)
        exporter = make_exporter_stub()

        with patch("run.TranscriptionService", return_value=service), \
             patch("run.TranscriptionExporter", return_value=exporter):
            import run, sys
            sys.argv = ["run.py", "--input", str(audio)]
            run.main()

        exporter.save_all.assert_not_called()
        captured = capsys.readouterr()
        assert "falhou" in captured.out.lower()