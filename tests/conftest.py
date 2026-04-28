"""
Fixtures compartilhadas entre todos os módulos de teste.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ------------------------------------------------------------------
# Resultado de transcrição bem-sucedida (padrão para os testes)
# ------------------------------------------------------------------

@pytest.fixture
def sample_result():
    return {
        "success": True,
        "text": "Olá mundo. Este é um teste de transcrição.",
        "segments": [
            {"start": 0.0,  "end": 2.5,  "text": "Olá mundo."},
            {"start": 2.5,  "end": 5.0,  "text": "Este é um teste de transcrição."},
        ],
        "error": None,
        "metadata": {
            "file_name": "audio_teste.mp3",
            "file_path": "data/input/audio_teste.mp3",
            "language": "pt",
            "language_probability": 0.99,
            "duration": 5.0,
        },
    }


@pytest.fixture
def failed_result():
    return {
        "success": False,
        "text": None,
        "segments": [],
        "error": "Erro simulado na transcrição.",
    }


# ------------------------------------------------------------------
# Mock do WhisperModel
# ------------------------------------------------------------------

@pytest.fixture
def mock_whisper_segment():
    seg = MagicMock()
    seg.start = 0.0
    seg.end = 2.5
    seg.text = " Olá mundo."
    return seg


@pytest.fixture
def mock_whisper_info():
    info = MagicMock()
    info.language = "pt"
    info.language_probability = 0.99
    info.duration = 5.0
    return info


@pytest.fixture
def mock_model(mock_whisper_segment, mock_whisper_info):
    model = MagicMock()
    model.transcribe.return_value = ([mock_whisper_segment], mock_whisper_info)
    return model


# ------------------------------------------------------------------
# TranscriptionService com modelo mockado
# ------------------------------------------------------------------

@pytest.fixture
def service(mock_model):
    with patch("app.services.transcription_service.WhisperModel", return_value=mock_model):
        from app.services.transcription_service import TranscriptionService
        return TranscriptionService()


# ------------------------------------------------------------------
# TranscriptionExporter
# ------------------------------------------------------------------

@pytest.fixture
def exporter():
    from app.services.transcription_exporter import TranscriptionExporter
    return TranscriptionExporter()


# ------------------------------------------------------------------
# Diretório temporário de output
# ------------------------------------------------------------------

@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return d


# ------------------------------------------------------------------
# Diretório temporário de input com arquivos de mídia fictícios
# ------------------------------------------------------------------

@pytest.fixture
def input_dir(tmp_path):
    d = tmp_path / "input"
    d.mkdir()
    (d / "audio_a.mp3").write_bytes(b"fake audio a")
    (d / "audio_b.wav").write_bytes(b"fake audio b")
    return d


@pytest.fixture
def input_dir_mixed(tmp_path):
    """Input com arquivos suportados e não suportados."""
    d = tmp_path / "input_mixed"
    d.mkdir()
    (d / "audio.mp3").write_bytes(b"fake")
    (d / "imagem.png").write_bytes(b"fake")
    (d / "documento.pdf").write_bytes(b"fake")
    return d


@pytest.fixture
def input_dir_empty(tmp_path):
    """Input sem nenhum arquivo de mídia suportado."""
    d = tmp_path / "input_empty"
    d.mkdir()
    (d / "readme.txt").write_text("nada aqui")
    return d
