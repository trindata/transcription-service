"""
Testes para TranscriptionExporter.

Cobertura:
  save()       — formato inválido, resultado com erro, cada writer (txt/srt/vtt/json/docx),
                 tolerância a case e ponto, criação automática de diretório
  save_all()   — todos os formatos, subconjunto, resultado com erro, falha parcial
  _fmt_timestamp() — múltiplas entradas incluindo borda de hora
  _resolve_stem()  — via file_name, via metadata, fallback
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ==================================================================
# save()
# ==================================================================

class TestSave:

    def test_formato_invalido(self, exporter, sample_result, output_dir):
        out = exporter.save(sample_result, "mp3", str(output_dir))
        assert out["success"] is False
        assert "não suportado" in out["error"]
        assert out["file_path"] is None

    def test_resultado_com_erro(self, exporter, failed_result, output_dir):
        out = exporter.save(failed_result, "txt", str(output_dir))
        assert out["success"] is False
        assert out["file_path"] is None

    def test_formato_case_insensitive(self, exporter, sample_result, output_dir):
        """'SRT' e '.srt' devem funcionar igual a 'srt'."""
        out_upper = exporter.save(sample_result, "SRT", str(output_dir))
        out_dot   = exporter.save(sample_result, ".srt", str(output_dir))
        assert out_upper["success"] is True
        assert out_dot["success"] is True

    def test_cria_diretorio_automaticamente(self, exporter, sample_result, tmp_path):
        novo_dir = tmp_path / "novo" / "subdir"
        out = exporter.save(sample_result, "txt", str(novo_dir))
        assert out["success"] is True
        assert novo_dir.exists()

    # --- Writers individuais ---

    def test_save_txt(self, exporter, sample_result, output_dir):
        out = exporter.save(sample_result, "txt", str(output_dir))
        assert out["success"] is True
        content = Path(out["file_path"]).read_text(encoding="utf-8")
        assert "Olá mundo" in content

    def test_save_srt(self, exporter, sample_result, output_dir):
        out = exporter.save(sample_result, "srt", str(output_dir))
        assert out["success"] is True
        content = Path(out["file_path"]).read_text(encoding="utf-8")
        assert "1\n" in content              # numeração SRT
        assert "-->" in content              # timestamps
        assert "," in content               # separador SRT (vírgula, não ponto)

    def test_save_vtt(self, exporter, sample_result, output_dir):
        out = exporter.save(sample_result, "vtt", str(output_dir))
        assert out["success"] is True
        content = Path(out["file_path"]).read_text(encoding="utf-8")
        assert content.startswith("WEBVTT")  # cabeçalho obrigatório
        assert "." in content.split("-->")[0].split(":")[-1]  # separador VTT (ponto)

    def test_save_json(self, exporter, sample_result, output_dir):
        out = exporter.save(sample_result, "json", str(output_dir))
        assert out["success"] is True
        data = json.loads(Path(out["file_path"]).read_text(encoding="utf-8"))
        assert "metadata" in data
        assert "text" in data
        assert "segments" in data
        assert "success" not in data  # campo interno não deve vazar

    def test_save_docx(self, exporter, sample_result, output_dir):
        pytest.importorskip("docx", reason="python-docx não instalado")
        out = exporter.save(sample_result, "docx", str(output_dir))
        assert out["success"] is True
        assert Path(out["file_path"]).suffix == ".docx"

    def test_save_docx_sem_dependencia(self, exporter, sample_result, output_dir):
        """Sem python-docx instalado, deve retornar erro descritivo em vez de travar."""
        with patch.dict("sys.modules", {"docx": None}):
            out = exporter.save(sample_result, "docx", str(output_dir))
        assert out["success"] is False
        assert out["file_path"] is None

    def test_nome_arquivo_derivado_do_metadata(self, exporter, sample_result, output_dir):
        out = exporter.save(sample_result, "txt", str(output_dir))
        assert "audio_teste" in out["file_path"]

    def test_nome_arquivo_customizado(self, exporter, sample_result, output_dir):
        out = exporter.save(sample_result, "txt", str(output_dir), file_name="meu_audio")
        assert "meu_audio" in out["file_path"]

    def test_file_path_retornado_existe(self, exporter, sample_result, output_dir):
        out = exporter.save(sample_result, "txt", str(output_dir))
        assert Path(out["file_path"]).exists()


# ==================================================================
# save_all()
# ==================================================================

class TestSaveAll:

    def test_resultado_com_erro(self, exporter, failed_result, output_dir):
        out = exporter.save_all(failed_result, output_dir=str(output_dir))
        assert out["success"] is False
        assert out["results"] == {}

    def test_subconjunto_de_formatos(self, exporter, sample_result, output_dir):
        out = exporter.save_all(sample_result, formats=["txt", "json"], output_dir=str(output_dir))
        assert set(out["results"].keys()) == {"txt", "json"}

    def test_todos_formatos_por_default(self, exporter, sample_result, output_dir):
        out = exporter.save_all(sample_result, output_dir=str(output_dir))
        from app.services.transcription_exporter import TranscriptionExporter
        assert set(out["results"].keys()) == TranscriptionExporter.SUPPORTED_FORMATS

    def test_success_true_se_ao_menos_um_funcionou(self, exporter, sample_result, output_dir):
        with patch.object(exporter, "save") as mock_save:
            mock_save.side_effect = lambda r, fmt, *a, **kw: (
                {"success": True,  "file_path": "x.txt", "error": None} if fmt == "txt"
                else {"success": False, "file_path": None, "error": "erro"}
            )
            out = exporter.save_all(sample_result, formats=["txt", "srt"], output_dir=str(output_dir))
        assert out["success"] is True

    def test_success_false_se_todos_falharam(self, exporter, sample_result, output_dir):
        with patch.object(exporter, "save", return_value={"success": False, "file_path": None, "error": "x"}):
            out = exporter.save_all(sample_result, formats=["txt", "srt"], output_dir=str(output_dir))
        assert out["success"] is False

    def test_errors_none_quando_tudo_ok(self, exporter, sample_result, output_dir):
        out = exporter.save_all(sample_result, formats=["txt"], output_dir=str(output_dir))
        assert out["errors"] is None

    def test_errors_preenchido_em_falha_parcial(self, exporter, sample_result, output_dir):
        with patch.object(exporter, "save") as mock_save:
            mock_save.side_effect = lambda r, fmt, *a, **kw: (
                {"success": True,  "file_path": "x.txt", "error": None} if fmt == "txt"
                else {"success": False, "file_path": None, "error": "falhou"}
            )
            out = exporter.save_all(sample_result, formats=["txt", "srt"], output_dir=str(output_dir))
        assert "srt" in out["errors"]


# ==================================================================
# _fmt_timestamp()
# ==================================================================

class TestFmtTimestamp:

    @pytest.mark.parametrize("seconds, sep, expected", [
        (0.0,      ",", "00:00:00,000"),
        (1.5,      ",", "00:00:01,500"),
        (59.999,   ",", "00:00:59,999"),
        (60.0,     ",", "00:01:00,000"),
        (3599.0,   ",", "00:59:59,000"),
        (3600.0,   ",", "01:00:00,000"),
        (3723.456, ",", "01:02:03,456"),
        (3723.456, ".", "01:02:03.456"),  # VTT
    ])
    def test_conversao(self, exporter, seconds, sep, expected):
        from app.services.transcription_exporter import TranscriptionExporter
        assert TranscriptionExporter._fmt_timestamp(seconds, sep) == expected


# ==================================================================
# _resolve_stem()
# ==================================================================

class TestResolveStem:

    def test_file_name_explicito(self, exporter):
        from app.services.transcription_exporter import TranscriptionExporter
        assert TranscriptionExporter._resolve_stem({}, "meu_audio") == "meu_audio"

    def test_file_name_com_extensao(self, exporter):
        from app.services.transcription_exporter import TranscriptionExporter
        assert TranscriptionExporter._resolve_stem({}, "meu_audio.mp3") == "meu_audio"

    def test_via_metadata(self, exporter, sample_result):
        from app.services.transcription_exporter import TranscriptionExporter
        assert TranscriptionExporter._resolve_stem(sample_result, None) == "audio_teste"

    def test_fallback_sem_metadata(self, exporter):
        from app.services.transcription_exporter import TranscriptionExporter
        assert TranscriptionExporter._resolve_stem({}, None) == "transcricao"
