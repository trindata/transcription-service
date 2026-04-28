"""
Testes para TranscriptionService.

Cobertura:
  transcribe()      — arquivo não encontrado, extensão inválida, sucesso,
                      exceção no modelo, áudio sem segmentos
  transcribe_batch() — dir inválido, não é dir, sem arquivos suportados,
                       sucesso parcial, ignora arquivos não suportados
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ==================================================================
# transcribe()
# ==================================================================

class TestTranscribe:

    def test_arquivo_nao_encontrado(self, service):
        result = service.transcribe("nao_existe.mp3")
        assert result["success"] is False
        assert "não encontrado" in result["error"]
        assert result["text"] is None
        assert result["segments"] == []

    def test_extensao_nao_suportada(self, service, tmp_path):
        f = tmp_path / "audio.pdf"
        f.write_bytes(b"fake")
        result = service.transcribe(str(f))
        assert result["success"] is False
        assert "Formato não suportado" in result["error"]

    def test_extensao_maiuscula_nao_suportada(self, service, tmp_path):
        """Extensão em caixa alta não deve passar na validação."""
        f = tmp_path / "audio.PDF"
        f.write_bytes(b"fake")
        result = service.transcribe(str(f))
        assert result["success"] is False

    def test_sucesso_retorna_estrutura_completa(self, service, tmp_path):
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"fake audio")
        result = service.transcribe(str(f))
        assert result["success"] is True
        assert isinstance(result["text"], str)
        assert isinstance(result["segments"], list)
        assert result["error"] is None
        assert "metadata" in result

    def test_sucesso_metadados_preenchidos(self, service, tmp_path):
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"fake audio")
        result = service.transcribe(str(f))
        meta = result["metadata"]
        assert meta["file_name"] == "audio.mp3"
        assert meta["language"] == "pt"
        assert meta["language_probability"] == 0.99
        assert meta["duration"] == 5.0

    def test_excecao_no_modelo_retorna_erro(self, service, tmp_path):
        """Se o modelo lançar exceção, transcribe() deve retornar success=False."""
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"fake audio")
        service.model.transcribe.side_effect = RuntimeError("modelo quebrou")
        result = service.transcribe(str(f))
        assert result["success"] is False
        assert "modelo quebrou" in result["error"]

    def test_audio_sem_segmentos(self, service, tmp_path):
        """Modelo retorna lista vazia de segmentos (silêncio)."""
        f = tmp_path / "silencio.wav"
        f.write_bytes(b"fake")
        service.model.transcribe.return_value = ([], MagicMock(
            language="pt", language_probability=0.5, duration=3.0
        ))
        result = service.transcribe(str(f))
        assert result["success"] is True
        assert result["text"] == ""
        assert result["segments"] == []

    def test_segmentos_vazios_nao_entram_no_texto(self, service, tmp_path):
        """Segmentos com text vazio são preservados na lista mas ignorados no texto."""
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"fake")
        seg_vazio = MagicMock(start=0.0, end=1.0, text="   ")
        seg_real  = MagicMock(start=1.0, end=2.0, text=" Olá.")
        service.model.transcribe.return_value = ([seg_vazio, seg_real], MagicMock(
            language="pt", language_probability=0.99, duration=2.0
        ))
        result = service.transcribe(str(f))
        assert result["text"] == "Olá."
        assert len(result["segments"]) == 2

    @pytest.mark.parametrize("ext", [".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg"])
    def test_extensoes_suportadas(self, service, tmp_path, ext):
        f = tmp_path / f"audio{ext}"
        f.write_bytes(b"fake audio")
        result = service.transcribe(str(f))
        assert result["success"] is True


# ==================================================================
# transcribe_batch()
# ==================================================================

class TestTranscribeBatch:

    def test_dir_nao_encontrado(self, service):
        result = service.transcribe_batch("caminho/inexistente")
        assert result["success"] is False
        assert "não encontrado" in result["error"]
        assert result["results"] == []

    def test_caminho_nao_e_diretorio(self, service, tmp_path):
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"fake")
        result = service.transcribe_batch(str(f))
        assert result["success"] is False
        assert "não é um diretório" in result["error"]

    def test_sem_arquivos_suportados(self, service, input_dir_empty):
        result = service.transcribe_batch(str(input_dir_empty))
        assert result["success"] is False
        assert "Nenhum arquivo" in result["error"]

    def test_ignora_arquivos_nao_suportados(self, service, input_dir_mixed):
        """PNG e PDF não devem entrar no batch."""
        result = service.transcribe_batch(str(input_dir_mixed))
        assert result["success"] is True
        assert result["summary"]["total_files"] == 1  # só o .mp3

    def test_batch_sucesso_retorna_estrutura(self, service, input_dir):
        result = service.transcribe_batch(str(input_dir))
        assert result["success"] is True
        assert "results" in result
        assert "summary" in result
        assert result["error"] is None

    def test_batch_summary_correto(self, service, input_dir):
        result = service.transcribe_batch(str(input_dir))
        summary = result["summary"]
        assert summary["total_files"] == 2
        assert summary["success_count"] == 2
        assert summary["error_count"] == 0

    def test_falha_parcial_nao_cancela_batch(self, service, input_dir):
        """Se o modelo falhar no primeiro arquivo, o segundo ainda deve ser processado."""
        call_count = 0
        original = service.model.transcribe

        def model_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("falhou no primeiro")
            return original.return_value

        service.model.transcribe.side_effect = model_side_effect
        result = service.transcribe_batch(str(input_dir))

        assert result["success"] is True
        assert result["summary"]["total_files"] == 2
        assert result["summary"]["error_count"] == 1
        assert result["summary"]["success_count"] == 1

    def test_batch_success_true_com_erros_parciais(self, service, input_dir):
        """success=True no nível do batch mesmo que alguns arquivos falhem."""
        service.model.transcribe.side_effect = RuntimeError("erro geral")
        result = service.transcribe_batch(str(input_dir))
        assert result["success"] is True  # batch executou
        assert result["summary"]["error_count"] == 2

    def test_cada_item_tem_file_e_result(self, service, input_dir):
        result = service.transcribe_batch(str(input_dir))
        for item in result["results"]:
            assert "file" in item
            assert "result" in item
