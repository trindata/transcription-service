from pathlib import Path
from typing import Optional, Dict, Any, List

from faster_whisper import WhisperModel


class TranscriptionService:
    """
    Transcreve arquivos de áudio e vídeo usando o modelo Whisper via faster-whisper.

    O modelo é carregado uma única vez na inicialização e reutilizado em todas
    as transcrições. Todos os métodos retornam dicts estruturados com "success"
    e "error" — nunca lançam exceções para o caller.

    Formatos de entrada suportados:
        .mp3, .wav, .m4a, .mp4, .flac, .ogg

    Uso:
        service = TranscriptionService(model_size="small", device="cpu")
        result  = service.transcribe("audio.mp3", language="pt")
        batch   = service.transcribe_batch("data/input/", language="pt")
    """

    SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg"}

    def __init__(
        self,
        model_size: str = "base",   # Opções: tiny, base, small, medium, large-v1, large-v2, large-v3
        device: str = "cpu",        # Opções: "cpu", "cuda" (NVIDIA) — AMD não suporta cuda
        compute_type: str = "int8"  # Opções: "int8" (cpu), "float16" / "int8_float16" (cuda)
    ):
        """
        Inicializa o serviço carregando o modelo Whisper.

        Args:
            model_size:   Tamanho do modelo. Modelos maiores são mais precisos
                          e mais lentos. Opções: tiny, base, small, medium,
                          large-v1, large-v2, large-v3.
            device:       Device de inferência. "cuda" exige GPU NVIDIA.
                          Opções: "cpu", "cuda".
            compute_type: Precisão numérica da inferência. "int8" é recomendado
                          para CPU. "float16" ou "int8_float16" para CUDA.
                          Opções: "int8", "float16", "int8_float16".
        """
        # Carrega o modelo na inicialização — custo alto feito uma vez, reutilizado em todas as transcrições
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe(self, file_path: str, language: Optional[str] = "pt") -> Dict[str, Any]:
        """
        Transcreve um arquivo de áudio ou vídeo.

        Valida o arquivo antes de chamar o modelo. Em caso de falha — arquivo
        inexistente, formato inválido ou erro no modelo — retorna success=False
        com a descrição do erro, sem lançar exceção.

        Args:
            file_path: Caminho para o arquivo de mídia.
            language:  Código do idioma (ex: "pt", "en"). Se None, o modelo
                       detecta automaticamente, com custo adicional de tempo.

        Returns:
            {
                "success":  bool,
                "text":     str | None,       # Transcrição completa
                "segments": list[dict],        # [{start, end, text}, ...]
                "error":    str | None,
                "metadata": {
                    "file_name":            str,
                    "file_path":            str,
                    "language":             str | None,
                    "language_probability": float | None,
                    "duration":             float | None,
                }
            }
        """
        
        path = Path(file_path)

        # Validações fail-fast — retornam antes de chegar ao modelo
        if not path.exists():
            return {
                "success": False,
                "text": None,
                "segments": [],
                "error": f"Arquivo não encontrado: {file_path}"
            }

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return {
                "success": False,
                "text": None,
                "segments": [],
                "error": f"Formato não suportado: {path.suffix}",
                "supported_formats": list(self.SUPPORTED_EXTENSIONS)
            }
            
        info = None  # Variável para armazenar metadados do áudio, mesmo em caso de erro

        try:
            # model.transcribe() retorna um gerador de segmentos + objeto info com metadados do áudio
            segments, info = self.model.transcribe(str(path), language=language)

            segment_list = []
            full_text = []

            # Itera o gerador — os segmentos são processados sob demanda, não todos de uma vez
            for segment in segments:
                text = segment.text.strip()

                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": text
                })

                # Segmentos vazios são ignorados no texto completo mas preservados na lista de segmentos
                if text:
                    full_text.append(text)

            return {
                "success": True,
                "text": " ".join(full_text).strip(),
                "segments": segment_list,
                "error": None,
                "metadata": {
                    "file_name": path.name,
                    "file_path": str(path),
                    # getattr com fallback None — campos podem não existir dependendo da versão do modelo
                    "language": getattr(info, "language", None),
                    "language_probability": getattr(info, "language_probability", None),
                    "duration": getattr(info, "duration", None)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "text": None,
                "segments": [],
                "error": f"Erro ao transcrever arquivo: {str(e)}",
                "metadata": {
                    "file_name": path.name,
                    "file_path": str(path),
                    "language": getattr(info, "language", None),
                    "language_probability": getattr(info, "language_probability", None),
                    "duration": getattr(info, "duration", None)
                }
            }

    def transcribe_batch(
        self,
        input_dir: str = "data/input",
        language: Optional[str] = "pt",
    ) -> Dict[str, Any]:
        """
        Transcreve todos os arquivos de mídia suportados em um diretório.

        Arquivos não suportados são ignorados silenciosamente. Falhas em
        arquivos individuais não interrompem o lote — cada resultado é
        encapsulado separadamente. O campo "success" no retorno indica que
        o batch executou, não que todos os arquivos foram transcritos com
        sucesso — verifique "summary" para o detalhamento.

        Args:
            input_dir: Caminho do diretório de entrada.
            language:  Código do idioma (ex: "pt", "en"). Se None, o modelo
                       detecta automaticamente para cada arquivo.

        Returns:
            {
                "success": bool,  # False se: diretório não encontrado,
                                  # caminho não é um diretório, ou nenhum
                                  # arquivo de mídia suportado encontrado.
                "results":  list[dict],     # [{"file": str, "result": dict}, ...]
                "summary": {
                    "total_files":   int,
                    "success_count": int,
                    "error_count":   int,
                },
                "error": str | None
            }
        """
        
        input_path = Path(input_dir)

        # Validações do diretório — fail-fast antes de qualquer transcrição
        if not input_path.exists():
            return {
                "success": False,
                "results": [],
                "error": f"Diretório não encontrado: {input_dir}"
            }

        if not input_path.is_dir():
            return {
                "success": False,
                "results": [],
                "error": f"O caminho informado não é um diretório: {input_dir}"
            }

        # Filtra apenas arquivos com extensões suportadas — ignora subdiretórios e outros arquivos
        files = [
            file for file in input_path.iterdir()
            if file.is_file() and file.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not files:
            return {
                "success": False,
                "results": [],
                "error": "Nenhum arquivo de mídia suportado encontrado no diretório."
            }

        results = []

        # Itera sem interrupção — falha em um arquivo não cancela os demais
        for file in files:
            result = self.transcribe(str(file), language=language)

            item = {
                "file": file.name,
                "result": result
            }

            results.append(item)
            
        # Contadores derivados dos resultados — calculados ao final, não incrementados no loop
        success_count = sum(1 for item in results if item["result"].get("success"))
        error_count = len(results) - success_count

        return {
            "success": True,  # True mesmo com erros parciais — indica que o batch executou
            "results": results,
            "summary": {
                "total_files": len(results),
                "success_count": success_count,
                "error_count": error_count
            },
            "error": None
        }