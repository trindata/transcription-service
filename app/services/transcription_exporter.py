import json
from pathlib import Path
from typing import Any, Dict, List, Optional

class TranscriptionExporter:
    """
    Exporta resultados de transcrição para múltiplos formatos de arquivo.

    Recebe o dict retornado por TranscriptionService.transcribe() e delega
    a escrita para writers especializados por formato.

    Formatos suportados:
        txt  — texto puro
        srt  — legenda com timestamps (SubRip)
        vtt  — legenda para web (WebVTT)
        json — estrutura completa com metadados e segmentos
        docx — documento Word formatado com metadados e segmentos

    Uso:
        exporter = TranscriptionExporter()
        exporter.save(result, format="srt", output_dir="data/output")
        exporter.save_all(result, formats=["srt", "vtt", "json"])
    """

    SUPPORTED_FORMATS = {"txt", "srt", "vtt", "json", "docx"}

    # ------------------------------------------------------------------
    # Ponto de entrada principal
    # ------------------------------------------------------------------

    def save(
        self,
        result: Dict[str, Any],
        format: str,
        output_dir: str = "data/output",
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Orquestra a escrita para disco, delegando para o método _write_*.

        Args:
            result:     Dict retornado por TranscriptionService.transcribe()
            format:     Formato de saída: "txt" | "srt" | "vtt" | "json" | "docx"
            output_dir: Diretório de saída
            file_name:  Nome base do arquivo (sem extensão). Usa o nome
                        original do áudio se não informado.

        Returns:
            {"success": bool, "file_path": str | None, "error": str | None}
        """
        
        # Tolerância a diferentes inputs do usuário: "SRT", ".srt" e "srt" 
        fmt = format.lower().lstrip(".")

        # Saídas padronizadas para casos de erro
        if fmt not in self.SUPPORTED_FORMATS:
            return {
                "success": False,
                "file_path": None,
                "error": (
                    f"Formato '{fmt}' não suportado. "
                    f"Formatos disponíveis: {sorted(self.SUPPORTED_FORMATS)}"
                ),
            }

        if not result.get("success"):
            return {
                "success": False,
                "file_path": None,
                "error": "Não é possível exportar uma transcrição com erro.",
            }

        # Determina o nome base do arquivo (sem extensão)
        stem = self._resolve_stem(result, file_name)
        
        # Garante que o diretório de saída exista e constrói o caminho final
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dest = output_path / f"{stem}.{fmt}"

        # Dicionário de dispatch para os métodos de escrita
        writers = {
            "txt":  self._write_txt,
            "srt":  self._write_srt,
            "vtt":  self._write_vtt,
            "json": self._write_json,
            "docx": self._write_docx,
        }

        # Tenta escrever o arquivo usando o método correspondente ao formato
        try:
            writers[fmt](result, dest)
            return {"success": True, "file_path": str(dest), "error": None}
        
        # ImportError é capturado separado porque o _write_docx importa python-docx em tempo de execução
        except ImportError as e:
            return {
                "success": False,
                "file_path": None,
                "error": f"Dependência ausente para formato '{fmt}': {e}",
            }
            
        # Captura genérica para outros erros (ex: permissão de escrita, erros de codificação, etc)
        except Exception as e:
            return {
                "success": False,
                "file_path": None,
                "error": f"Erro ao exportar '{fmt}': {e}",
            }

    def save_all(
        self,
        result: Dict[str, Any],
        formats: Optional[List[str]] = None,
        output_dir: str = "data/output",
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Exporta para múltiplos formatos de uma vez.
        Orquestra múltiplas chamadas para save(), agregando os resultados.

        Args:
            result:     Dict retornado por TranscriptionService.transcribe()
            formats:    Lista de formatos. Usa todos os suportados se None.
                        Aceita: "txt", "srt", "vtt", "json", "docx"
            output_dir: Diretório de saída
            file_name:  Nome base do arquivo (sem extensão). Usa o nome
                        original do áudio se não informado.

        Returns:
            {
                "success": bool,          # True se ao menos um formato exportou
                "results": {
                    "srt": {"success": ..., "file_path": ..., "error": ...},
                    ...
                },
                "errors": dict | None     # None quando tudo funcionou
            }
        """
        
        # Validação prévia — falha rápido sem disparar o loop
        if not result.get("success"):
            return {
                "success": False,
                "results": {},
                "error": "Não é possível exportar uma transcrição com erro.",
            }
        
        # Tolerância a diferentes inputs do usuário: "SRT", ".srt" e "srt" 
        # Se formats for None, exporta para todos os formatos suportados
        targets = [f.lower().lstrip(".") for f in (formats or self.SUPPORTED_FORMATS)]
        
        # Acumuladores: resultados individuais e erros por formato
        results = {}
        errors  = {}
        
        # Itera sem interrupção — falha isolada não cancela os demais formatos
        for fmt in targets:
            outcome = self.save(result, fmt, output_dir, file_name)
            results[fmt] = outcome
            if not outcome["success"]:
                errors[fmt] = outcome["error"]

        # Identifica se ao menos um formato foi exportado com sucesso
        any_success = any(r["success"] for r in results.values())

        return {
            "success": any_success,
            "results": results,
            "errors":  errors or None,   # None quando tudo funcionou
        }

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def _write_txt(self, result: Dict[str, Any], dest: Path) -> None:
        # Fallback para string vazia garante escrita mesmo se "text" estiver ausente
        text = result.get("text", "")
        # UTF-8 explícito — evita quebra de acentos em sistemas com encoding padrão diferente
        dest.write_text(text, encoding="utf-8")

    def _write_srt(self, result: Dict[str, Any], dest: Path) -> None:
        # Fallback para lista vazia se "segments" não existir
        segments = result.get("segments", [])
        # Acumulador — o arquivo é construído em memória e escrito de uma vez
        lines = []

        # SRT é 1-indexed — start=1 garante que o primeiro bloco seja numerado como 1
        for i, seg in enumerate(segments, start=1):
            # separator="," é obrigatório no formato SRT (VTT usa ".")
            start = self._fmt_timestamp(seg["start"], separator=",")
            end   = self._fmt_timestamp(seg["end"],   separator=",")

            # Cada bloco SRT: índice / timestamps / texto / linha em branco
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(seg["text"].strip())   # .strip() remove espaços que o Whisper pode incluir
            lines.append("")                    # linha em branco entre blocos

        # Uma única operação de I/O para o arquivo inteiro
        dest.write_text("\n".join(lines), encoding="utf-8")

    def _write_vtt(self, result: Dict[str, Any], dest: Path) -> None:
        # Fallback para lista vazia se "segments" não existir
        segments = result.get("segments", [])
        # Cabeçalho WEBVTT obrigatório + linha em branco — sem isso nenhum player HTML5 reconhece o arquivo
        lines = ["WEBVTT", ""]

        # VTT não exige índice numérico por bloco — sem enumerate
        for seg in segments:
            # separator="." é obrigatório no formato VTT (SRT usa ",")
            start = self._fmt_timestamp(seg["start"], separator=".")
            end   = self._fmt_timestamp(seg["end"],   separator=".")
            lines.append(f"{start} --> {end}")
            lines.append(seg["text"].strip())   # .strip() remove espaços que o Whisper pode incluir
            lines.append("")                    # linha em branco entre blocos

        # Uma única operação de I/O para o arquivo inteiro
        dest.write_text("\n".join(lines), encoding="utf-8")

    def _write_json(self, result: Dict[str, Any], dest: Path) -> None:
        # Seleciona apenas os campos de valor para quem consome o arquivo —
        # "success" e "error" são controle interno e ficam de fora
        export = {
            "metadata": result.get("metadata", {}),
            "text":     result.get("text", ""),
            "segments": result.get("segments", []),
        }
        dest.write_text(
            json.dumps(
                export,
                ensure_ascii=False,  # preserva acentos legíveis em vez de sequências \uXXXX
                indent=2,            # formatado para leitura humana
            ),
            encoding="utf-8",
        )

    def _write_docx(self, result: Dict[str, Any], dest: Path) -> None:
        # Import local — se python-docx não estiver instalado, só .docx falha
        # O save() captura o ImportError separado por causa dessa decisão
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = Document()

        # duration e lang_prob ficam sem fallback — None é tratado inline na formatação
        metadata  = result.get("metadata", {})
        file_name = metadata.get("file_name", "Transcrição")
        language  = metadata.get("language", "—")
        duration  = metadata.get("duration")
        lang_prob = metadata.get("language_probability")

        # .stem remove a extensão do arquivo: "aula_sample.mp4" → "aula_sample"
        # LEFT explícito porque o padrão do python-docx para headings é centralizado
        title = doc.add_heading(Path(file_name).stem, level=1)
        title.alignment = WD_ALIGN_PARAGRAPH.LEFT

        doc.add_heading("Informações", level=2)

        # Formatação condicional inline: lang_prob:.0% → "99%", duration:.1f → "83.4s"
        # "Palavras" é derivado do texto — não vem dos metadados
        meta_items = [
            ("Arquivo",  file_name),
            ("Idioma",   f"{language} ({lang_prob:.0%})" if lang_prob else language),
            ("Duração",  f"{duration:.1f}s" if duration else "—"),
            ("Palavras", str(len(result.get("text", "").split()))),
        ]

        # Dois runs por parágrafo: label em negrito + valor em texto normal
        for label, value in meta_items:
            p = doc.add_paragraph()
            run_label = p.add_run(f"{label}: ")
            run_label.bold = True
            p.add_run(value)

        doc.add_heading("Transcrição completa", level=2)
        doc.add_paragraph(result.get("text", ""))

        doc.add_heading("Segmentos", level=2)

        for seg in result.get("segments", []):
            # separator padrão "," — aqui é display, não precisa seguir spec de legenda
            start = self._fmt_timestamp(seg["start"])
            end   = self._fmt_timestamp(seg["end"])

            p = doc.add_paragraph()
            # Dois runs: timestamp em negrito cinza (#888888) + texto do segmento em preto normal
            ts = p.add_run(f"[{start} → {end}]  ")
            ts.bold = True
            ts.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            p.add_run(seg["text"].strip())

        # python-docx exige str — Path não é aceito diretamente
        doc.save(str(dest))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_timestamp(seconds: float, separator: str = ",") -> str:
        """
        Converte segundos para HH:MM:SS{sep}mmm.
        separator="," → SRT    (00:01:23,456)
        separator="." → VTT    (00:01:23.456)
        separator="," (default) → display legível
        """
        # Converte para milissegundos inteiros primeiro —
        # evita erros de precisão de float (ex: 83.456 * 1000 = 83455.99...)
        total_ms = int(round(seconds * 1000))

        # Desmembramento por camadas: % isola o resto, // desce um nível
        ms      = total_ms % 1000       # 83456 → 456ms
        total_s = total_ms // 1000      # 83456 → 83s totais
        s       = total_s % 60          # 83    → 23s
        m       = (total_s // 60) % 60  # 83    → 1m  | % 60 necessário para durações acima de 1h
        h       = total_s // 3600       # 83    → 0h

        # 02d → mínimo 2 dígitos com zero à esquerda | 03d → mínimo 3 dígitos para ms
        return f"{h:02d}:{m:02d}:{s:02d}{separator}{ms:03d}"

    @staticmethod
    def _resolve_stem(result: Dict[str, Any], file_name: Optional[str]) -> str:
        # Caminho 1: file_name foi passado diretamente
        # .stem garante tolerância a inputs como "audio.mp3" ou "audio" — resultado sempre sem extensão
        if file_name:
            return Path(file_name).stem
        
        # Caminho 2: busca o nome nos metadados do resultado
        # Se não achar nada nos metadados, usa o fallback "transcricao"
        original = result.get("metadata", {}).get("file_name", "transcricao")
        return Path(original).stem  