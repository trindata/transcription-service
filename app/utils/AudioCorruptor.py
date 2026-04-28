from pathlib import Path
import random
import shutil
import wave
import struct


class AudioCorruptor:
    """
    Gera variantes corrompidas de arquivos de áudio para testes de robustez.

    Cada método produz um arquivo independente no diretório de saída, sem
    modificar o original. Os métodos de corrupção binária (truncar, alterar
    bytes, quebrar header, inverter, zerar trecho) operam diretamente nos
    bytes do arquivo. Os métodos de geração (silêncio, ruído) produzem WAVs
    válidos que testam o comportamento do modelo com áudio sem conteúdo útil.

    Uso:
        corruptor = AudioCorruptor("audio.mp3", output_dir="arquivos_teste")
        corruptor.gerar_todos()
    """
    def __init__(self, input_file: str, output_dir: str = "arquivos_corrompidos"):
        """
        Args:
            input_file: Caminho do arquivo de áudio original.
            output_dir: Diretório onde os arquivos gerados serão salvos.
                        Criado automaticamente se não existir.

        Raises:
            FileNotFoundError: Se input_file não existir.
        """
        
        self.input_path = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.input_path}")

        self.stem = self.input_path.stem
        self.suffix = self.input_path.suffix.lower()

    def _read_bytes(self) -> bytes:
        """Lê e retorna o conteúdo binário do arquivo de entrada."""
        return self.input_path.read_bytes()

    def truncar_final(self, percent_keep: float = 0.7) -> Path:
        """
        Mantém apenas uma parte do arquivo.
        Ex.: 0.7 = mantém 70% e remove 30% do final.
        """
        if not (0 < percent_keep < 1):
            raise ValueError("percent_keep deve estar entre 0 e 1")

        data = self._read_bytes()
        cut_index = int(len(data) * percent_keep)
        out_path = self.output_dir / f"{self.stem}_truncado{self.suffix}"
        out_path.write_bytes(data[:cut_index])
        return out_path

    def alterar_bytes_aleatorios(self, num_changes: int = 1000, preserve_header: int = 256) -> Path:
        """
        Altera bytes aleatórios no corpo do arquivo.
        preserve_header evita quebrar imediatamente o cabeçalho.
        """
        data = bytearray(self._read_bytes())

        if len(data) <= preserve_header:
            raise ValueError("Arquivo muito pequeno para preservar header e alterar bytes")

        max_changes = len(data) - preserve_header
        num_changes = min(num_changes, max_changes)

        positions = random.sample(range(preserve_header, len(data)), num_changes)

        for pos in positions:
            data[pos] = random.randint(0, 255)

        out_path = self.output_dir / f"{self.stem}_bytes_alterados{self.suffix}"
        out_path.write_bytes(data)
        return out_path

    def quebrar_header(self, bytes_to_remove: int = 128) -> Path:
        """
        Remove bytes iniciais do arquivo.
        Tende a quebrar a identificação/decodificação.
        """
        data = self._read_bytes()

        if len(data) <= bytes_to_remove:
            raise ValueError("Arquivo muito pequeno para remover esse tanto do header")

        out_path = self.output_dir / f"{self.stem}_header_quebrado{self.suffix}"
        out_path.write_bytes(data[bytes_to_remove:])
        return out_path

    def trocar_extensao(self, new_extension: str = ".wav") -> Path:
        """
        Apenas copia o arquivo mudando a extensão.
        Não converte de verdade.
        Isso testa se seu app confia na extensão em vez do conteúdo.
        """
        if not new_extension.startswith("."):
            new_extension = "." + new_extension

        out_path = self.output_dir / f"{self.stem}_extensao_trocada{new_extension}"
        shutil.copy2(self.input_path, out_path)
        return out_path

    def inverter_bytes(self) -> Path:
        """
        Inverte completamente os bytes do arquivo.
        É uma corrupção extrema e pouco realista, mas útil para teste bruto.
        """
        data = self._read_bytes()
        out_path = self.output_dir / f"{self.stem}_bytes_invertidos{self.suffix}"
        out_path.write_bytes(data[::-1])
        return out_path

    def sobrescrever_trecho_com_zeros(self, start_percent: float = 0.3, end_percent: float = 0.5) -> Path:
        """
        Zera um trecho interno do arquivo binário.
        Testa arquivos parcialmente danificados.
        """
        if not (0 <= start_percent < end_percent <= 1):
            raise ValueError("Use percentuais válidos entre 0 e 1")

        data = bytearray(self._read_bytes())
        start = int(len(data) * start_percent)
        end = int(len(data) * end_percent)

        for i in range(start, end):
            data[i] = 0

        out_path = self.output_dir / f"{self.stem}_trecho_zerado{self.suffix}"
        out_path.write_bytes(data)
        return out_path

    def gerar_silencio_wav(self, duration_sec: int = 10, sample_rate: int = 16000) -> Path:
        """
        Gera um WAV válido contendo apenas silêncio.
        Isso não é corrupção binária, mas é um caso crítico de teste.
        """
        out_path = self.output_dir / f"{self.stem}_silencio.wav"

        with wave.open(str(out_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(sample_rate)

            num_frames = duration_sec * sample_rate
            silence_frame = struct.pack("<h", 0)

            for _ in range(num_frames):
                wf.writeframesraw(silence_frame)

        return out_path

    def gerar_ruido_wav(self, duration_sec: int = 10, sample_rate: int = 16000, amplitude: int = 12000) -> Path:
        """
        Gera um WAV válido com ruído aleatório.
        Também não é corrupção binária, mas testa robustez da transcrição.
        """
        out_path = self.output_dir / f"{self.stem}_ruido.wav"

        with wave.open(str(out_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(sample_rate)

            num_frames = duration_sec * sample_rate

            for _ in range(num_frames):
                sample = random.randint(-amplitude, amplitude)
                wf.writeframesraw(struct.pack("<h", sample))

        return out_path

    def gerar_todos(self) -> list[Path]:
        """
        Executa todos os métodos de corrupção e geração em sequência.

        Returns:
            Lista com os caminhos de todos os arquivos gerados.
        """
        outputs = []

        outputs.append(self.truncar_final())
        outputs.append(self.alterar_bytes_aleatorios())
        outputs.append(self.quebrar_header())
        outputs.append(self.trocar_extensao(".wav"))
        outputs.append(self.inverter_bytes())
        outputs.append(self.sobrescrever_trecho_com_zeros())
        outputs.append(self.gerar_silencio_wav())
        outputs.append(self.gerar_ruido_wav())

        return outputs


if __name__ == "__main__":
    arquivo = Path(r"caminho_do_arquivo")  # troque aqui

    corruptor = AudioCorruptor(arquivo)
    arquivos_gerados = corruptor.gerar_todos()

    print("Arquivos gerados:")
    for path in arquivos_gerados:
        print(f"- {path}")