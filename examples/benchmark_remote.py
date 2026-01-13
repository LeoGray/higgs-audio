"""Example script for generating audio using HiggsAudio (Remote Benchmark Version)."""

import click
import soundfile as sf
import langid
import sys
import os
import time
import requests
import json
import base64
import wave
import io
import concurrent.futures
import numpy as np

# Add parent directory to path to import boson_multimodal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loguru import logger
# We attempt to import from boson_multimodal, assuming the submodule is correctly placed
try:
    from boson_multimodal.data_types import Message, AudioContent, TextContent
except ImportError:
    # Fallback/Mock classes if submodule is missing (or just failing to load deps)
    class Message:
        def __init__(self, role, content): self.role, self.content = role, content
    class AudioContent:
        def __init__(self, audio_url): self.audio_url = audio_url
    class TextContent:
        def __init__(self, text): self.text = text

from typing import List, Optional

AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

# --- Helper Functions from examples/generation.py ---

def normalize_chinese_punctuation(text):
    chinese_to_english_punct = {
        "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!",
        "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
        "“": '"', "”": '"', "‘": "'", "’": "'", "、": ",", "—": "-",
        "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text

def prepare_chunk_text(text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1):
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            if language == "zh":
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")

# --- Remote Client Class (Replacing HiggsAudioModelClient) ---

class RemoteHiggsAudioClient:
    def __init__(self, base_url, api_key="EMPTY"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # Fetch model ID
        try:
            res = requests.get(f"{self.base_url}/models", timeout=5)
            res.raise_for_status()
            self.model_id = res.json()['data'][0]['id']
            logger.info(f"Discovered Model ID: {self.model_id}")
        except Exception as e:
            self.model_id = "default_model"
            logger.warning(f"Failed to fetch model list ({e}), using default: {self.model_id}")

    def encode_audio_file(self, path):
        if not path or not os.path.exists(path): return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate(self, messages, chunked_text, **kwargs):
        # 1. Prepare Base Messages (Context)
        base_api_messages = []
        for msg in messages:
            role = msg.role
            content = msg.content
            if isinstance(content, str):
                base_api_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Mixed content
                api_content = []
                for item in content:
                    if hasattr(item, 'text'):
                        api_content.append({"type": "text", "text": item.text})
                    elif hasattr(item, 'audio_url'):
                        if item.audio_url and os.path.exists(item.audio_url):
                            b64 = self.encode_audio_file(item.audio_url)
                            api_content.append({"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}})
                base_api_messages.append({"role": role, "content": api_content})
            elif hasattr(content, 'audio_url'):
                 if content.audio_url and os.path.exists(content.audio_url):
                    b64 = self.encode_audio_file(content.audio_url)
                    base_api_messages.append({
                        "role": role, 
                        "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]
                    })

        generated_audio_chunks = []
        final_text = ""
        total_audio_duration = 0.0
        
        # Benchmarking Stats
        latencies = []
        ttfts = []
        
        logger.info(f"Starting Generation for {len(chunked_text)} chunks...")
        
        for idx, chunk_text in tqdm.tqdm(enumerate(chunked_text), desc="Remote Generation"):
            current_messages = copy.deepcopy(base_api_messages)
            current_messages.append({"role": "user", "content": chunk_text})
            
            payload = {
                "model": self.model_id,
                "messages": current_messages,
                "max_tokens": kwargs.get("max_new_tokens", 2048),
                "temperature": kwargs.get("temperature", 1.0),
                "top_p": kwargs.get("top_p", 0.95),
                "stream": False # Streaming audio is experimental; use non-streaming for stability
            }
            
            start_time = time.time()
            # TTFT n/a for non-streaming
            


            try:
                logger.info(f"Sending request for chunk {idx} (max_tokens={kwargs.get('max_new_tokens', 2048)})...")
                res = requests.post(
                    f"{self.base_url}/chat/completions", 
                    json=payload, 
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=600 # 10 minute timeout
                )
                logger.info(f"Response received: {res.status_code}")
                res.raise_for_status()
                data = res.json()

                
                end_time = time.time()
                latency = end_time - start_time
                latencies.append(latency)
                
                choice = data['choices'][0]
                message = choice['message']
                
                # Handle Text
                if 'content' in message and message['content']:
                    final_text += message['content']
                    
                # Handle Audio (Standard Object or Custom)
                b64_audio = None
                if 'audio' in message and message['audio']:
                    b64_audio = message['audio']['data']
                elif 'content' in message and isinstance(message['content'], str) and "<|AUDIO_OUT|>" in message['content']:
                    # Fallback for Token-based audio (logic needed to extract)
                    pass
                
                if b64_audio:
                    chunk_audio_bytes = base64.b64decode(b64_audio)
                    with open(f"/tmp/temp_chunk_{idx}.wav", "wb") as f:
                        f.write(chunk_audio_bytes)
                    data, sr = sf.read(f"/tmp/temp_chunk_{idx}.wav")
                    generated_audio_chunks.append((data, sr))
                    duration = len(data) / sr
                    total_audio_duration += duration
                    logger.info(f"Chunk {idx}: Latency={latency:.2f}s, Audio={duration:.2f}s, RTF={(latency/duration if duration > 0 else 0):.2f}")
                else:
                    logger.warning(f"Chunk {idx}: No audio object found in response.")
                    logger.debug(f"Response keys: {message.keys()}")

            except Exception as e:
                logger.error(f"Failed chunk {idx}: {e}")


        # Metrics Summary for this single generation
        metrics = {
             "total_latency": sum(latencies) if latencies else 0,
             "total_duration": total_audio_duration,
             "avg_latency_per_chunk": sum(latencies)/len(latencies) if latencies else 0,
             "rtf": (sum(latencies)/total_audio_duration) if total_audio_duration > 0 else 0,
             "ttfts": ttfts
        }

        if not generated_audio_chunks:
            return None, 24000, final_text, metrics
            
        final_sr = generated_audio_chunks[0][1]
        final_wav = np.concatenate([c[0] for c in generated_audio_chunks])
        return final_wav, final_sr, final_text, metrics

# --- Wrapper for Context Preparation (Simplified for Remote) ---

def prepare_generation_context_remote(scene_prompt, transcript):
    messages = []
    sys_content = "You are an AI assistant designed to convert text into speech."
    if scene_prompt:
        sys_content += f"\n\nScene: {scene_prompt}"
    
    messages.append(Message(role="system", content=sys_content))
    return messages, []

@click.command()
@click.option("--base_url", type=str, default="http://172.202.29.125:26000/v1", help="Remote Endpoint URL")
@click.option("--transcript", type=str, default="transcript.txt", help="Input text file path or raw string")
@click.option("--out_path", type=str, default="generation.wav", help="Output wav path")
@click.option("--scene_prompt", type=str, default=None)
@click.option("--chunk_method", type=str, default="word")
@click.option("--chunk_size", type=int, default=50)
@click.option("--concurrency", type=int, default=1, help="Number of concurrent threads")
@click.option("--num_requests", type=int, default=1, help="Total number of requests to run")
@click.option("--max_new_tokens", type=int, default=2048, help="Max tokens to generate")
def main(base_url, transcript, out_path, scene_prompt, chunk_method, chunk_size, concurrency, num_requests, max_new_tokens):
    
    # Resolve transcript (file or string)
    if os.path.exists(transcript):
         with open(transcript, "r", encoding="utf-8") as f:
            base_text = f.read().strip()
    else:
        base_text = transcript

    client = RemoteHiggsAudioClient(base_url)
    
    # Pre-process text once (normalization, context)
    text = normalize_chinese_punctuation(base_text)
    messages, _ = prepare_generation_context_remote(scene_prompt, text)
    chunked_text = prepare_chunk_text(text, chunk_method=chunk_method, chunk_max_word_num=chunk_size)
    
    logger.info(f"Starting Load Test: Concurrency={concurrency}, Requests={num_requests}, MaxTokens={max_new_tokens}")
    logger.info(f"Payload: {len(chunked_text)} chunks, Total Char Length={len(text)}")
    
    import concurrent.futures
    import numpy as np
    
    results = []
    
    start_wall_time = time.time()
    
    def _worker(req_id):
        # Unique output path for each request to avoid file conflicts
        req_out = out_path.replace(".wav", f"_{req_id}.wav") if num_requests > 1 else out_path
        try:
            wav, sr, res_text, metrics = client.generate(messages, chunked_text, max_new_tokens=max_new_tokens)
            if wav is not None:
                # Only save file if it's a single run or explicitly debugged
                # For load testing, saving 1000 files might be bad, but saving 10 is fine
                if num_requests <= 10:
                    sf.write(req_out, wav, sr)
                return True, metrics
            return False, None
        except Exception as e:
            logger.error(f"Request {req_id} failed: {e}")
            return False, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_worker, i) for i in range(num_requests)]
        
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=num_requests, desc="Progress"):
            success, metrics = future.result()
            if success:
                results.append(metrics)

    end_wall_time = time.time()
    total_wall_time = end_wall_time - start_wall_time
    
    # Analysis
    if not results:
        logger.error("No successful requests.")
        return

    report_latencies = [r['total_latency'] for r in results]
    report_rtfs = [r['rtf'] for r in results]
    report_durations = [r['total_duration'] for r in results]
    
    total_audio_generated = sum(report_durations)
    system_throughput = total_audio_generated / total_wall_time
    success_rate = (len(results) / num_requests) * 100
    
    print("\n" + "="*50)
    print(f"BENCHMARK REPORT (C={concurrency}, N={num_requests})")
    print("="*50)
    print(f"Total Wall Time:    {total_wall_time:.2f} s")
    print(f"Success Rate:       {success_rate:.1f}% ({len(results)}/{num_requests})")
    print(f"System Throughput:  {system_throughput:.2f} audio_sec/sec")
    print("-" * 50)
    print("LATENCY (End-to-End per request)")
    print(f"  Avg: {np.mean(report_latencies):.4f} s")
    print(f"  P50: {np.percentile(report_latencies, 50):.4f} s")
    print(f"  P90: {np.percentile(report_latencies, 90):.4f} s")
    print(f"  P95: {np.percentile(report_latencies, 95):.4f} s")
    print(f"  P99: {np.percentile(report_latencies, 99):.4f} s")
    print("-" * 50)
    print("REAL-TIME FACTOR (RTF)")
    print(f"  Avg: {np.mean(report_rtfs):.4f}")
    print(f"  P50: {np.percentile(report_rtfs, 50):.4f}")
    print(f"  P95: {np.percentile(report_rtfs, 95):.4f}")
    print(f"  P99: {np.percentile(report_rtfs, 99):.4f}")
    print("="*50)


if __name__ == "__main__":
    main()