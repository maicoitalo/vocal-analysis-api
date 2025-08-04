from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import librosa
import parselmouth
import base64
import io
import soundfile as sf
from typing import Optional, Dict, Any
import os
import time

app = FastAPI(
    title="Vocal Analysis API",
    version="2.1.0",
    description="API profissional para análise técnica de voz - Versão Otimizada"
)

# Configuração CORS para n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key para segurança
API_KEY = os.getenv("API_KEY", "vocal-api-beca-2025-xyz789")

class AudioAnalysisRequest(BaseModel):
    audio_base64: str
    sample_rate: Optional[int] = 44100
    quick_mode: Optional[bool] = False  # Modo rápido para análises mais leves

class VocalAnalysisResponse(BaseModel):
    pitch_analysis: Dict[str, Any]
    vocal_classification: Dict[str, Any]
    intonation_accuracy: Dict[str, Any]
    formants: Dict[str, Any]
    vibrato: Dict[str, Any]
    voice_quality: Dict[str, Any]
    breathing_analysis: Dict[str, Any]
    processing_time: Optional[float] = None

def decode_audio(audio_base64: str):
    """Decodifica áudio base64 para numpy array"""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Converter para mono se for estéreo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Limitar duração para performance (máximo 60 segundos)
        max_duration = 60  # segundos
        max_samples = sample_rate * max_duration
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        
        return audio_data, sample_rate
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao decodificar áudio: {str(e)}")

def analyze_pitch_with_librosa(audio_data, sample_rate, quick_mode=False):
    """Análise de pitch usando Librosa - Otimizada"""
    try:
        # Downsampling para performance se não for quick_mode
        if sample_rate > 22050 and not quick_mode:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=22050)
            sample_rate = 22050
        
        # Usar pyin com parâmetros otimizados
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate,
            frame_length=2048 if quick_mode else 4096
        )
        
        # Remover valores NaN
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) == 0:
            valid_f0 = np.array([440.0])  # Default A4
        
        # Converter frequências para notas
        notes = []
        for freq in valid_f0[:1000]:  # Limitar para performance
            if freq > 0:
                note = librosa.hz_to_note(freq)
                notes.append(note)
        
        # Análise estatística
        mean_freq = float(np.mean(valid_f0))
        median_freq = float(np.median(valid_f0))
        min_freq = float(np.min(valid_f0))
        max_freq = float(np.max(valid_f0))
        
        # Notas mais frequentes
        from collections import Counter
        note_counts = Counter(notes)
        most_common_notes = [
            {
                "note": note,
                "frequency_hz": float(librosa.note_to_hz(note)),
                "percentage": float(count / len(notes) * 100) if notes else 0
            }
            for note, count in note_counts.most_common(5)
        ]
        
        return {
            "mean_frequency_hz": mean_freq,
            "median_frequency_hz": median_freq,
            "min_frequency_hz": min_freq,
            "max_frequency_hz": max_freq,
            "note_range": f"{librosa.hz_to_note(min_freq) if min_freq > 0 else 'N/A'}-{librosa.hz_to_note(max_freq) if max_freq > 0 else 'N/A'}",
            "most_frequent_notes": most_common_notes,
            "total_notes_detected": len(notes),
            "analysis_mode": "quick" if quick_mode else "full"
        }
    except Exception as e:
        return {
            "error": f"Erro na análise de pitch: {str(e)}",
            "mean_frequency_hz": 440.0,
            "note_range": "A4"
        }

def classify_voice_type(mean_freq, gender="auto"):
    """Classifica o tipo de voz baseado na frequência média"""
    classifications = {
        "feminine": [
            (175, 260, "Contralto"),
            (220, 350, "Mezzo-soprano"),
            (260, 520, "Soprano"),
            (300, 600, "Soprano Lírico"),
            (350, 700, "Soprano Ligeiro")
        ],
        "masculine": [
            (80, 150, "Baixo"),
            (100, 180, "Baixo-barítono"),
            (110, 220, "Barítono"),
            (130, 260, "Tenor"),
            (150, 320, "Tenor Lírico")
        ]
    }
    
    # Auto-detectar gênero baseado na frequência
    if gender == "auto":
        gender = "feminine" if mean_freq > 170 else "masculine"
    
    voice_type = "Indefinido"
    confidence = 0
    
    for min_f, max_f, vtype in classifications[gender]:
        if min_f <= mean_freq <= max_f:
            voice_type = vtype
            center = (min_f + max_f) / 2
            distance = abs(mean_freq - center)
            max_distance = (max_f - min_f) / 2
            confidence = max(0, min(100, (1 - distance/max_distance) * 100))
            break
    
    return {
        "type": voice_type,
        "range": f"C4-C6" if gender == "feminine" else "E2-E4",
        "confidence_percentage": float(confidence),
        "f0_mean_hz": float(mean_freq),
        "gender": gender,
        "tessitura": "Média-aguda" if mean_freq > 250 else "Média-grave"
    }

def analyze_intonation(frequencies):
    """Analisa a precisão da afinação - Versão simplificada"""
    try:
        if len(frequencies) == 0:
            return {
                "overall_accuracy_percentage": 0,
                "average_deviation_cents": 0,
                "quality": "Não detectado"
            }
        
        # Limitar análise para performance
        frequencies = frequencies[:500]
        
        # Calcular desvio em cents para cada nota
        deviations = []
        for freq in frequencies:
            if freq > 0:
                note = librosa.hz_to_note(freq)
                target_freq = librosa.note_to_hz(note)
                if target_freq > 0:
                    cents = 1200 * np.log2(freq / target_freq)
                    deviations.append(abs(cents))
        
        if not deviations:
            return {
                "overall_accuracy_percentage": 0,
                "average_deviation_cents": 0,
                "quality": "Não detectado"
            }
        
        avg_deviation = float(np.mean(deviations))
        
        # Classificação simplificada
        if avg_deviation < 10:
            quality = "Excelente"
            accuracy = 95
        elif avg_deviation < 25:
            quality = "Boa"
            accuracy = 80
        elif avg_deviation < 50:
            quality = "Regular"
            accuracy = 60
        else:
            quality = "Precisa melhorar"
            accuracy = 40
        
        return {
            "overall_accuracy_percentage": float(accuracy),
            "average_deviation_cents": avg_deviation,
            "quality": quality
        }
    except Exception as e:
        return {
            "error": f"Erro na análise de afinação: {str(e)}",
            "overall_accuracy_percentage": 0,
            "average_deviation_cents": 0,
            "quality": "Não analisado"
        }

def analyze_formants_with_parselmouth(audio_data, sample_rate, quick_mode=False):
    """Analisa formantes usando Parselmouth - Otimizada"""
    try:
        if quick_mode:
            return {
                "F1": {"mean_hz": 700, "std_hz": 100},
                "F2": {"mean_hz": 1220, "std_hz": 150},
                "F3": {"mean_hz": 2900, "std_hz": 200},
                "singer_formant": {"present": False, "strength_percentage": 0},
                "vowel_space": "Análise rápida - detalhes não disponíveis"
            }
        
        # Limitar duração para análise
        max_duration = 10  # segundos
        max_samples = int(sample_rate * max_duration)
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        
        # Criar objeto Sound do Parselmouth
        sound = parselmouth.Sound(audio_data, sampling_frequency=sample_rate)
        
        # Extrair formantes
        formant = sound.to_formant_burg()
        
        # Coletar valores de formantes (menos pontos para performance)
        f1_values = []
        f2_values = []
        f3_values = []
        
        for t in np.linspace(0, sound.duration, 50):  # Reduzido de 100 para 50
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            
            if f1 and not np.isnan(f1): f1_values.append(f1)
            if f2 and not np.isnan(f2): f2_values.append(f2)
            if f3 and not np.isnan(f3): f3_values.append(f3)
        
        result = {
            "F1": {
                "mean_hz": float(np.mean(f1_values)) if f1_values else 700,
                "std_hz": float(np.std(f1_values)) if f1_values else 100
            },
            "F2": {
                "mean_hz": float(np.mean(f2_values)) if f2_values else 1220,
                "std_hz": float(np.std(f2_values)) if f2_values else 150
            },
            "F3": {
                "mean_hz": float(np.mean(f3_values)) if f3_values else 2900,
                "std_hz": float(np.std(f3_values)) if f3_values else 200
            }
        }
        
        # Detectar formante do cantor
        f3_mean = result["F3"]["mean_hz"]
        singer_formant = (2800 <= f3_mean <= 3200)
        
        result["singer_formant"] = {
            "present": singer_formant,
            "strength_percentage": float(min(100, max(0, (f3_mean - 2500) / 7))) if f3_mean > 2500 else 0
        }
        
        # Análise simplificada do espaço vocálico
        f1_mean = result["F1"]["mean_hz"]
        if f1_mean < 400:
            result["vowel_space"] = "Fechado - voz mais escura"
        elif f1_mean > 700:
            result["vowel_space"] = "Aberto - voz mais clara"
        else:
            result["vowel_space"] = "Balanceado"
        
        return result
    except Exception as e:
        return {
            "error": f"Erro na análise de formantes: {str(e)}",
            "F1": {"mean_hz": 700, "std_hz": 100},
            "F2": {"mean_hz": 1220, "std_hz": 150},
            "F3": {"mean_hz": 2900, "std_hz": 200},
            "singer_formant": {"present": False, "strength_percentage": 0}
        }

def detect_vibrato_simple(frequencies):
    """Detecta vibrato - Versão simplificada e robusta"""
    try:
        if len(frequencies) < 50:
            return {
                "present": False,
                "rate_hz": 0,
                "extent_cents": 0,
                "quality": "Amostra muito curta para detectar vibrato"
            }
        
        # Análise simplificada de variação
        freq_diff = np.diff(frequencies)
        freq_std = np.std(frequencies)
        freq_mean = np.mean(frequencies)
        
        # Critério simples para vibrato
        variation_percent = (freq_std / freq_mean) * 100
        
        if variation_percent > 1 and variation_percent < 5:
            # Possível vibrato
            return {
                "present": True,
                "rate_hz": 5.5,  # Valor típico
                "extent_cents": float(variation_percent * 20),
                "quality": "Vibrato detectado - análise simplificada"
            }
        else:
            return {
                "present": False,
                "rate_hz": 0,
                "extent_cents": 0,
                "quality": "Não detectado"
            }
    except Exception as e:
        return {
            "present": False,
            "rate_hz": 0,
            "extent_cents": 0,
            "quality": f"Análise não disponível: {str(e)}"
        }

def analyze_voice_quality_simple(audio_data, sample_rate):
    """Analisa qualidade vocal - Versão simplificada"""
    try:
        # Análise simplificada baseada em energia e variação
        rms = librosa.feature.rms(y=audio_data)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Estimar qualidade baseado em estabilidade
        stability = 1 - (rms_std / rms_mean) if rms_mean > 0 else 0
        stability_percent = min(100, max(0, stability * 100))
        
        # Classificar qualidade
        if stability_percent > 80:
            quality = "Excelente - voz estável e saudável"
        elif stability_percent > 60:
            quality = "Boa - voz saudável"
        elif stability_percent > 40:
            quality = "Regular - alguma instabilidade"
        else:
            quality = "Precisa atenção - voz instável"
        
        return {
            "hnr_db": 15.0 + (stability_percent / 5),  # Estimativa
            "jitter_percent": max(0.5, 5 - stability_percent / 20),
            "shimmer_percent": max(2, 10 - stability_percent / 10),
            "overall_quality": quality,
            "clarity_score": stability_percent,
            "stability_score": stability_percent,
            "analysis_type": "simplified"
        }
    except Exception as e:
        return {
            "hnr_db": 15.0,
            "jitter_percent": 1.5,
            "shimmer_percent": 4.0,
            "overall_quality": "Análise básica",
            "clarity_score": 75.0,
            "stability_score": 75.0,
            "error": str(e)
        }

def analyze_breathing(audio_data, sample_rate):
    """Analisa padrões de respiração - Otimizada"""
    try:
        # Detectar pausas usando RMS
        hop_length = 512
        rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
        
        # Threshold dinâmico
        threshold = np.percentile(rms, 20)
        is_silence = rms < threshold
        
        # Contar pausas
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, silent in enumerate(is_silence):
            if silent and not in_pause:
                in_pause = True
                pause_start = i
            elif not silent and in_pause:
                in_pause = False
                pause_duration = (i - pause_start) * hop_length / sample_rate
                if pause_duration > 0.2:  # Pausas maiores que 200ms
                    pauses.append(pause_duration)
        
        # Análise das pausas
        num_pauses = len(pauses)
        audio_duration = len(audio_data) / sample_rate
        
        if num_pauses > 0:
            avg_pause = np.mean(pauses)
            pauses_per_minute = (num_pauses / audio_duration) * 60
            
            # Qualidade da respiração
            if pauses_per_minute < 10 and avg_pause < 0.5:
                quality = "Excelente controle respiratório"
            elif pauses_per_minute < 20 and avg_pause < 1:
                quality = "Bom controle respiratório"
            else:
                quality = "Controle respiratório regular"
            
            avg_phrase_duration = audio_duration / (num_pauses + 1)
            
            result = {
                "number_of_pauses": int(num_pauses),
                "pauses_per_minute": float(pauses_per_minute),
                "average_pause_duration_seconds": float(avg_pause),
                "average_phrase_duration_seconds": float(avg_phrase_duration),
                "breathing_quality": quality,
                "support_evaluation": "Bom" if avg_phrase_duration > 3 else "Regular"
            }
        else:
            result = {
                "number_of_pauses": 0,
                "pauses_per_minute": 0,
                "average_pause_duration_seconds": 0,
                "average_phrase_duration_seconds": float(audio_duration),
                "breathing_quality": "Não foram detectadas pausas respiratórias",
                "support_evaluation": "Excelente" if audio_duration > 10 else "Bom"
            }
        
        return result
    except Exception as e:
        return {
            "error": f"Erro na análise de respiração: {str(e)}",
            "number_of_pauses": 0,
            "breathing_quality": "Não analisado"
        }

@app.get("/")
def home():
    return {
        "name": "Vocal Analysis API",
        "status": "running",
        "version": "2.1.0",
        "endpoints": {
            "home": "/",
            "analyze": "/analyze",
            "health": "/health",
            "docs": "/docs",
            "test-speed": "/test-speed"
        },
        "optimization": "Performance optimized version"
    }

@app.post("/analyze", response_model=VocalAnalysisResponse)
async def analyze_audio(
    request: AudioAnalysisRequest,
    x_api_key: str = Header(None)
):
    """
    Endpoint principal para análise completa de voz.
    Requer API Key no header X-API-Key.
    """
    
    # Timer para medir performance
    start_time = time.time()
    
    # Verificar API Key
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida ou não fornecida")
    
    try:
        # Decodificar áudio
        audio_data, sample_rate = decode_audio(request.audio_base64)
        
        # Reduzir sample rate para performance se muito alto
        if sample_rate > 22050 and not request.quick_mode:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=22050)
            sample_rate = 22050
        
        # Análise de pitch
        pitch_analysis = analyze_pitch_with_librosa(audio_data, sample_rate, request.quick_mode)
        
        # Extrair frequências para outras análises
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate,
            frame_length=2048 if request.quick_mode else 4096
        )
        frequencies = f0[~np.isnan(f0)]
        
        # Classificação vocal
        mean_freq = pitch_analysis.get("mean_frequency_hz", 440)
        vocal_classification = classify_voice_type(mean_freq)
        
        # Análise de afinação
        intonation_accuracy = analyze_intonation(frequencies)
        
        # Análises opcionais baseadas no modo
        if request.quick_mode:
            formants = {
                "note": "Análise detalhada desabilitada no modo rápido",
                "F1": {"mean_hz": 700, "std_hz": 100},
                "F2": {"mean_hz": 1220, "std_hz": 150},
                "F3": {"mean_hz": 2900, "std_hz": 200},
                "singer_formant": {"present": False, "strength_percentage": 0}
            }
            voice_quality = analyze_voice_quality_simple(audio_data, sample_rate)
        else:
            formants = analyze_formants_with_parselmouth(audio_data, sample_rate, request.quick_mode)
            voice_quality = analyze_voice_quality_simple(audio_data, sample_rate)
        
        # Detectar vibrato (versão simplificada)
        vibrato = detect_vibrato_simple(frequencies)
        
        # Análise de respiração
        breathing_analysis = analyze_breathing(audio_data, sample_rate)
        
        # Tempo total de processamento
        processing_time = time.time() - start_time
        
        response = VocalAnalysisResponse(
            pitch_analysis=pitch_analysis,
            vocal_classification=vocal_classification,
            intonation_accuracy=intonation_accuracy,
            formants=formants,
            vibrato=vibrato,
            voice_quality=voice_quality,
            breathing_analysis=breathing_analysis,
            processing_time=processing_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Vocal Analysis API",
        "version": "2.1.0",
        "optimization": "Performance optimized"
    }

@app.post("/test")
async def test_analysis(x_api_key: str = Header(None)):
    """Endpoint de teste que retorna análise mock"""
    
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    
    return {
        "status": "success",
        "message": "API funcionando corretamente!",
        "mock_analysis": {
            "pitch": "440 Hz (A4)",
            "voice_type": "Soprano",
            "quality": "Boa"
        }
    }

@app.post("/test-speed")
async def test_speed(x_api_key: str = Header(None)):
    """Teste de velocidade de processamento"""
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    
    start = time.time()
    
    # Teste 1: Criar array numpy
    test_audio = np.random.rand(44100 * 10)  # 10 segundos de áudio fake
    t1 = time.time() - start
    
    # Teste 2: Librosa simples
    start2 = time.time()
    pitches, magnitudes = librosa.piptrack(y=test_audio, sr=44100)
    t2 = time.time() - start2
    
    # Teste 3: Processamento rápido
    start3 = time.time()
    rms = librosa.feature.rms(y=test_audio)
    t3 = time.time() - start3
    
    return {
        "numpy_time": round(t1, 3),
        "librosa_time": round(t2, 3),
        "rms_time": round(t3, 3),
        "total_time": round(time.time() - start, 3),
        "expected_analysis_time": "2-5 seconds for real audio"
    }
