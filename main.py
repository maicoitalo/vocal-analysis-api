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

app = FastAPI(
    title="Vocal Analysis API",
    version="2.0.0",
    description="API profissional para análise técnica de voz"
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

class VocalAnalysisResponse(BaseModel):
    pitch_analysis: Dict[str, Any]
    vocal_classification: Dict[str, Any]
    intonation_accuracy: Dict[str, Any]
    formants: Dict[str, Any]
    vibrato: Dict[str, Any]
    voice_quality: Dict[str, Any]
    breathing_analysis: Dict[str, Any]

def decode_audio(audio_base64: str):
    """Decodifica áudio base64 para numpy array"""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Converter para mono se for estéreo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        return audio_data, sample_rate
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao decodificar áudio: {str(e)}")

def analyze_pitch_with_librosa(audio_data, sample_rate):
    """Análise de pitch usando Librosa"""
    try:
        # Usar pyin para detecção de pitch (mais preciso que piptrack)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate
        )
        
        # Remover valores NaN
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) == 0:
            # Fallback para piptrack se pyin falhar
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            valid_f0 = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    valid_f0.append(pitch)
            valid_f0 = np.array(valid_f0)
        
        if len(valid_f0) == 0:
            valid_f0 = np.array([440.0])  # Default A4
        
        # Converter frequências para notas
        notes = []
        for freq in valid_f0:
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
            "total_notes_detected": len(notes)
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
            # Calcular confiança baseado em quão central está a frequência
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
    """Analisa a precisão da afinação"""
    try:
        if len(frequencies) == 0:
            return {
                "overall_accuracy_percentage": 0,
                "average_deviation_cents": 0,
                "distribution": {},
                "quality": "Não detectado"
            }
        
        # Calcular desvio em cents para cada nota
        deviations = []
        for freq in frequencies:
            if freq > 0:
                # Encontrar a nota mais próxima
                note = librosa.hz_to_note(freq)
                target_freq = librosa.note_to_hz(note)
                
                # Calcular desvio em cents
                if target_freq > 0:
                    cents = 1200 * np.log2(freq / target_freq)
                    deviations.append(abs(cents))
        
        if not deviations:
            return {
                "overall_accuracy_percentage": 0,
                "average_deviation_cents": 0,
                "distribution": {},
                "quality": "Não detectado"
            }
        
        # Classificar desvios
        perfect = sum(1 for d in deviations if d < 5)
        excellent = sum(1 for d in deviations if 5 <= d < 10)
        good = sum(1 for d in deviations if 10 <= d < 25)
        regular = sum(1 for d in deviations if 25 <= d < 50)
        poor = sum(1 for d in deviations if d >= 50)
        
        total = len(deviations)
        accuracy = ((perfect * 1.0 + excellent * 0.9 + good * 0.7 + regular * 0.4) / total) * 100
        
        # Tendência sharp/flat
        signed_deviations = []
        for freq in frequencies:
            if freq > 0:
                note = librosa.hz_to_note(freq)
                target_freq = librosa.note_to_hz(note)
                if target_freq > 0:
                    cents = 1200 * np.log2(freq / target_freq)
                    signed_deviations.append(cents)
        
        avg_signed = np.mean(signed_deviations) if signed_deviations else 0
        tendency = "sharp" if avg_signed > 5 else "flat" if avg_signed < -5 else "centered"
        
        return {
            "overall_accuracy_percentage": float(accuracy),
            "average_deviation_cents": float(np.mean(deviations)),
            "distribution": {
                "perfect_<5_cents": int(perfect),
                "excellent_5-10_cents": int(excellent),
                "good_10-25_cents": int(good),
                "regular_25-50_cents": int(regular),
                "poor_>50_cents": int(poor)
            },
            "sharp_flat_tendency": tendency,
            "quality": "Excelente" if accuracy > 85 else "Boa" if accuracy > 70 else "Regular" if accuracy > 50 else "Precisa melhorar"
        }
    except Exception as e:
        return {
            "error": f"Erro na análise de afinação: {str(e)}",
            "overall_accuracy_percentage": 0
        }

def analyze_formants_with_parselmouth(audio_data, sample_rate):
    """Analisa formantes usando Parselmouth (Praat)"""
    try:
        # Criar objeto Sound do Parselmouth
        sound = parselmouth.Sound(audio_data, sampling_frequency=sample_rate)
        
        # Extrair formantes
        formant = sound.to_formant_burg()
        
        # Coletar valores de formantes ao longo do tempo
        f1_values = []
        f2_values = []
        f3_values = []
        f4_values = []
        
        for t in np.linspace(0, sound.duration, 100):
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            f4 = formant.get_value_at_time(4, t)
            
            if f1 and not np.isnan(f1): f1_values.append(f1)
            if f2 and not np.isnan(f2): f2_values.append(f2)
            if f3 and not np.isnan(f3): f3_values.append(f3)
            if f4 and not np.isnan(f4): f4_values.append(f4)
        
        # Calcular médias e desvios
        result = {
            "F1": {
                "mean_hz": float(np.mean(f1_values)) if f1_values else 0,
                "std_hz": float(np.std(f1_values)) if f1_values else 0
            },
            "F2": {
                "mean_hz": float(np.mean(f2_values)) if f2_values else 0,
                "std_hz": float(np.std(f2_values)) if f2_values else 0
            },
            "F3": {
                "mean_hz": float(np.mean(f3_values)) if f3_values else 0,
                "std_hz": float(np.std(f3_values)) if f3_values else 0
            },
            "F4": {
                "mean_hz": float(np.mean(f4_values)) if f4_values else 0,
                "std_hz": float(np.std(f4_values)) if f4_values else 0
            }
        }
        
        # Detectar formante do cantor (2800-3200 Hz)
        f3_mean = result["F3"]["mean_hz"]
        singer_formant = (2800 <= f3_mean <= 3200)
        
        result["singer_formant"] = {
            "present": singer_formant,
            "strength_percentage": float(min(100, max(0, (f3_mean - 2500) / 7))) if f3_mean > 2500 else 0
        }
        
        # Análise de qualidade vocal baseada em formantes
        f1_mean = result["F1"]["mean_hz"]
        f2_mean = result["F2"]["mean_hz"]
        
        if f1_mean > 0 and f2_mean > 0:
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
            "F1": {"mean_hz": 0, "std_hz": 0},
            "F2": {"mean_hz": 0, "std_hz": 0},
            "F3": {"mean_hz": 0, "std_hz": 0},
            "singer_formant": {"present": False, "strength_percentage": 0}
        }

def detect_vibrato(frequencies, time_array):
    """Detecta e analisa vibrato"""
    try:
        if len(frequencies) < 100:
            return {
                "present": False,
                "rate_hz": 0,
                "extent_cents": 0,
                "regularity_percentage": 0,
                "quality": "Não detectado"
            }
        
        # Suavizar frequências
        from scipy.signal import savgol_filter
        smoothed = savgol_filter(frequencies, 51, 3) if len(frequencies) > 51 else frequencies
        
        # Calcular variação de frequência
        freq_variation = frequencies - smoothed
        
        # Análise FFT para detectar taxa de vibrato
        from scipy.fft import fft, fftfreq
        N = len(freq_variation)
        yf = fft(freq_variation)
        xf = fftfreq(N, time_array[1] - time_array[0])[:N//2]
        
        # Procurar pico na faixa de vibrato (4-8 Hz)
        vibrato_range = (xf > 4) & (xf < 8)
        if np.any(vibrato_range):
            vibrato_freqs = xf[vibrato_range]
            vibrato_power = np.abs(yf[vibrato_range])
            
            if len(vibrato_power) > 0 and np.max(vibrato_power) > np.mean(np.abs(yf)) * 2:
                # Vibrato detectado
                peak_idx = np.argmax(vibrato_power)
                vibrato_rate = vibrato_freqs[peak_idx]
                
                # Calcular extensão em cents
                extent_semitones = np.std(freq_variation) / np.mean(frequencies) * 12
                extent_cents = extent_semitones * 100
                
                # Calcular regularidade
                regularity = (np.max(vibrato_power) / np.mean(vibrato_power)) * 10
                regularity = min(100, regularity)
                
                # Avaliar qualidade
                if 5 <= vibrato_rate <= 6.5 and 20 <= extent_cents <= 50:
                    quality = "Natural e equilibrado"
                elif vibrato_rate < 4:
                    quality = "Muito lento"
                elif vibrato_rate > 7:
                    quality = "Muito rápido"
                elif extent_cents < 20:
                    quality = "Muito sutil"
                elif extent_cents > 80:
                    quality = "Muito amplo"
                else:
                    quality = "Presente mas irregular"
                
                return {
                    "present": True,
                    "rate_hz": float(vibrato_rate),
                    "extent_cents": float(extent_cents),
                    "regularity_percentage": float(regularity),
                    "quality": quality
                }
        
        return {
            "present": False,
            "rate_hz": 0,
            "extent_cents": 0,
            "regularity_percentage": 0,
            "quality": "Não detectado"
        }
    except Exception as e:
        return {
            "error": f"Erro na detecção de vibrato: {str(e)}",
            "present": False
        }

def analyze_voice_quality(audio_data, sample_rate):
    """Analisa qualidade vocal (HNR, Jitter, Shimmer)"""
    try:
        # Criar objeto Sound do Parselmouth
        sound = parselmouth.Sound(audio_data, sampling_frequency=sample_rate)
        
        # HNR (Harmonic-to-Noise Ratio)
        harmonicity = sound.to_harmonicity()
        hnr_values = []
        for t in np.linspace(0, sound.duration, 50):
            hnr = harmonicity.get_value_at_time(t)
            if hnr and not np.isnan(hnr):
                hnr_values.append(hnr)
        
        mean_hnr = np.mean(hnr_values) if hnr_values else 0
        
        # Criar objeto PointProcess para análise de pulsos
        pitch = sound.to_pitch()
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
        
        # Jitter (variação de período)
        jitter = parselmouth.praat.call(point_process, sound, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        
        # Shimmer (variação de amplitude)
        shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # Avaliar qualidade
        if mean_hnr > 20 and jitter < 1 and shimmer < 3:
            quality = "Excelente - voz muito saudável"
        elif mean_hnr > 15 and jitter < 2 and shimmer < 5:
            quality = "Boa - voz saudável"
        elif mean_hnr > 10 and jitter < 3 and shimmer < 8:
            quality = "Regular - alguma rouquidão"
        else:
            quality = "Precisa atenção - possível fadiga vocal"
        
        return {
            "hnr_db": float(mean_hnr),
            "jitter_percent": float(jitter * 100),
            "shimmer_percent": float(shimmer * 100),
            "overall_quality": quality,
            "clarity_score": float(min(100, mean_hnr * 5)),
            "stability_score": float(max(0, 100 - jitter * 50 - shimmer * 10))
        }
    except Exception as e:
        return {
            "error": f"Erro na análise de qualidade: {str(e)}",
            "hnr_db": 0,
            "jitter_percent": 0,
            "shimmer_percent": 0,
            "overall_quality": "Não analisado"
        }

def analyze_breathing(audio_data, sample_rate):
    """Analisa padrões de respiração"""
    try:
        # Detectar pausas (silêncios)
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)
        
        # Calcular energia
        energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Threshold para detectar silêncio
        threshold = np.mean(energy) * 0.1
        is_silence = energy < threshold
        
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
            elif pauses_per_minute < 30:
                quality = "Controle respiratório regular"
            else:
                quality = "Precisa trabalhar a respiração"
            
            # Calcular duração média das frases
            avg_phrase_duration = audio_duration / (num_pauses + 1)
            
            result = {
                "number_of_pauses": int(num_pauses),
                "pauses_per_minute": float(pauses_per_minute),
                "average_pause_duration_seconds": float(avg_pause),
                "average_phrase_duration_seconds": float(avg_phrase_duration),
                "longest_phrase_seconds": float(audio_duration / (num_pauses + 1) * 1.5),
                "breathing_quality": quality,
                "support_evaluation": "Bom" if avg_phrase_duration > 3 else "Precisa melhorar"
            }
        else:
            result = {
                "number_of_pauses": 0,
                "pauses_per_minute": 0,
                "average_pause_duration_seconds": 0,
                "average_phrase_duration_seconds": float(audio_duration),
                "longest_phrase_seconds": float(audio_duration),
                "breathing_quality": "Não foram detectadas pausas",
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
        "version": "2.0.0",
        "endpoints": {
            "home": "/",
            "analyze": "/analyze",
            "health": "/health",
            "docs": "/docs"
        },
        "capabilities": [
            "Pitch detection (Librosa)",
            "Voice classification",
            "Intonation analysis",
            "Formant analysis",
            "Vibrato detection",
            "Voice quality metrics",
            "Breathing patterns"
        ]
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
    
    # Verificar API Key
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida ou não fornecida")
    
    try:
        # Decodificar áudio
        audio_data, sample_rate = decode_audio(request.audio_base64)
        
        # Análise de pitch com Librosa
        pitch_analysis = analyze_pitch_with_librosa(audio_data, sample_rate)
        
        # Extrair frequências para outras análises
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate
        )
        frequencies = f0[~np.isnan(f0)]
        
        # Classificação vocal
        mean_freq = pitch_analysis.get("mean_frequency_hz", 440)
        vocal_classification = classify_voice_type(mean_freq)
        
        # Análise de afinação
        intonation_accuracy = analyze_intonation(frequencies)
        
        # Análise de formantes
        formants = analyze_formants_with_parselmouth(audio_data, sample_rate)
        
        # Detectar vibrato
        time_array = np.linspace(0, len(audio_data) / sample_rate, len(frequencies))
        vibrato = detect_vibrato(frequencies, time_array)
        
        # Qualidade vocal
        voice_quality = analyze_voice_quality(audio_data, sample_rate)
        
        # Análise de respiração
        breathing_analysis = analyze_breathing(audio_data, sample_rate)
        
        return VocalAnalysisResponse(
            pitch_analysis=pitch_analysis,
            vocal_classification=vocal_classification,
            intonation_accuracy=intonation_accuracy,
            formants=formants,
            vibrato=vibrato,
            voice_quality=voice_quality,
            breathing_analysis=breathing_analysis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Vocal Analysis API",
        "version": "2.0.0",
        "using": "Librosa (without CREPE/TensorFlow)"
    }

# Endpoint simplificado para testes
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
