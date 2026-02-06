import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import math

class AIDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading models on {self.device}...")
        
        # 1. Pipeline de clasificación (Roberta-OpenAI-Detector)
        # Este modelo es un clásico para detectar GPT-2, a veces funciona para otros.
        # Podríamos cambiarlo por uno más moderno ajustado en HC3 si encontramos uno público ligero.
        try:
            self.classifier = pipeline(
                "text-classification", 
                model="roberta-base-openai-detector", 
                device=0 if self.device == "cuda" else -1
            )
            self.classifier_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
        except Exception as e:
            print(f"Warning: Could not load roberta-base-openai-detector. {e}")
            self.classifier = None
            self.classifier_tokenizer = None

        # 2. Modelo Causal para cálculo de Perplejidad (GPT-2 Small para rapidez)
        self.ppl_model_id = "gpt2"
        self.ppl_tokenizer = AutoTokenizer.from_pretrained(self.ppl_model_id)
        self.ppl_model = AutoModelForCausalLM.from_pretrained(self.ppl_model_id).to(self.device)

        # 3. Modelo T5 para Perturbaciones (Fase 3: Deep Scan)
        # Cargamos T5-small bajo demanda o inicio si hay memoria, para DetectGPT-lite
        try:
            from transformers import AutoModelForSeq2SeqLM
            self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(self.device)
        except Exception as e:
             print(f"Warning: Could not load t5-small. {e}")
             self.t5_model = None

    def stylometric_analysis(self, text):
        """
        Analiza patrones estilísticos simples:
        - Riqueza léxica (Type-Token Ratio)
        - Uso de conectores lógicos comunes en IA
        """
        words = text.lower().split()
        if not words: return {}
        
        # 1. Riqueza Léxica (TTR)
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # 2. Conectores de IA (lista heurística)
        ai_connectors = ["sin embargo", "por lo tanto", "además", "envl conclusión", "es importante destacar", "en resumen"]
        connector_count = sum(1 for c in ai_connectors if c in text.lower())
        
        return {
            "ttr": ttr,
            "connector_count": connector_count,
            "connector_density": connector_count / len(words)
        }

    def generate_perturbation(self, text, span_length=3):
        """
        Genera una variación ligera del texto usando T5 (mask filling simulado o paráfrasis simple).
        Para MVP usaremos T5 para reescribir frases cortas.
        """
        if not self.t5_model: return text
        
        # Simulación: T5 summarization como "reescritura" rápida
        # Un verdadero DetectGPT usa mask filling, pero T5-small es mejor resumiendo/traduciendo.
        # Truco: Traducir a Aleman y volver a Ingles (Backtranslation) es lento.
        # Usaremos el modo 'summarize' con parámetros relajados para variar el texto.
        input_ids = self.t5_tokenizer("paraphrase: " + text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.t5_model.generate(input_ids, max_length=len(text.split()), do_sample=True, temperature=0.9)
        return self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def calculate_perplexity(self, text):
        """
        Calcula la perplejidad del texto usando GPT-2.
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        encodings = self.ppl_tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        
        # Stride window approach simple
        max_length = self.ppl_model.config.n_positions
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]
            
        with torch.no_grad():
            outputs = self.ppl_model(input_ids, labels=input_ids)
            loss = outputs.loss
            
        return torch.exp(loss).item()

    def get_sentence_perplexities(self, text):
        """
        Divide el texto en oraciones y calcula PPL para cada una.
        Utili para gráfica de Burstiness.
        """
        # División simple por puntos (se podría mejorar con nltk/spacy)
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
        ppl_values = []
        
        for s in sentences:
            try:
                ppl = self.calculate_perplexity(s)
                ppl_values.append(ppl)
            except:
                pass
                
        return sentences, ppl_values

    def analyze_text(self, text):
        results = {}
        
        # --- Métrica 1: Perplejidad Global (con Chunking) ---
        # Dividimos el texto en chunks de aprox 512 palabras para no saturar memoria
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        chunk_ppls = []
        for chunk in chunks:
            chunk_ppls.append(self.calculate_perplexity(chunk))
        
        avg_ppl = sum(chunk_ppls) / len(chunk_ppls) if chunk_ppls else 0
        results['perplexity'] = avg_ppl
        
        # Heurística simple
        if avg_ppl < 40:
            results['ppl_verdict'] = "Probable IA (Texto muy predecible)"
            results['ppl_score'] = max(0, 100 - avg_ppl) 
        else:
            results['ppl_verdict'] = "Probable Humano (Texto complejo/variado)"
            results['ppl_score'] = max(0, 100 - (avg_ppl * 0.5)) 

        # --- Métrica 2: Burstiness (Variación Frase a Frase) ---
        sentences, sentence_ppls = self.get_sentence_perplexities(text[:5000]) # Limitamos analisis visual
        results['burstiness_data'] = {
            'sentences': sentences,
            'scores': sentence_ppls
        }

        # --- Métrica 3: Clasificador Pipeline con Sliding Window (Mejora Fase 2.1) ---
        if self.classifier:
            try:
                # Tokenizamos el texto completo usando el tokenizer del clasificador
                # Esto es mucho más seguro que cortar por caracteres
                tokens = self.classifier_tokenizer(text, return_tensors="pt", truncation=False, padding=False).input_ids[0]
                
                # Configuración de Sliding Window
                window_size = 510 # Dejamos espacio para tokens especiales
                stride = 256 # Solape del 50% para no perder contexto entre cortes
                
                chunks_text = []
                for i in range(0, len(tokens), stride):
                    # Cortamos ventana de tokens
                    chunk_tokens = tokens[i : i + window_size]
                    if len(chunk_tokens) < 10: break # Ignorar fragmentos muy pequeños al final
                    
                    # Decodificamos de nuevo a texto para el pipeline
                    chunk_str = self.classifier_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    chunks_text.append(chunk_str)
                    
                    # Limitamos a analizar primero 5 chunks para rendimiento (MVP)
                    if len(chunks_text) >= 5: break

                if not chunks_text:
                    # Fallback si por alguna razón no hay chunks
                    chunks_text = [text[:1000]]

                class_scores = []
                for chunk_str in chunks_text:
                    # Ahora chunk_str garantiza tener < 512 tokens
                    res = self.classifier(chunk_str, truncation=True, max_length=512)[0]
                    score = res['score'] if res['label'] == 'Fake' else (1 - res['score'])
                    class_scores.append(score)
                
                avg_fake_score = sum(class_scores) / len(class_scores)
                
                if avg_fake_score > 0.5:
                     results['classifier_label'] = "Fake"
                     results['classifier_score'] = avg_fake_score
                else:
                     results['classifier_label'] = "Real"
                     results['classifier_score'] = 1 - avg_fake_score

            except Exception as e:
                 print(f"Error classifier: {e}")
                 results['classifier_label'] = "Error"
                 results['classifier_score'] = 0.0
        
        return results

    def deep_scan(self, text):
        """
        Fase 3: Análisis profundo con DetectGPT-lite y Estilometría.
        """
        results = {}
        
        # 1. Estilometría
        style_metrics = self.stylometric_analysis(text)
        results['style'] = style_metrics
        
        # 2. DetectGPT-Lite (Perturbaciones)
        # Tomamos una frase representativa (no todo el texto por velocidad)
        sentences = text.split('.')
        sample_sentence = max(sentences, key=len).strip() if sentences else text[:200]
        
        if len(sample_sentence) > 50 and self.t5_model:
            original_ppl = self.calculate_perplexity(sample_sentence)
            
            # Generamos 3 perturbaciones
            perturbations = [self.generate_perturbation(sample_sentence) for _ in range(3)]
            pert_ppls = [self.calculate_perplexity(p) for p in perturbations]
            avg_pert_ppl = sum(pert_ppls) / len(pert_ppls) if pert_ppls else original_ppl
            
            # Lógica DetectGPT: Si PPL sube mucho al reescribir => Humano (era óptimo local)
            # Si PPL baja o se mantiene => IA (era genérico)
            # *Corrección Teórica*: DetectGPT dice que IA está en curvatura negativa,
            # así que sus perturbaciones tienen MAYOR perplejidad (menor prob).
            # Texto humano tiene MENOR cambio relativo.
            
            ppl_ratio = avg_pert_ppl / (original_ppl + 1e-5)
            results['detectgpt'] = {
                'original_ppl': original_ppl,
                'perturbed_ppl': avg_pert_ppl,
                'ratio': ppl_ratio, # >1.0 indica que original era mejor (posible IA), ~1.0 humano
                'verdict': "IA (Frágil)" if ppl_ratio > 1.2 else "Humano (Robusto)"
            }
        else:
             results['detectgpt'] = {'verdict': "No disponible (Texto corto/Sin modelo)"}
             
        return results

if __name__ == "__main__":
    detector = AIDetector()
    text_ia = "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence."
    print("Test IA:", detector.analyze_text(text_ia))

