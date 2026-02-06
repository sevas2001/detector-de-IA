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
        except Exception as e:
            print(f"Warning: Could not load roberta-base-openai-detector. {e}")
            self.classifier = None

        # 2. Modelo Causal para cálculo de Perplejidad (GPT-2 Small para rapidez)
        # Usamos GPT-2 para medir qué tan "sorprendido" estaría un modelo básico por el texto.
        self.ppl_model_id = "gpt2"
        self.ppl_tokenizer = AutoTokenizer.from_pretrained(self.ppl_model_id)
        self.ppl_model = AutoModelForCausalLM.from_pretrained(self.ppl_model_id).to(self.device)

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

        # --- Métrica 3: Clasificador Pipeline ---
        if self.classifier:
            try:
                # Analizamos hasta 3 chunks para tener mejor cobertura
                class_scores = []
                for chunk in chunks[:3]:
                    res = self.classifier(chunk[:512*4], truncation=True, max_length=512)[0]
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

if __name__ == "__main__":
    detector = AIDetector()
    text_ia = "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence."
    print("Test IA:", detector.analyze_text(text_ia))
