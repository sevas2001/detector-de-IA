import streamlit as st
import time
from src.utils.pdf_processor import extract_text_from_pdf
from src.models.detector import AIDetector

# Configuración de página
st.set_page_config(
    page_title="Detector de IA Pro",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS para estética "Premium"
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF4B4B, #FF914D);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
    }
    .metric-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00C9FF;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar detector en cache para no recargar modelos
@st.cache_resource
def load_detector():
    return AIDetector()

st.title("🤖 Detector de Texto IA")
st.markdown("### Analiza documentos y detecta huellas de Inteligencia Artificial")

# Sidebar
with st.sidebar:
    st.header("Configuración")
    st.info("Este prototipo usa modelos locales (GPT-2) para calcular la 'perplejidad' del texto. Una perplejidad baja sugiere que fue escrito por otra IA.")
    st.warning("⚠️ Nota: Los detectores no son 100% precisos. Úsalo como una herramienta de apoyo.")

# Carga de modelo
with st.spinner("Cargando cerebro digital... (Esto puede tardar unos segundos la primera vez)"):
    detector = load_detector()

# Tabs para diferentes modos de entrada
tab1, tab2 = st.tabs(["📄 Subir PDF", "✍️ Pegar Texto"])

text_to_analyze = ""

with tab1:
    uploaded_file = st.file_uploader("Arrastra tu PDF aquí", type="pdf")
    if uploaded_file is not None:
        with st.status("Procesando documento...", expanded=True) as status:
            st.write("Extrayendo texto del PDF...")
            time.sleep(1) # UX Simulation
            text_to_analyze = extract_text_from_pdf(uploaded_file)
            st.write("✅ Texto extraído con éxito!")
            status.update(label="Documento listo", state="complete", expanded=False)
        
        st.expander("Ver texto extraído").text_area("", text_to_analyze, height=150)

with tab2:
    text_input = st.text_area("Escribe o pega tu texto aquí", height=200)
    if st.button("Analizar Texto Pegado"):
        text_to_analyze = text_input

# Análisis
if text_to_analyze:
    if len(text_to_analyze) < 50:
        st.error("El texto es muy corto para un análisis fiable. Por favor proporciona al menos unas oraciones.")
    else:
        st.divider()
        st.subheader("🔍 Resultados del Análisis")
        
        with st.spinner("Escaneando patrones neuronales..."):
            # Barra de progreso falsa para UX
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
                
            results = detector.analyze_text(text_to_analyze)

        # Mostrar resultados visuales (KPIs)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Perplejidad Promedio</h4>
                <h2>{results['perplexity']:.2f}</h2>
                <p>{results['ppl_verdict']}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Clasificador RoBERTa</h4>
                <h2>{results.get('classifier_label', 'N/A')}</h2>
                <p>Confianza: {results.get('classifier_score', 0):.2%}</p>
            </div>
            """, unsafe_allow_html=True)

        # GRAFICO DE BURSTINESS (NUEVO FASE 2)
        st.subheader("📉 Análisis de Burstiness (Línea de Pensamiento)")
        burst_data = results.get('burstiness_data', {})
        if burst_data and burst_data.get('scores'):
            import plotly.express as px
            import pandas as pd
            
            df_burst = pd.DataFrame({
                'Oración': range(1, len(burst_data['scores']) + 1),
                'Perplejidad': burst_data['scores'],
                'Texto': burst_data['sentences']
            })
            
            fig = px.line(df_burst, x='Oración', y='Perplejidad', markers=True, 
                          title='Variación de Perplejidad por Frase (Más picos = Más Humano)',
                          hover_data=['Texto'])
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Interpretación final
        st.markdown("### 🧠 Veredicto Final")
        
        is_fake_roberta = results.get('classifier_label') == 'Fake'
        low_ppl = results['perplexity'] < 40
        
        final_verdict = "HUMANO"
        if is_fake_roberta or low_ppl:
            final_verdict = "IA"
            st.error("🚨 Alta probabilidad de contenido generado por IA detectada.")
            st.markdown("""
            **Señales encontradas:**
            * El texto tiene una estructura muy predecible (Baja perplejidad).
            * Los patrones coinciden con modelos de lenguaje conocidos.
            """)
        else:
            st.success("✅ El texto parece tener características de escritura humana.")
            st.markdown("""
            **Señales encontradas:**
            * Variabilidad y sorpresas en la estructura (Alta perplejidad).
            * No se detectaron patrones obvios de generación sintética.
            """)

        # FEEDBACK LOOP (NUEVO FASE 2)
        st.divider()
        st.markdown("#### 🎯 ¿Acertamos?")
        col_f1, col_f2, col_f3 = st.columns([1,1,3])
        with col_f1:
            if st.button("👍 Correcto"):
                st.toast("¡Gracias por entrenarme! Datos guardados.")
                # Aquí guardaríamos en CSV
        with col_f2:
            if st.button("👎 Incorrecto"):
                st.toast("Anotado. Analizaremos este fallo.")
                # Aquí guardaríamos en CSV

else:
    st.info("👈 Sube un archivo o pega texto para comenzar.")
