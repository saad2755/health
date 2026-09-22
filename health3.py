# -*- coding: utf-8 -*-
"""
Assistant santé — Interface Streamlit RAG
"""

import os
import re
import datetime
import pandas as pd
import streamlit as st
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# --------------------------------------------------------------------------
# Configuration générale
# --------------------------------------------------------------------------

st.set_page_config(page_title="Assistant santé", layout="wide")

THEMES = {
    "Sommeil": {
        "fichier": "mieux-dormir-inpes.pdf",
        "description": {
            "fr": "Conseils pour mieux dormir (brochure INPES).",
            "ar": "نصائح للنوم بشكل أفضل (كتيب INPES).",
            "dar": "نصائح باش تنعس مزيان (كتيب INPES)."
        },
    },
    "Readaptation cardiaque": {
        "fichier": "APA-readapt-cardiaque.pdf",
        "description": {
            "fr": "Livret d'activité physique adaptée après réadaptation cardiaque (CHU Montpellier).",
            "ar": "كتيب النشاط البدني المكيف بعد إعادة تأهيل القلب (CHU Montpellier).",
            "dar": "كتيب الرياضة والنشاط البدني بعد إعادة تأهيل القلب (CHU Montpellier)."
        },
    },
    "Nutrition": {
        "fichier": "Polycopié_cours_nutrition_HZ_2.pdf",
        "description": {
            "fr": "Polycopié de cours de nutrition.",
            "ar": "مطبوع دروس التغذية.",
            "dar": "مطبوع دروس التغذية."
        },
    },
}

THEME_META = {
    "Sommeil": {"sujet": "sommeil"},
    "Readaptation cardiaque": {"sujet": "réadaptation cardiaque"},
    "Nutrition": {"sujet": "nutrition"},
}

TEXTES = {
    "fr": {
        "titre_app": "Assistant santé",
        "sous_titre_app": "Réponses fondées uniquement sur vos documents.",
        "langue_label": "Langue / اللغة",
        "theme_label": "Thème du document :",
        "reglages": "Réglages avancés",
        "k_label": "Nombre de passages récupérés (k)",
        "groq_key_label": "Clé API Groq",
        "err_doc_intro": "Le fichier est introuvable. Placez vos 3 PDF directement dans le même dossier que app.py :",
        "warn_key": "Veuillez renseigner votre clé API Groq dans les réglages avancés pour commencer.",
        "apercu": "Aperçu",
        "passages_indexes": "Passages indexés",
        "questions_session": "Questions (session)",
        "score_moyen": "Score de pertinence moy.",
        "pertinence_titre": "Pertinence des réponses",
        "pertinence_sub": "Score de similarité moyen des passages retrouvés",
        "empty_chart": "Posez une question à l'assistant pour voir apparaître le suivi de pertinence ici.",
        "passages_count": "passages",
        "effacer_histo": "Effacer l'historique de ce thème",
        "assistant_sub": "Les réponses sont générées uniquement à partir du document sélectionné.",
        "empty_chat": "Aucune question pour l'instant. Écrivez votre première question ci-dessous.",
        "pages_sources": "Pages sources :",
        "voir_passages": "Voir les passages utilisés",
        "input_placeholder": "Posez votre question ici...",
        "thinking": "Recherche dans le document puis génération de la réponse...",
        "indexing": "Indexation du document en cours...",
        "prompt_langue_inst": "Réponds impérativement en français."
    },
    "ar": {
        "titre_app": "المساعد الصحي",
        "sous_titre_app": "إجابات معتمدة حصريا على وثائقك.",
        "langue_label": "اللغة / Langue",
        "theme_label": "موضوع الوثيقة:",
        "reglages": "إعدادات متقدمة",
        "k_label": "عدد الفقرات المسترجعة (k)",
        "groq_key_label": "مفتاح Groq API",
        "err_doc_intro": "الملف غير موجود. يرجى وضع ملفات PDF في نفس المجلد مع app.py:",
        "warn_key": "يرجى إدخال مفتاح Groq API في الإعدادات المتقدمة للبدء.",
        "apercu": "نظرة عامة",
        "passages_indexes": "المقاطع المفهرسة",
        "questions_session": "الأسئلة (الجلسة)",
        "score_moyen": "متوسط درجة الملاءمة",
        "pertinence_titre": "ملاءمة الإجابات",
        "pertinence_sub": "متوسط درجة التشابه للمقاطع المسترجعة",
        "empty_chart": "اطرح سؤالاً على المساعد لرؤية تتبع الملاءمة هنا.",
        "passages_count": "مقاطع",
        "effacer_histo": "مسح سجل هذا الموضوع",
        "assistant_sub": "تم إنشاء الإجابات حصريًا بناءً على الوثيقة المحددة.",
        "empty_chat": "لا توجد أسئلة حتى الآن. اكتب سؤالك الأول أدناه.",
        "pages_sources": "الصفحات المصدر:",
        "voir_passages": "عرض المقاطع المستخدمة",
        "input_placeholder": "اطرح سؤالك هنا...",
        "thinking": "جاري البحث في الوثيقة وإنشاء الإجابة...",
        "indexing": "جاري فهرسة الوثيقة...",
        "prompt_langue_inst": "أجب باللغة العربية الفصحى بشكل واضح ودقيق."
    },
    "dar": {
        "titre_app": "المساعد الصحي",
        "sous_titre_app": "أجوبة معتمدة غير على الوثائق ديالك.",
        "langue_label": "اللغة / Langue",
        "theme_label": "موضوع الوثيقة:",
        "reglages": "إعدادات متقدمة",
        "k_label": "عدد الفقرات المأخوذة (k)",
        "groq_key_label": "مفتاح Groq API",
        "err_doc_intro": "الملف مالقيناهش. حط ملفات PDF ف نفس المجلد مع app.py:",
        "warn_key": "دخل مفتاح Groq API فالإعدادات المتقدمة باش تبدأ.",
        "apercu": "نظرة عامة",
        "passages_indexes": "الفقرات المفهرسة",
        "questions_session": "الأسئلة (هاد الجلسة)",
        "score_moyen": "معدل دقة الأجوبة",
        "pertinence_titre": "دقة الأجوبة",
        "pertinence_sub": "معدل التشابه د الفقرات المأخوذة",
        "empty_chart": "سول المساعد باش تبان لك دقة الأجوبة هنا.",
        "passages_count": "فقرات",
        "effacer_histo": "مسح الأرشيف د هاد الموضوع",
        "assistant_sub": "الأجوبة مصاوبة غير من الوثيقة اللي عزليتي.",
        "empty_chat": "ماكاين حتى سؤال لحد الآن. اكتب السؤال الأول ديالك لتحت.",
        "pages_sources": "الصفحات المصدر:",
        "voir_passages": "شوف الفقرات المستعملة",
        "input_placeholder": "سول السؤال ديالك هنا...",
        "thinking": "كانهزو المعلومات من الوثيقة ونقادو الجواب...",
        "indexing": "جاري تحضير وفهرسة الوثيقة...",
        "prompt_langue_inst": "أجب بالدارجة المغربية بشكل واضح وبسيط ومفهوم."
    }
}

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-20b"


# --------------------------------------------------------------------------
# Fonctions RAG
# --------------------------------------------------------------------------

def extraire_pages(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []
    for numero, page in enumerate(reader.pages, start=1):
        texte = page.extract_text() or ""
        texte = re.sub(r"\s+", " ", texte).strip()
        pages.append({"page": numero, "texte": texte})
    return pages


def decouper_texte(texte, taille=900, chevauchement=150):
    morceaux = []
    debut = 0
    while debut < len(texte):
        fin = min(debut + taille, len(texte))
        if fin < len(texte):
            espace = texte.rfind(" ", debut, fin)
            if espace > debut:
                fin = espace
        morceau = texte[debut:fin].strip()
        if morceau:
            morceaux.append(morceau)
        if fin == len(texte):
            break
        debut = max(fin - chevauchement, debut + 1)
    return morceaux


def construire_chunks(pages):
    chunks = []
    for page in pages:
        for i, texte_chunk in enumerate(decouper_texte(page["texte"]), start=1):
            chunks.append({"page": page["page"], "chunk": i, "texte": texte_chunk})
    return chunks


@st.cache_resource(show_spinner=False)
def charger_modele_embedding():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def construire_index(pdf_path: str):
    modele = charger_modele_embedding()
    pages = extraire_pages(pdf_path)
    chunks = construire_chunks(pages)
    textes_chunks = [c["texte"] for c in chunks]

    vecteurs = modele.encode(
        textes_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(vecteurs.shape[1])
    index.add(vecteurs)

    return {"index": index, "chunks": chunks, "modele": modele}


def rechercher(ressources, question, k=4):
    modele = ressources["modele"]
    index = ressources["index"]
    chunks = ressources["chunks"]

    vecteur_question = modele.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, indices = index.search(vecteur_question, k)

    resultats = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            item = chunks[int(idx)].copy()
            item["score"] = float(score)
            resultats.append(item)
    return resultats


def construire_prompt(question, passages, langue_code):
    contexte = "\n\n".join(
        f'[Source : page {p["page"]}]\n{p["texte"]}' for p in passages
    )
    instruction_langue = TEXTES[langue_code]["prompt_langue_inst"]
    
    return f"""Tu es un assistant santé.
{instruction_langue}
Réponds uniquement à partir du CONTEXTE fourni.
Donne une réponse courte, claire et fidèle au document.
Cite la ou les pages sous la forme (page X).
Si la réponse exacte n'est pas présente, écris que l'information n'est pas précisée dans le document.
N'invente jamais une valeur numérique et ne complète pas avec tes connaissances.
Cet assistant ne remplace pas un avis médical professionnel.

CONTEXTE :
{contexte}

QUESTION : {question}

RÉPONSE :"""


def repondre(client, ressources, question, langue_code, k=4):
    passages = rechercher(ressources, question, k=k)
    prompt = construire_prompt(question, passages, langue_code)

    completion = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    texte_reponse = completion.choices[0].message.content
    pages_sources = sorted({p["page"] for p in passages})
    return texte_reponse, passages, pages_sources


# --------------------------------------------------------------------------
# Animation Jaune & Styles CSS Harmonisés
# --------------------------------------------------------------------------

def afficher_animation_jaune(message):
    st.markdown(
        f"""
        <div class="yellow-loader-container">
            <div class="yellow-spinner"></div>
            <div class="yellow-loader-text">{message}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def injecter_css(is_rtl=False):
    direction = "rtl" if is_rtl else "ltr"
    text_align = "right" if is_rtl else "left"
    
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            direction: {direction};
            text-align: {text_align};
        }}

        .stApp {{
            background: #F4F6F9;
        }}

        section.main > div.block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }}

        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }}

        .page-eyebrow {{
            color: #64748B;
            font-size: 0.85rem;
            margin-bottom: 2px;
        }}
        .page-title {{
            font-size: 1.9rem;
            font-weight: 800;
            color: #1E293B;
            margin: 0 0 4px 0;
        }}

        /* Rectangle Langue Stylisé */
        .lang-card-box {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 12px 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            margin-bottom: 16px;
        }}
        .lang-card-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            font-weight: 700;
            color: #475569;
            margin-bottom: 6px;
        }}

        /* Message d'avertissement avec clé tournante */
        .key-warning-box {{
            background-color: #FEF9C3;
            border: 1px solid #FDE047;
            border-radius: 12px;
            padding: 16px 20px;
            display: flex;
            align-items: center;
            gap: 14px;
            color: #854D0E;
            font-weight: 600;
            font-size: 0.95rem;
            margin-top: 12px;
        }}
        .key-icon-spin {{
            font-size: 1.5rem;
            display: inline-block;
            animation: spinKey 2s linear infinite;
        }}
        @keyframes spinKey {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        /* Carte principale de discussion unifiée (fond clair) */
        .chat-container-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
            margin-bottom: 12px;
        }}

        .chat-header-title {{
            font-size: 1.15rem;
            font-weight: 800;
            color: #0F172A;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .chat-header-badge {{
            background-color: #E0F2FE;
            color: #0369A1;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 20px;
            text-transform: uppercase;
        }}

        .chat-header-sub {{
            font-size: 0.83rem;
            color: #64748B;
            margin-top: 4px;
        }}

        /* Zone d'affichage des messages */
        div[data-testid="stColumn"]:nth-child(2) div[data-testid="stChatMessageContainer"] {{
            background-color: transparent !important;
        }}

        div[data-testid="stColumn"]:nth-child(2) div[data-testid="stChatMessage"] {{
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            color: #0F172A !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
            margin-bottom: 10px;
        }}

        div[data-testid="stColumn"]:nth-child(2) div[data-testid="stChatMessageContent"] p {{
            color: #0F172A !important;
        }}

        /* Style de la zone de saisie (st.chat_input) */
        div[data-testid="stColumn"]:nth-child(2) div[data-testid="stChatInput"] {{
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
        }}

        div[data-testid="stColumn"]:nth-child(2) div[data-testid="stChatInput"]:focus-within {{
            border-color: #0284C7 !important;
            box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15) !important;
        }}

        .empty-chat-msg {{
            color: #94A3B8;
            font-size: 0.88rem;
            padding: 20px 0;
            font-style: italic;
        }}

        /* Animation Jaune */
        .yellow-loader-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 24px;
            text-align: center;
        }}
        .yellow-spinner {{
            width: 40px;
            height: 40px;
            border: 4px solid #FEF08A;
            border-top: 4px solid #EAB308;
            border-radius: 50%;
            animation: spin 0.9s linear infinite;
            margin-bottom: 10px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .yellow-loader-text {{
            color: #EAB308;
            font-size: 0.88rem;
            font-weight: 600;
        }}

        /* Cartes Statistiques */
        .card-stat {{
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(0,0,0,0.05);
        }}
        .card-stat-1 {{ background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 100%); color: #0369A1; }}
        .card-stat-2 {{ background: linear-gradient(135deg, #FFE4E6 0%, #FECDD3 100%); color: #BE123C; }}
        .card-stat-3 {{ background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%); color: #15803D; }}

        .m-value {{ font-size: 1.8rem; font-weight: 800; line-height: 1.1; }}
        .m-label {{ font-size: 0.85rem; font-weight: 600; margin-top: 4px; opacity: 0.85; }}

        .card-heading {{ font-weight: 700; color: #1E293B; font-size: 1.05rem; margin-bottom: 2px; }}
        .card-subheading {{ color: #64748B; font-size: 0.82rem; margin-bottom: 12px; }}

        .source-pill {{
            background: #F1F5F9;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 4px 12px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #1E293B;
            display: inline-block;
            margin-top: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Interface Principale
# --------------------------------------------------------------------------

def main():
    langues_options = {
        "Français": "fr",
        "العربية (Arabe)": "ar",
        "الدارجة (Darija)": "dar"
    }

    with st.sidebar:
        st.markdown(
            """
            <div class="lang-card-box">
                <div class="lang-card-header">
                    <span>🌐</span>
                    <span>Sélection / اللغة</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        langue_selectionnee = st.selectbox("", list(langues_options.keys()), label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        langue_code = langues_options[langue_selectionnee]
        txt = TEXTES[langue_code]

        st.markdown(f"**{txt['titre_app']}**")
        st.caption(txt["sous_titre_app"])
        st.divider()

        theme_choisi = st.radio(txt["theme_label"], list(THEMES.keys()))

        with st.expander(txt["reglages"]):
            k = st.slider(txt["k_label"], min_value=1, max_value=10, value=4)
            api_key = st.text_input(
                txt["groq_key_label"],
                type="password",
                value=os.environ.get("GROQ_API_KEY", ""),
                help="Obtenez une clé gratuite sur https://console.groq.com/keys",
            )

    is_rtl = langue_code in ["ar", "dar"]
    injecter_css(is_rtl=is_rtl)

    meta = THEME_META[theme_choisi]
    pdf_path = THEMES[theme_choisi]["fichier"]

    st.markdown(f'<div class="page-eyebrow">{txt["apercu"]} — {meta["sujet"].capitalize()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{theme_choisi}</div>', unsafe_allow_html=True)
    date_str = datetime.date.today().strftime("%d/%m/%Y")
    st.caption(date_str)

    if not os.path.exists(pdf_path):
        st.error(
            f"{txt['err_doc_intro']}\n"
            f"- {THEMES['Sommeil']['fichier']}\n"
            f"- {THEMES['Readaptation cardiaque']['fichier']}\n"
            f"- {THEMES['Nutrition']['fichier']}"
        )
        st.stop()

    # Animation Clé en Rotation lors du besoin de clé API
    if not api_key:
        st.markdown(
            f"""
            <div class="key-warning-box">
                <span class="key-icon-spin">🔑</span>
                <span>{txt['warn_key']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()

    loader_placeholder = st.empty()
    with loader_placeholder.container():
        afficher_animation_jaune(txt["indexing"])
        ressources = construire_index(pdf_path)
    loader_placeholder.empty()

    client = Groq(api_key=api_key)

    if "historique" not in st.session_state:
        st.session_state.historique = {}
    if theme_choisi not in st.session_state.historique:
        st.session_state.historique[theme_choisi] = []

    historique_theme = st.session_state.historique[theme_choisi]
    nb_questions = len(historique_theme)
    scores_moyens = [h["score_moyen"] for h in historique_theme if h.get("score_moyen") is not None]
    score_moyen_global = round(sum(scores_moyens) / len(scores_moyens), 2) if scores_moyens else "—"

    st.write("")
    col_gauche, col_droite = st.columns([1.2, 1], gap="large")

    # ---------------------------------------------------------------
    # Gauche — Statistiques
    # ---------------------------------------------------------------
    with col_gauche:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""
                <div class="card-stat card-stat-1">
                    <div class="m-value">{len(ressources["chunks"])}</div>
                    <div class="m-label">{txt["passages_indexes"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="card-stat card-stat-2">
                    <div class="m-value">{nb_questions}</div>
                    <div class="m-label">{txt["questions_session"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="card-stat card-stat-3">
                    <div class="m-value">{score_moyen_global}</div>
                    <div class="m-label">{txt["score_moyen"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        with st.container(border=True):
            st.markdown(f'<div class="card-heading">{txt["pertinence_titre"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card-subheading">{txt["pertinence_sub"]}</div>', unsafe_allow_html=True)
            if scores_moyens:
                df_scores = pd.DataFrame(
                    {"Score": scores_moyens},
                    index=[f"Q{i+1}" for i in range(len(scores_moyens))],
                )
                st.bar_chart(df_scores, height=190)
            else:
                st.markdown(f'<div class="empty-state">{txt["empty_chart"]}</div>', unsafe_allow_html=True)

        with st.container(border=True):
            desc_theme = THEMES[theme_choisi]["description"].get(langue_code, THEMES[theme_choisi]["description"]["fr"])
            st.markdown(
                f"""
                <div>
                    <div class="card-heading">{os.path.basename(pdf_path)}</div>
                    <div class="card-subheading">{desc_theme}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f'<span class="source-pill">{len(ressources["chunks"])} {txt["passages_count"]}</span>', unsafe_allow_html=True)

        st.write("")
        if st.button(txt["effacer_histo"]):
            st.session_state.historique[theme_choisi] = []
            st.rerun()

    # ---------------------------------------------------------------
    # Droite — Bloc de discussion clair et assorti au site
    # ---------------------------------------------------------------
    with col_droite:
        st.markdown(
            f"""
            <div class="chat-container-card">
                <div class="chat-header-title">
                    <span>{txt["titre_app"]}</span>
                    <span class="chat-header-badge">{theme_choisi}</span>
                </div>
                <div class="chat-header-sub">{txt["assistant_sub"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.container(height=380):
            if not historique_theme:
                st.markdown(f'<div class="empty-chat-msg">{txt["empty_chat"]}</div>', unsafe_allow_html=True)
            for item in historique_theme:
                with st.chat_message("user"):
                    st.write(item["question"])
                with st.chat_message("assistant"):
                    st.write(item["reponse"])
                    if item["pages"]:
                        st.caption(f"{txt['pages_sources']} {', '.join(map(str, item['pages']))}")
                    with st.expander(txt["voir_passages"]):
                        for p in item["passages"]:
                            st.markdown(f"**Page {p['page']} — score {p['score']:.3f}**")
                            st.write(p["texte"][:500] + ("..." if len(p["texte"]) > 500 else ""))

        question = st.chat_input(txt["input_placeholder"])

        if question:
            chat_loader_placeholder = st.empty()
            with chat_loader_placeholder.container():
                afficher_animation_jaune(txt["thinking"])
                try:
                    reponse, passages, pages_sources = repondre(client, ressources, question, langue_code, k=k)
                except Exception as e:
                    chat_loader_placeholder.empty()
                    st.error(f"Erreur Groq API : {e}")
                    st.stop()
            chat_loader_placeholder.empty()

            score_moyen = round(sum(p["score"] for p in passages) / len(passages), 3) if passages else None

            st.session_state.historique[theme_choisi].append(
                {
                    "question": question,
                    "reponse": reponse,
                    "pages": pages_sources,
                    "passages": passages,
                    "score_moyen": score_moyen,
                }
            )
            st.rerun()


if __name__ == "__main__":
    main()