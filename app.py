import streamlit as st
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from textblob import TextBlob
from deep_translator import GoogleTranslator
import plotly.express as px
import time
import io
import csv as csv_module

DetectorFactory.seed = 0

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual Comment Analyzer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, .stMarkdown h1, .stMarkdown h2 { font-family: 'Syne', sans-serif !important; }

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem;
    text-align: center; position: relative; overflow: hidden;
}
.main-header::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(120,80,255,0.3), transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(0,210,255,0.2), transparent 50%);
}
.main-header h1 {
    color: #fff; font-family: 'Syne', sans-serif !important;
    font-size: 2.8rem; font-weight: 800; margin: 0;
    position: relative; z-index: 1; letter-spacing: -1px;
}
.main-header p {
    color: rgba(255,255,255,0.65); font-size: 1.05rem;
    margin: 0.5rem 0 0; position: relative; z-index: 1;
}

.file-info-badge {
    background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.35);
    border-radius: 10px; padding: 0.55rem 1rem; color: #38bdf8;
    font-size: 0.85rem; margin-bottom: 0.5rem; display: inline-block;
}

.metric-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1.4rem 1.6rem;
    text-align: center; color: white;
}
.metric-card .value {
    font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card .label {
    font-size: 0.82rem; text-transform: uppercase; letter-spacing: 1.5px;
    color: rgba(255,255,255,0.5); margin-top: 0.3rem;
}

.sentiment-positive { background: linear-gradient(135deg,#064e3b,#065f46); border-left: 4px solid #10b981; }
.sentiment-negative { background: linear-gradient(135deg,#450a0a,#7f1d1d); border-left: 4px solid #ef4444; }
.sentiment-neutral  { background: linear-gradient(135deg,#1e3a5f,#1e40af); border-left: 4px solid #3b82f6; }

.comment-card { border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: 0.8rem; color: white; }
.comment-card .original   { font-size: 1rem; font-weight: 500; }
.comment-card .translated { font-size: 0.9rem; color: rgba(255,255,255,0.65); font-style: italic; margin-top: 0.3rem; }
.comment-card .meta       { font-size: 0.78rem; margin-top: 0.5rem; color: rgba(255,255,255,0.45); }

.lang-badge {
    display: inline-block; background: rgba(167,139,250,0.2); color: #a78bfa;
    border: 1px solid rgba(167,139,250,0.4); border-radius: 6px;
    padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
    margin-right: 6px; font-family: 'Syne', sans-serif; letter-spacing: 0.5px;
}

.stTextArea textarea {
    background: #1a1a2e !important; color: white !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.6rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
    font-size: 1rem !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.5) !important;
}

div[data-testid="stSidebar"] { background: #0f0c29 !important; }

.footer {
    text-align: center; color: rgba(255,255,255,0.25); font-size: 0.78rem;
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.07);
}
</style>
""", unsafe_allow_html=True)

# ─── Language Code Map ─────────────────────────────────────────────────────────
LANG_NAMES = {
    "en":"English","hi":"Hindi","es":"Spanish","fr":"French","de":"German",
    "ar":"Arabic","zh-cn":"Chinese","ja":"Japanese","ko":"Korean","pt":"Portuguese",
    "ru":"Russian","it":"Italian","tr":"Turkish","nl":"Dutch","pl":"Polish",
    "sv":"Swedish","da":"Danish","fi":"Finnish","no":"Norwegian","id":"Indonesian",
    "vi":"Vietnamese","th":"Thai","fa":"Persian","ur":"Urdu","bn":"Bengali",
    "pa":"Punjabi","ta":"Tamil","te":"Telugu","ml":"Malayalam","gu":"Gujarati",
    "mr":"Marathi","kn":"Kannada","or":"Odia","si":"Sinhala",
}

def get_lang_name(code):
    return LANG_NAMES.get(code, code.upper() if code else "Unknown")

# ─── Core NLP Functions ────────────────────────────────────────────────────────
def detect_language(text):
    try:
        return detect(text.strip())
    except LangDetectException:
        return "unknown"

def analyze_sentiment(text_en):
    blob = TextBlob(text_en)
    polarity     = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0.05:
        label, category = "Positive 😊", "positive"
    elif polarity < -0.05:
        label, category = "Negative 😞", "negative"
    else:
        label, category = "Neutral 😐",  "neutral"
    return {"label": label, "category": category,
            "polarity": round(polarity, 3), "subjectivity": round(subjectivity, 3)}

def translate_to_english(text, src_lang):
    if src_lang in ("en", "unknown"):
        return text
    try:
        translated = GoogleTranslator(source=src_lang, target="en").translate(text)
        return translated or text
    except Exception:
        return text

def parse_uploaded_file(uploaded_file):
    """Parse a .txt or .csv upload → list of comment strings, skipping blank lines and # comments."""
    raw = uploaded_file.read().decode("utf-8")
    if uploaded_file.name.lower().endswith(".csv"):
        reader = csv_module.reader(io.StringIO(raw))
        lines  = [row[0].strip() for row in reader if row and row[0].strip() and not row[0].strip().startswith("#")]
    else:
        lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.strip().startswith("#")]
    return lines

# ─── Sample Data — fully mixed multilingual (mirrors comments.txt order) ───────
SAMPLE_COMMENTS = """El producto está bien, pero el precio me parece un poco elevado para lo que ofrece.
यह उत्पाद बहुत अच्छा है! मैं इसे सभी को recommend करता हूँ।
No lo recomendaría. Se rompió al mes de usarlo y la garantía no cubre nada.
मुझे यह प्रोडक्ट बिल्कुल पसंद नहीं आया। वापस कर दिया।
素晴らしい製品です！品質がとても高く、使うたびに満足しています。
太让人失望了。产品和图片完全不一样，退款流程也非常麻烦。
Customer support took 3 weeks to respond. Completely unacceptable in today's world.
পণ্যটি অসাধারণ! গুণমান অত্যন্ত উচ্চ এবং ডেলিভারি খুব দ্রুত ছিল। সবাইকে সুপারিশ করি।
बेकार प्रोडक्ट है, पैसे बर्बाद हो गए। कभी मत खरीदना।
Uitstekende klantenservice! Mijn probleem werd binnen een uur opgelost. Erg tevreden!
Totale Enttäuschung. Das Produkt funktioniert nicht wie beschrieben und der Kundendienst ist eine Katastrophe.
খুবই হতাশাজনক। পণ্যটি বিজ্ঞাপনের মতো নয় এবং রিটার্ন প্রক্রিয়া অনেক জটিল।
Adorei! Minha família toda já quer comprar um igual. Chegou muito antes do prazo!
Prodotto fantastico! La qualità è eccellente e la consegna è arrivata in anticipo.
تجربة سيئة للغاية. المنتج لا يعمل كما هو موضح وخدمة العملاء لا تستجيب.
خدمة العملاء كانت ممتازة وحلوا مشكلتي في وقت قصير جداً. شكراً جزيلاً!
Absolutely terrible experience from start to finish. The customer service was rude and the product broke within a week.
家族全員が気に入っています！毎日使っていて、本当に買ってよかったです。
Fiyat-performans oranı muhteşem! Bu kalite için bu fiyat gerçekten inanılmaz.
全家人都喜欢！每天都在用，买得非常值。下次还会再买。
Le produit ne correspond pas du tout à la description. Je demande un remboursement.
This changed my life! I use it every single day and can't imagine going back to before.
コスパ最高！この価格でこの品質は信じられません。絶対おすすめです。
Perfekt als Geschenk geeignet! Meine Mutter war total begeistert. Fünf Sterne!
Ótimo custo-benefício! Com certeza vou comprar de novo e indicar para os amigos.
Rapporto qualità-prezzo eccellente! Lo consiglio senza riserve a tutti.
¡Fantástico! Mi familia entera lo usa y todos están encantados. Volveremos a comprar.
性价比超高！这个价位能有这样的质量真的很难得，已经推荐给朋友了。
Продукт сломался через две недели. Очень низкое качество, не советую покупать.
Отличный продукт! Качество на высшем уровне, доставка быстрая. Рекомендую всем!
Çok hayal kırıklığı yarattı. Ürün fotoğraflarla hiç uyuşmuyor, iade etmek zorunda kaldım.
고객 서비스가 정말 친절했어요. 문제를 빠르게 해결해 주셔서 감사합니다.
İki haftada bozuldu. Bu kadar düşük kalite için bu fiyatı ödemek çok üzücü.
बहुत बढ़िया अनुभव रहा। अगली बार भी यहीं से खरीदूँगा।
Magnifique qualité, design élégant et très pratique au quotidien. Cinq étoiles!
The build quality feels premium and it looks exactly like the photos. Very satisfied!
দাম অনুযায়ী মান খুবই ভালো! আবার কিনবো এবং বন্ধুদেরকেও জানাবো।
Geweldig product! De kwaliteit is uitstekend en de levering was supersnel. Zeker aan te raden!
Atendimento ao cliente excelente! Resolveram meu problema em menos de uma hora.
Honestly one of the best investments I've made this year. Highly recommend to everyone!
配送が非常に早くて驚きました！梱包も丁寧で、製品も完璧な状態でした。
Não recomendo. A qualidade é muito inferior ao que foi anunciado nas fotos.
가성비 최고! 이 가격에 이런 품질은 정말 찾기 어려워요. 강력 추천합니다!
Le produit est good but delivery was very slow. Not happy about that part at all.
Ich bin begeistert! Seit Wochen benutze ich es täglich und bin vollkommen zufrieden.
It's an okay product. Does what it says on the box, nothing more, nothing less.
விலை மற்றும் தரத்தின் விகிதம் மிகவும் சிறந்தது! மீண்டும் வாங்குவேன்.
أفضل منتج اشتريته هذا العام! الجودة تفوق السعر بكثير. سأشتري مرة أخرى بالتأكيد.
لا أنصح بهذا المنتج أبداً. كسر بعد أسبوعين فقط من الاستخدام.
La livraison a pris trois semaines. C'est beaucoup trop long pour un produit standard.
जबरदस्त! मेरे परिवार को बहुत पसंद आया। पाँच में से पाँच स्टार।
Très bon rapport qualité-prix. Je suis agréablement surpris par les performances.
Hervorragendes Produkt! Die Qualität ist erstklassig und die Lieferung war blitzschnell.
Harika bir ürün! Kalitesi beklentilerimi aştı ve teslimat çok hızlıydı. Kesinlikle tavsiye ederim.
डिलीवरी बहुत जल्दी आई और पैकेजिंग भी बढ़िया थी। खुश हूँ।
Das Produkt ist okay but I expected better quality for this price honestly.
কাস্টমার সার্ভিস অসাধারণ ছিল! খুব দ্রুত সমস্যার সমাধান করে দিল।
¡Increíble producto! Superó todas mis expectativas, lo recomiendo ampliamente.
Péssima experiência. O produto chegou com defeito e o atendimento foi horrível.
Not great, not terrible. It works but I expected more for the price I paid.
Très déçu par la qualité. Pour ce prix, je m'attendais à beaucoup mieux.
Produto incrível! A qualidade superou todas as minhas expectativas. Recomendo muito!
The packaging was damaged when it arrived but the product itself seems fine so far.
Produit excellent, je suis vraiment satisfait de mon achat. Je le recommande vivement!
Das Produkt ist nach zwei Wochen kaputt gegangen. Absolut inakzeptable Qualität.
Schnelle Lieferung, gute Verpackung und das Produkt entspricht genau den Fotos. Top!
Decent product for the price. Won't blow your mind but gets the job done reliably.
最悪な買い物でした。すぐに壊れてしまい、カスタマーサポートも全く役に立ちませんでした。
हे उत्पादन खूपच उत्तम आहे! गुणवत्ता अप्रतिम आहे आणि डिलिव्हरी जलद होती.
ग्राहक सेवा उत्कृष्ट होती! माझी समस्या काही मिनिटांत सुटली. धन्यवाद!
Служба поддержки сработала молниеносно! Проблему решили за несколько часов. Спасибо!
இந்த தயாரிப்பு மிகவும் சிறந்தது! தரம் மிகவும் உயர்வானது மற்றும் டெலிவரி வேகமாக இருந்தது.
ग्राहक सेवा बहुत खराब है। घंटों फोन पर रहा, कोई जवाब नहीं मिला।
Esperienza pessima. Il prodotto non funziona e il servizio clienti è irreperibile.
Der Kundenservice hat nicht auf meine E-Mails geantwortet. Sehr unprofessionell.
આ ઉત્પાદ ખૂબ જ સરસ છે! ગુણવત્તા ઉત્તમ છે અને ડિલિવરી ઝડપી હતી. બધાને ભલામણ કરું છું.
क्वालिटी एकदम टॉप क्लास है! इतने कम दाम में इतनी अच्छी चीज़ नहीं मिलती।
This product completely exceeded my expectations! I've been using it for three months now and it just keeps getting better.
Geweldige prijs-kwaliteitsverhouding! Ik zou het zo opnieuw kopen zonder twijfel.
ખૂબ જ નિરાશાજનક. ઉત્પાદ ફોટા સાથે મેળ ખાતું નથી અને રિટર્ન પ્રક્રિયા ઘણી મુશ્કેલ છે.
선물용으로 구매했는데 받으신 분이 너무 좋아하셨어요. 다음에도 또 살게요.
Zeer teleurstellend. Het product werkt niet zoals beschreven en de klantenservice reageert niet.
정말 최고의 제품이에요! 품질도 훌륭하고 배송도 빨라서 매우 만족합니다.
ભાવ અને ગુણવત્તાનો ગુણોત્તર ઉત્તમ! ફરીથી ખરીદીશ.
Regalo perfetto! La mia famiglia è rimasta entusiasta. Cinque stelle meritate.
Muy decepcionante. La calidad es pésima y el servicio al cliente no responde.
客服态度非常好，很快就解决了我的问题。服务让我很满意！
வாடிக்கையாளர் சேவை அருமையாக இருந்தது! எனது பிரச்சனை சீக்கிரம் தீர்க்கப்பட்டது.
Соотношение цены и качества превосходное! Куплю ещё раз без колебаний.
منتج رائع جداً! الجودة ممتازة والتوصيل كان سريعاً جداً. أنصح به بشدة.
Excelente relación calidad-precio. Estoy muy satisfecho con mi compra.
¡Lo mejor que he comprado este año! Funciona perfectamente y llegó antes de lo esperado.
કસ્ટમર સર્વિસ ઉત્કૃષ્ટ હતી! સમસ્યા ઝડપથી ઉકેલાઈ ગઈ. આભાર!
Totally paisa vasool! Best product mila hai mujhe. Will buy again for sure.
Five stars without hesitation! The quality is outstanding and delivery was super fast.
완전 실망이에요. 사진이랑 실제 제품이 너무 달라요. 환불 요청했습니다.
Ужасный опыт. Товар пришёл повреждённым, а служба поддержки не реагирует уже неделю.
Returned it after two days. It didn't work as advertised and the refund process was painful.
这个产品非常棒！质量超出了我的预期，而且送货速度很快。强烈推荐！
किंमतीच्या तुलनेत गुणवत्ता खूप चांगली आहे! पुन्हा नक्की खरेदी करेन.
Müşteri hizmetleri mükemmeldi! Sorunum dakikalar içinde çözüldü, teşekkürler.
I bought this as a gift and my friend absolutely loved it. Will definitely buy again.
This product is ekdum best! Bilkul worth it hai, highly recommend karunga.
ठीक-ठाक है, कुछ खास नहीं। उम्मीद से कम निकला।
Tuve problemas con la entrega y nadie me ayudó. Una experiencia muy frustrante.
La atención al cliente fue amable y resolvieron mi problema en minutos. ¡Gracias!
Service client impeccable! Ils ont résolu mon problème en moins de 24 heures.
மிகவும் ஏமாற்றமாக இருந்தது. தயாரிப்பு படங்களுக்கு ஒத்திருக்கவில்லை, திரும்பப் பெற வேண்டியதாயிற்று.
I've never been so disappointed in a purchase. Total waste of money and time.
The instructions were unclear and I spent two hours trying to set it up. Very frustrating.
Il servizio clienti è stato straordinario! Hanno risolto tutto in poche ore. Ottimo!""".strip()

# ─── Session State Init ────────────────────────────────────────────────────────
if "comment_text"    not in st.session_state: st.session_state.comment_text    = SAMPLE_COMMENTS
if "uploaded_lines"  not in st.session_state: st.session_state.uploaded_lines  = []
if "uploaded_name"   not in st.session_state: st.session_state.uploaded_name   = ""

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🌐 Multilingual Comment Analyzer</h1>
  <p>Detect language · Translate · Analyze sentiment — all in one place</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    show_translation = st.toggle("Show translations", value=True)
    show_raw_table   = st.toggle("Show raw data table", value=False)
    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("""
    **Multilingual Comment Analyzer** detects language, translates to English,
    and runs sentiment analysis using TextBlob.

    Built with:
    - `langdetect` — language detection
    - `deep-translator` — Google Translate API
    - `TextBlob` — NLP sentiment
    - `Plotly` — interactive charts
    """)
    st.markdown("---")
    st.markdown("Made for college assignment 🎓")

# ─── Input Section ────────────────────────────────────────────────────────────
st.markdown("## 📝 Input Comments")

tab_type, tab_upload = st.tabs(["✏️ Type / Paste Comments", "📂 Upload a File"])

run_analysis = False
final_input  = []
source_label = ""

# ── Tab 1 : Type / Paste ──────────────────────────────────────────────────────
with tab_type:
    typed_input = st.text_area(
        "Enter comments (one per line):",
        value=st.session_state.comment_text,
        height=260,
        placeholder="Paste your comments here, one per line...",
        key="comment_textarea",
    )

    analyze_typed = st.button("🔍 Analyze", key="analyze_typed")

    if analyze_typed:
        final_input  = [c for c in typed_input.split("\n") if c.strip() and not c.strip().startswith("#")]
        source_label = "✏️ typed input"
        run_analysis = True

# ── Tab 2 : File Upload ───────────────────────────────────────────────────────
with tab_upload:
    st.markdown("Upload a **`.txt`** or **`.csv`** file with one comment per line.")
    st.caption("💡 Tip: You can directly upload the `comments.txt` file that came with this project!")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "csv"],
        label_visibility="collapsed",
        key="file_uploader",
    )

    if uploaded_file is not None:
        try:
            parsed = parse_uploaded_file(uploaded_file)
            st.session_state.uploaded_lines = parsed
            st.session_state.uploaded_name  = uploaded_file.name
        except Exception as e:
            st.error(f"Could not read file: {e}")
            parsed = []

    if st.session_state.uploaded_lines:
        parsed = st.session_state.uploaded_lines
        st.markdown(
            f'<div class="file-info-badge">📄 <b>{st.session_state.uploaded_name}</b>'
            f' &nbsp;—&nbsp; {len(parsed)} comments loaded</div>',
            unsafe_allow_html=True,
        )
        with st.expander("👀 Preview first 10 comments"):
            for i, line in enumerate(parsed[:10], 1):
                st.markdown(f"`{i}.` {line}")
            if len(parsed) > 10:
                st.caption(f"… and {len(parsed) - 10} more comments")

    upload_ready     = bool(st.session_state.uploaded_lines)
    analyze_uploaded = st.button(
        "🔍 Analyze Uploaded File",
        disabled=not upload_ready,
        key="analyze_uploaded",
    )
    if not upload_ready:
        st.caption("⬆️ Upload a file above to enable this button.")

    if analyze_uploaded and upload_ready:
        final_input  = st.session_state.uploaded_lines
        source_label = f"📄 {st.session_state.uploaded_name}"
        run_analysis = True

# ─── Analysis ─────────────────────────────────────────────────────────────────
if run_analysis:
    if not final_input:
        st.warning("No valid comments found. Please add some comments first.")
        st.stop()

    st.info(f"Analyzing **{len(final_input)} comments** from {source_label}…")

    with st.spinner("🔄 Detecting languages · Translating · Scoring sentiment…"):
        progress = st.progress(0)
        results  = []
        for i, comment in enumerate(final_input):
            lang_code  = detect_language(comment)
            lang_name  = get_lang_name(lang_code)
            translated = translate_to_english(comment, lang_code)
            sentiment  = analyze_sentiment(translated)
            results.append({
                "id":                 i + 1,
                "original":           comment,
                "translated":         translated if lang_code not in ("en","unknown") else None,
                "lang_code":          lang_code,
                "lang_name":          lang_name,
                "sentiment_label":    sentiment["label"],
                "sentiment_category": sentiment["category"],
                "polarity":           sentiment["polarity"],
                "subjectivity":       sentiment["subjectivity"],
            })
            progress.progress((i + 1) / len(final_input))
            time.sleep(0.03)
        progress.empty()

    df = pd.DataFrame(results)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("## 📊 Overview")
    pos     = len(df[df.sentiment_category == "positive"])
    neg     = len(df[df.sentiment_category == "negative"])
    neu     = len(df[df.sentiment_category == "neutral"])
    langs_n = df.lang_name.nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [len(df), langs_n, pos, neg, neu],
        ["Comments", "Languages", "Positive 😊", "Negative 😞", "Neutral 😐"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    color_map = {"positive":"#10b981","negative":"#ef4444","neutral":"#3b82f6"}

    with col_a:
        st.markdown("### 🎭 Sentiment Distribution")
        sc = df["sentiment_category"].value_counts().reset_index()
        sc.columns = ["Sentiment","Count"]
        fig_pie = px.pie(sc, names="Sentiment", values="Count",
                         color="Sentiment", color_discrete_map=color_map, hole=0.55)
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white",family="DM Sans"),
            legend=dict(font=dict(color="white")), margin=dict(t=10,b=10,l=10,r=10))
        fig_pie.update_traces(textfont_color="white")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("### 🌍 Language Breakdown")
        lc = df["lang_name"].value_counts().reset_index()
        lc.columns = ["Language","Count"]
        fig_bar = px.bar(lc, x="Language", y="Count",
                         color="Count", color_continuous_scale="Purples")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white",family="DM Sans"),
            xaxis=dict(tickfont=dict(color="white"),
                       gridcolor="rgba(255,255,255,0.05)", tickangle=-30),
            yaxis=dict(tickfont=dict(color="white"),
                       gridcolor="rgba(255,255,255,0.05)"),
            coloraxis_showscale=False, margin=dict(t=10,b=60))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 📈 Polarity vs Subjectivity")
    fig_scatter = px.scatter(
        df, x="polarity", y="subjectivity",
        color="sentiment_category", color_discrete_map=color_map,
        hover_data={"original":True,"lang_name":True}, size_max=16,
        labels={"polarity":"Polarity","subjectivity":"Subjectivity","sentiment_category":"Sentiment"})
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white",family="DM Sans"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickfont=dict(color="white"),
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickfont=dict(color="white")),
        legend=dict(font=dict(color="white")), margin=dict(t=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Comment Cards ─────────────────────────────────────────────────────────
    st.markdown("## 💬 Comment Details")

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        sentiment_filter = st.multiselect(
            "Filter by sentiment:",
            options=["positive","negative","neutral"],
            default=["positive","negative","neutral"])
    with fcol2:
        lang_options = sorted(df["lang_name"].unique().tolist())
        lang_filter  = st.multiselect("Filter by language:", options=lang_options, default=lang_options)

    filtered_df = df[df["sentiment_category"].isin(sentiment_filter) & df["lang_name"].isin(lang_filter)]
    st.caption(f"Showing {len(filtered_df)} of {len(df)} comments")

    for _, row in filtered_df.iterrows():
        tr_html = ""
        if show_translation and row["translated"]:
            tr_html = f'<div class="translated">🔄 {row["translated"]}</div>'
        st.markdown(f"""
        <div class="comment-card sentiment-{row['sentiment_category']}">
            <div class="original">{row['original']}</div>
            {tr_html}
            <div class="meta">
                <span class="lang-badge">{row['lang_name']}</span>
                <b>{row['sentiment_label']}</b> &nbsp;|&nbsp;
                Polarity: <b>{row['polarity']}</b> &nbsp;|&nbsp;
                Subjectivity: <b>{row['subjectivity']}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Raw Table ──────────────────────────────────────────────────────────────
    if show_raw_table:
        st.markdown("## 📋 Raw Data")
        st.dataframe(
            df[["id","original","lang_name","sentiment_label","polarity","subjectivity"]],
            use_container_width=True)

    # ── Export ─────────────────────────────────────────────────────────────────
    st.markdown("## 💾 Export Results")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("⬇️ Download Full CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="comment_analysis_full.csv", mime="text/csv")
    with dl2:
        st.download_button("⬇️ Download Filtered CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="comment_analysis_filtered.csv", mime="text/csv")

st.markdown(
    '<div class="footer">Multilingual Comment Analyzer · Built with Streamlit 🚀</div>',
    unsafe_allow_html=True)
