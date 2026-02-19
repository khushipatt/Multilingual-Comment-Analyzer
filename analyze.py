"""
analyze.py — Multilingual Comment Analyzer · VS Code / Terminal

Usage:
    python analyze.py                          # interactive chooser (pick input method)
    python analyze.py --sample                 # run built-in mixed multilingual samples
    python analyze.py --file comments.txt      # analyze any .txt or .csv file
    python analyze.py --text "Great! यह बहुत अच्छा है!"
    python analyze.py --file comments.txt --save              # save results to results.csv
    python analyze.py --file comments.txt --save --out my.csv # custom output filename
    python analyze.py --file comments.txt --limit 20          # analyze first 20 only
"""

import argparse
import sys
import os
import csv
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from textblob import TextBlob
from deep_translator import GoogleTranslator
from collections import Counter

DetectorFactory.seed = 0

# ─── Language Map ──────────────────────────────────────────────────────────────
LANG_NAMES = {
    "en":"English","hi":"Hindi","es":"Spanish","fr":"French","de":"German",
    "ar":"Arabic","zh-cn":"Chinese","ja":"Japanese","ko":"Korean","pt":"Portuguese",
    "ru":"Russian","it":"Italian","tr":"Turkish","nl":"Dutch","pl":"Polish",
    "sv":"Swedish","da":"Danish","fi":"Finnish","no":"Norwegian","id":"Indonesian",
    "vi":"Vietnamese","th":"Thai","fa":"Persian","ur":"Urdu","bn":"Bengali",
    "pa":"Punjabi","ta":"Tamil","te":"Telugu","ml":"Malayalam","gu":"Gujarati",
    "mr":"Marathi","kn":"Kannada","or":"Odia","si":"Sinhala",
}

# ─── Terminal Colors ───────────────────────────────────────────────────────────
COLORS = {
    "positive":"\033[92m","negative":"\033[91m","neutral":"\033[94m",
    "reset":"\033[0m","bold":"\033[1m","dim":"\033[2m",
    "cyan":"\033[96m","yellow":"\033[93m","magenta":"\033[95m","white":"\033[97m",
}

def c(text, *keys):
    return "".join(COLORS[k] for k in keys) + str(text) + COLORS["reset"]

def get_lang_name(code):
    return LANG_NAMES.get(code, code.upper() if code else "Unknown")

# ─── Core NLP ─────────────────────────────────────────────────────────────────
def detect_language(text):
    try:
        return detect(text.strip())
    except LangDetectException:
        return "unknown"

def translate_to_english(text, src_lang):
    if src_lang in ("en", "unknown"):
        return text
    try:
        result = GoogleTranslator(source=src_lang, target="en").translate(text)
        return result or text
    except Exception:
        return text

def analyze_sentiment(text_en):
    blob = TextBlob(text_en)
    polarity     = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0.05:
        label, cat = "Positive 😊", "positive"
    elif polarity < -0.05:
        label, cat = "Negative 😞", "negative"
    else:
        label, cat = "Neutral  😐", "neutral"
    return {"label": label, "category": cat,
            "polarity": round(polarity, 3), "subjectivity": round(subjectivity, 3)}

def analyze_comment(comment):
    lang_code  = detect_language(comment)
    lang_name  = get_lang_name(lang_code)
    translated = translate_to_english(comment, lang_code)
    sentiment  = analyze_sentiment(translated)
    return {
        "original":   comment,
        "translated": translated if lang_code not in ("en","unknown") else None,
        "lang_code":  lang_code,
        "lang_name":  lang_name,
        **sentiment,
    }

# ─── File Loaders ─────────────────────────────────────────────────────────────
def load_txt(path):
    with open(path, encoding="utf-8") as f:
        # skip blank lines and comment lines starting with #
        return [line.strip() for line in f
                if line.strip() and not line.strip().startswith("#")]

def load_csv(path):
    comments = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            val = row[0].strip() if row else ""
            if val and not val.startswith("#"):
                comments.append(val)
    return comments

def load_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: '{path}'")
    ext = os.path.splitext(path)[1].lower()
    return load_csv(path) if ext == ".csv" else load_txt(path)

# ─── Display Helpers ──────────────────────────────────────────────────────────
def print_banner():
    print(f"\n{'═'*62}")
    print(c("  🌐  MULTILINGUAL COMMENT ANALYZER", "bold", "cyan"))
    print(f"  {c('Detect · Translate · Sentiment', 'dim')}")
    print(f"{'═'*62}\n")

def print_result(idx, r, total):
    idx_str = c(f"[{idx:>3}/{total}]", "bold", "yellow")
    print(f"\n{idx_str}  {r['original']}")
    if r["translated"]:
        print(f"           {c('→ EN:', 'dim')} {c(r['translated'], 'dim')}")
    lang_str = c(f"[{r['lang_name']}]", "cyan")
    sent_str = c(r["label"], r["category"], "bold")
    print(f"           {lang_str}  {sent_str}  "
          f"pol={c(r['polarity'],'bold')}  subj={c(r['subjectivity'],'bold')}")

def print_summary(results):
    cats  = Counter(r["category"] for r in results)
    langs = Counter(r["lang_name"] for r in results)
    avg_p = sum(r["polarity"] for r in results) / len(results)

    print(f"\n{'─'*62}")
    print(c("  SUMMARY", "bold", "cyan"))
    print(f"{'─'*62}")
    print(f"  Total    : {c(len(results), 'bold', 'white')}")
    print(f"  Positive : {c(cats.get('positive',0), 'positive', 'bold')}")
    print(f"  Negative : {c(cats.get('negative',0), 'negative', 'bold')}")
    print(f"  Neutral  : {c(cats.get('neutral', 0), 'neutral',  'bold')}")
    print(f"\n  Languages detected ({len(langs)}):")
    for lang, cnt in langs.most_common():
        bar = "█" * cnt
        print(f"    {lang:<22} {c(bar, 'magenta')} {cnt}")
    print(f"\n  Avg polarity : {c(round(avg_p, 3), 'bold')}")
    print(f"{'─'*62}\n")

def save_results(results, output_path="results.csv"):
    fieldnames = ["original","translated","lang_code","lang_name",
                  "label","category","polarity","subjectivity"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(c(f"  ✅ Results saved → {output_path}", "positive", "bold"))

# ─── Interactive Chooser (no flags given) ─────────────────────────────────────
def interactive_chooser():
    print(c("  How would you like to provide comments?\n", "white"))
    print(f"  {c('[1]','bold','yellow')} Use built-in sample comments  {c('(113 mixed-language comments)','dim')}")
    print(f"  {c('[2]','bold','yellow')} Load from a file              {c('(.txt or .csv — e.g. comments.txt)','dim')}")
    print(f"  {c('[3]','bold','yellow')} Type / paste comments now     {c('(press Enter twice when done)','dim')}")
    print()

    choice = input(c("  Your choice (1 / 2 / 3): ", "bold")).strip()

    if choice == "1":
        return SAMPLE_COMMENTS, "built-in samples"

    elif choice == "2":
        path = input(c("  File path: ", "bold")).strip().strip('"').strip("'")
        try:
            comments = load_file(path)
            print(c(f"\n  ✔ Loaded {len(comments)} comments from '{os.path.basename(path)}'\n", "positive"))
            return comments, os.path.basename(path)
        except FileNotFoundError as e:
            print(c(f"\n  ✖ {e}", "negative", "bold"))
            sys.exit(1)

    elif choice == "3":
        print(c("  Type comments one per line. Press Enter on a blank line when done:\n", "dim"))
        lines = []
        while True:
            try:
                line = input("  > ")
                if line.strip() == "" and lines:
                    break
                if line.strip():
                    lines.append(line.strip())
            except EOFError:
                break
        if not lines:
            print(c("  No comments entered. Exiting.", "negative"))
            sys.exit(0)
        return lines, "manual input"

    else:
        print(c("  Invalid choice — running with built-in samples.\n", "dim"))
        return SAMPLE_COMMENTS, "built-in samples"

# ─── Built-in Sample — fully mixed multilingual (matches comments.txt) ─────────
SAMPLE_COMMENTS = [
    "El producto está bien, pero el precio me parece un poco elevado para lo que ofrece.",
    "यह उत्पाद बहुत अच्छा है! मैं इसे सभी को recommend करता हूँ।",
    "No lo recomendaría. Se rompió al mes de usarlo y la garantía no cubre nada.",
    "मुझे यह प्रोडक्ट बिल्कुल पसंद नहीं आया। वापस कर दिया।",
    "素晴らしい製品です！品質がとても高く、使うたびに満足しています。",
    "太让人失望了。产品和图片完全不一样，退款流程也非常麻烦。",
    "Customer support took 3 weeks to respond. Completely unacceptable in today's world.",
    "পণ্যটি অসাধারণ! গুণমান অত্যন্ত উচ্চ এবং ডেলিভারি খুব দ্রুত ছিল। সবাইকে সুপারিশ করি।",
    "बेकार प्रोडक्ट है, पैसे बर्बाद हो गए। कभी मत खरीदना।",
    "Uitstekende klantenservice! Mijn probleem werd binnen een uur opgelost. Erg tevreden!",
    "Totale Enttäuschung. Das Produkt funktioniert nicht wie beschrieben und der Kundendienst ist eine Katastrophe.",
    "খুবই হতাশাজনক। পণ্যটি বিজ্ঞাপনের মতো নয় এবং রিটার্ন প্রক্রিয়া অনেক জটিল।",
    "Adorei! Minha família toda já quer comprar um igual. Chegou muito antes do prazo!",
    "Prodotto fantastico! La qualità è eccellente e la consegna è arrivata in anticipo.",
    "تجربة سيئة للغاية. المنتج لا يعمل كما هو موضح وخدمة العملاء لا تستجيب.",
    "خدمة العملاء كانت ممتازة وحلوا مشكلتي في وقت قصير جداً. شكراً جزيلاً!",
    "Absolutely terrible experience from start to finish. The customer service was rude and the product broke within a week.",
    "家族全員が気に入っています！毎日使っていて、本当に買ってよかったです。",
    "Fiyat-performans oranı muhteşem! Bu kalite için bu fiyat gerçekten inanılmaz.",
    "全家人都喜欢！每天都在用，买得非常值。下次还会再买。",
    "Le produit ne correspond pas du tout à la description. Je demande un remboursement.",
    "This changed my life! I use it every single day and can't imagine going back to before.",
    "コスパ最高！この価格でこの品質は信じられません。絶対おすすめです。",
    "Perfekt als Geschenk geeignet! Meine Mutter war total begeistert. Fünf Sterne!",
    "Ótimo custo-benefício! Com certeza vou comprar de novo e indicar para os amigos.",
    "Rapporto qualità-prezzo eccellente! Lo consiglio senza riserve a tutti.",
    "¡Fantástico! Mi familia entera lo usa y todos están encantados. Volveremos a comprar.",
    "性价比超高！这个价位能有这样的质量真的很难得，已经推荐给朋友了。",
    "Продукт сломался через две недели. Очень низкое качество, не советую покупать.",
    "Отличный продукт! Качество на высшем уровне, доставка быстрая. Рекомендую всем!",
    "Çok hayal kırıklığı yarattı. Ürün fotoğraflarla hiç uyuşmuyor, iade etmek zorunda kaldım.",
    "고객 서비스가 정말 친절했어요. 문제를 빠르게 해결해 주셔서 감사합니다.",
    "İki haftada bozuldu. Bu kadar düşük kalite için bu fiyatı ödemek çok üzücü.",
    "बहुत बढ़िया अनुभव रहा। अगली बार भी यहीं से खरीदूँगा।",
    "Magnifique qualité, design élégant et très pratique au quotidien. Cinq étoiles!",
    "The build quality feels premium and it looks exactly like the photos. Very satisfied!",
    "দাম অনুযায়ী মান খুবই ভালো! আবার কিনবো এবং বন্ধুদেরকেও জানাবো।",
    "Geweldig product! De kwaliteit is uitstekend en de levering was supersnel. Zeker aan te raden!",
    "Atendimento ao cliente excelente! Resolveram meu problema em menos de uma hora.",
    "Honestly one of the best investments I've made this year. Highly recommend to everyone!",
    "配送が非常に早くて驚きました！梱包も丁寧で、製品も完璧な状態でした。",
    "Não recomendo. A qualidade é muito inferior ao que foi anunciado nas fotos.",
    "가성비 최고! 이 가격에 이런 품질은 정말 찾기 어려워요. 강력 추천합니다!",
    "Le produit est good but delivery was very slow. Not happy about that part at all.",
    "Ich bin begeistert! Seit Wochen benutze ich es täglich und bin vollkommen zufrieden.",
    "It's an okay product. Does what it says on the box, nothing more, nothing less.",
    "விலை மற்றும் தரத்தின் விகிதம் மிகவும் சிறந்தது! மீண்டும் வாங்குவேன்.",
    "أفضل منتج اشتريته هذا العام! الجودة تفوق السعر بكثير. سأشتري مرة أخرى بالتأكيد.",
    "لا أنصح بهذا المنتج أبداً. كسر بعد أسبوعين فقط من الاستخدام.",
    "La livraison a pris trois semaines. C'est beaucoup trop long pour un produit standard.",
    "जबरदस्त! मेरे परिवार को बहुत पसंद आया। पाँच में से पाँच स्टार।",
    "Très bon rapport qualité-prix. Je suis agréablement surpris par les performances.",
    "Hervorragendes Produkt! Die Qualität ist erstklassig und die Lieferung war blitzschnell.",
    "Harika bir ürün! Kalitesi beklentilerimi aştı ve teslimat çok hızlıydı. Kesinlikle tavsiye ederim.",
    "डिलीवरी बहुत जल्दी आई और पैकेजिंग भी बढ़िया थी। खुश हूँ।",
    "Das Produkt ist okay but I expected better quality for this price honestly.",
    "কাস্টমার সার্ভিস অসাধারণ ছিল! খুব দ্রুত সমস্যার সমাধান করে দিল।",
    "¡Increíble producto! Superó todas mis expectativas, lo recomiendo ampliamente.",
    "Péssima experiência. O produto chegou com defeito e o atendimento foi horrível.",
    "Not great, not terrible. It works but I expected more for the price I paid.",
    "Très déçu par la qualité. Pour ce prix, je m'attendais à beaucoup mieux.",
    "Produto incrível! A qualidade superou todas as minhas expectativas. Recomendo muito!",
    "The packaging was damaged when it arrived but the product itself seems fine so far.",
    "Produit excellent, je suis vraiment satisfait de mon achat. Je le recommande vivement!",
    "Das Produkt ist nach zwei Wochen kaputt gegangen. Absolut inakzeptable Qualität.",
    "Schnelle Lieferung, gute Verpackung und das Produkt entspricht genau den Fotos. Top!",
    "Decent product for the price. Won't blow your mind but gets the job done reliably.",
    "最悪な買い物でした。すぐに壊れてしまい、カスタマーサポートも全く役に立ちませんでした。",
    "हे उत्पादन खूपच उत्तम आहे! गुणवत्ता अप्रतिम आहे आणि डिलिव्हरी जलद होती.",
    "ग्राहक सेवा उत्कृष्ट होती! माझी समस्या काही मिनिटांत सुटली. धन्यवाद!",
    "Служба поддержки сработала молниеносно! Проблему решили за несколько часов. Спасибо!",
    "இந்த தயாரிப்பு மிகவும் சிறந்தது! தரம் மிகவும் உயர்வானது மற்றும் டெலிவரி வேகமாக இருந்தது.",
    "ग्राहक सेवा बहुत खराब है। घंटों फोन पर रहा, कोई जवाब नहीं मिला।",
    "Esperienza pessima. Il prodotto non funziona e il servizio clienti è irreperibile.",
    "Der Kundenservice hat nicht auf meine E-Mails geantwortet. Sehr unprofessionell.",
    "આ ઉત્પાદ ખૂબ જ સરસ છે! ગુણવત્તા ઉત્તમ છે અને ડિલિવરી ઝડપી હતી. બધાને ભલામણ કરું છું.",
    "क्वालिटी एकदम टॉप क्लास है! इतने कम दाम में इतनी अच्छी चीज़ नहीं मिलती।",
    "This product completely exceeded my expectations! I've been using it for three months now and it just keeps getting better.",
    "Geweldige prijs-kwaliteitsverhouding! Ik zou het zo opnieuw kopen zonder twijfel.",
    "ખૂબ જ નિરાશાજનક. ઉત્પાદ ફોટા સાથે મેળ ખાતું નથી અને રિટર્ન પ્રક્રિયા ઘણી મુશ્કેલ છે.",
    "선물용으로 구매했는데 받으신 분이 너무 좋아하셨어요. 다음에도 또 살게요.",
    "Zeer teleurstellend. Het product werkt niet zoals beschreven en de klantenservice reageert niet.",
    "정말 최고의 제품이에요! 품질도 훌륭하고 배송도 빨라서 매우 만족합니다.",
    "ભાવ અને ગુણવત્તાનો ગુણોત્તર ઉત્તમ! ફરીથી ખરીદીશ.",
    "Regalo perfetto! La mia famiglia è rimasta entusiasta. Cinque stelle meritate.",
    "Muy decepcionante. La calidad es pésima y el servicio al cliente no responde.",
    "客服态度非常好，很快就解决了我的问题。服务让我很满意！",
    "வாடிக்கையாளர் சேவை அருமையாக இருந்தது! எனது பிரச்சனை சீக்கிரம் தீர்க்கப்பட்டது.",
    "Соотношение цены и качества превосходное! Куплю ещё раз без колебаний.",
    "منتج رائع جداً! الجودة ممتازة والتوصيل كان سريعاً جداً. أنصح به بشدة.",
    "Excelente relación calidad-precio. Estoy muy satisfecho con mi compra.",
    "¡Lo mejor que he comprado este año! Funciona perfectamente y llegó antes de lo esperado.",
    "કસ્ટમર સર્વિસ ઉત્કૃષ્ટ હતી! સમસ્યા ઝડપથી ઉકેલાઈ ગઈ. આભાર!",
    "Totally paisa vasool! Best product mila hai mujhe. Will buy again for sure.",
    "Five stars without hesitation! The quality is outstanding and delivery was super fast.",
    "완전 실망이에요. 사진이랑 실제 제품이 너무 달라요. 환불 요청했습니다.",
    "Ужасный опыт. Товар пришёл повреждённым, а служба поддержки не реагирует уже неделю.",
    "Returned it after two days. It didn't work as advertised and the refund process was painful.",
    "这个产品非常棒！质量超出了我的预期，而且送货速度很快。强烈推荐！",
    "किंमतीच्या तुलनेत गुणवत्ता खूप चांगली आहे! पुन्हा नक्की खरेदी करेन.",
    "Müşteri hizmetleri mükemmeldi! Sorunum dakikalar içinde çözüldü, teşekkürler.",
    "I bought this as a gift and my friend absolutely loved it. Will definitely buy again.",
    "This product is ekdum best! Bilkul worth it hai, highly recommend karunga.",
    "ठीक-ठाक है, कुछ खास नहीं। उम्मीद से कम निकला।",
    "Tuve problemas con la entrega y nadie me ayudó. Una experiencia muy frustrante.",
    "La atención al cliente fue amable y resolvieron mi problema en minutos. ¡Gracias!",
    "Service client impeccable! Ils ont résolu mon problème en moins de 24 heures.",
    "மிகவும் ஏமாற்றமாக இருந்தது. தயாரிப்பு படங்களுக்கு ஒத்திருக்கவில்லை, திரும்பப் பெற வேண்டியதாயிற்று.",
    "I've never been so disappointed in a purchase. Total waste of money and time.",
    "The instructions were unclear and I spent two hours trying to set it up. Very frustrating.",
    "Il servizio clienti è stato straordinario! Hanno risolto tutto in poche ore. Ottimo!",
]

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multilingual Comment Analyzer — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py                           # interactive chooser
  python analyze.py --sample                  # built-in 113 mixed-language samples
  python analyze.py --file comments.txt       # load from text file
  python analyze.py --file data.csv           # load from CSV (uses first column)
  python analyze.py --text "Bonjour monde!"   # single comment
  python analyze.py --file comments.txt --save               # save → results.csv
  python analyze.py --file comments.txt --save --out out.csv # custom output name
  python analyze.py --file comments.txt --limit 30           # first 30 only
        """,
    )
    parser.add_argument("--file",   help="Path to a .txt or .csv file (one comment per line)")
    parser.add_argument("--text",   help="Analyze a single comment string directly")
    parser.add_argument("--sample", action="store_true", help="Use built-in mixed multilingual samples")
    parser.add_argument("--save",   action="store_true", help="Save results to a CSV file")
    parser.add_argument("--out",    default="results.csv", help="Output CSV filename (default: results.csv)")
    parser.add_argument("--limit",  type=int, default=None, help="Only analyze the first N comments")
    args = parser.parse_args()

    print_banner()

    # ── Determine input source ────────────────────────────────────────────────
    if args.text:
        comments     = [args.text]
        source_label = "single text input"

    elif args.sample:
        comments     = SAMPLE_COMMENTS
        source_label = "built-in samples"
        print(c(f"  Using built-in sample ({len(comments)} mixed-language comments)\n", "dim"))

    elif args.file:
        try:
            comments     = load_file(args.file)
            source_label = os.path.basename(args.file)
            print(c(f"  ✔ Loaded {len(comments)} comments from '{source_label}'\n", "positive"))
        except FileNotFoundError as e:
            print(c(f"  ✖ {e}", "negative", "bold"))
            sys.exit(1)

    else:
        # No flags → show interactive menu
        comments, source_label = interactive_chooser()
        print()

    # ── Apply limit ───────────────────────────────────────────────────────────
    if args.limit and args.limit < len(comments):
        print(c(f"  Limiting to first {args.limit} comments.\n", "dim"))
        comments = comments[:args.limit]

    if not comments:
        print(c("  No comments to analyze. Exiting.", "negative"))
        sys.exit(0)

    # ── Analyze ───────────────────────────────────────────────────────────────
    print(c(f"  Analyzing {len(comments)} comment(s) from: {source_label}\n", "white"))
    results = []
    for i, comment in enumerate(comments, 1):
        print(f"  {c('Processing','dim')} {i}/{len(comments)}…", end="\r")
        results.append(analyze_comment(comment.strip()))
    print(" " * 50, end="\r")

    # ── Print results ─────────────────────────────────────────────────────────
    for i, r in enumerate(results, 1):
        print_result(i, r, len(results))

    print_summary(results)

    # ── Optionally save ───────────────────────────────────────────────────────
    if args.save:
        save_results(results, args.out)

if __name__ == "__main__":
    main()
