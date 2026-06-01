import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import re
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockPulse | News Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

.stApp { background: #080B10; color: #C8D6E5; }
[data-testid="stSidebar"] { background: #0D1117; border-right: 1px solid #1C2333; }

.hero {
    background: linear-gradient(135deg, #0D1117 0%, #111827 50%, #0D1117 100%);
    border: 1px solid #1C2333;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,210,120,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #E8F4F0;
    letter-spacing: -0.03em;
    margin: 0;
}
.hero-sub { font-size: 0.9rem; color: #4A5568; margin-top: 0.3rem; }

.signal-box {
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    border: 1px solid;
    margin-bottom: 1rem;
}
.signal-bullish {
    background: rgba(0, 210, 120, 0.08);
    border-color: rgba(0, 210, 120, 0.3);
}
.signal-bearish {
    background: rgba(255, 75, 75, 0.08);
    border-color: rgba(255, 75, 75, 0.3);
}
.signal-neutral {
    background: rgba(200, 180, 50, 0.08);
    border-color: rgba(200, 180, 50, 0.3);
}
.signal-label { font-size: 0.75rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem; }
.signal-value { font-size: 2.2rem; font-weight: 700; font-family: 'Space Mono', monospace; letter-spacing: -0.02em; }
.signal-sub { font-size: 0.8rem; margin-top: 0.3rem; opacity: 0.7; }

.news-card {
    background: #0D1117;
    border: 1px solid #1C2333;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    border-left: 3px solid;
    transition: all 0.2s;
}
.news-card:hover { border-color: #2A3A50; }
.news-bull { border-left-color: #00D278; }
.news-bear { border-left-color: #FF4B4B; }
.news-neut { border-left-color: #C8B432; }
.news-title { font-size: 13px; font-weight: 500; color: #C8D6E5; margin-bottom: 4px; line-height: 1.5; }
.news-meta { font-size: 11px; color: #4A5568; display: flex; gap: 12px; }
.score-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.pill-bull { background: rgba(0,210,120,0.15); color: #00D278; }
.pill-bear { background: rgba(255,75,75,0.15); color: #FF4B4B; }
.pill-neut { background: rgba(200,180,50,0.15); color: #C8B432; }

.stat-card {
    background: #0D1117;
    border: 1px solid #1C2333;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.stat-val { font-size: 1.6rem; font-weight: 700; font-family: 'Space Mono', monospace; }
.stat-lbl { font-size: 11px; color: #4A5568; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 2px; }

.section-hdr {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4A5568;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1C2333;
}
.api-box {
    background: #0D1117;
    border: 1px solid #1C2333;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-size: 13px;
    color: #4A5568;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB DARK THEME ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0D1117',
    'axes.facecolor': '#0D1117',
    'axes.edgecolor': '#1C2333',
    'axes.labelcolor': '#4A5568',
    'xtick.color': '#4A5568',
    'ytick.color': '#4A5568',
    'text.color': '#C8D6E5',
    'grid.color': '#1C2333',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'figure.dpi': 130
})

# ─── STOCK PRESETS ─────────────────────────────────────────────────────────────
STOCK_PRESETS = {
    "🇮🇳 Reliance Industries (RELIANCE)": "Reliance Industries stock NSE",
    "🇮🇳 TCS (TCS.NS)":                   "TCS Tata Consultancy Services stock",
    "🇮🇳 Infosys (INFY)":                 "Infosys stock NSE earnings",
    "🇮🇳 HDFC Bank (HDFCBANK)":           "HDFC Bank stock NSE",
    "🇮🇳 Wipro (WIPRO)":                  "Wipro stock NSE results",
    "🇺🇸 Apple (AAPL)":                   "Apple AAPL stock earnings",
    "🇺🇸 Tesla (TSLA)":                   "Tesla TSLA stock",
    "🇺🇸 Microsoft (MSFT)":               "Microsoft MSFT stock earnings",
    "🇺🇸 NVIDIA (NVDA)":                  "NVIDIA NVDA stock AI",
    "Custom":                              "",
}

# ─── NEWS FETCHER ──────────────────────────────────────────────────────────────
def fetch_news_newsapi(query, api_key, days=15):
    """Fetch news from NewsAPI.org"""
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={requests.utils.quote(query)}"
        f"&from={from_date}"
        f"&sortBy=publishedAt"
        f"&language=en"
        f"&pageSize=30"
        f"&apiKey={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get('status') == 'ok':
            articles = data.get('articles', [])
            news = []
            for a in articles:
                if a.get('title') and a['title'] != '[Removed]':
                    news.append({
                        'title': a['title'],
                        'description': a.get('description', '') or '',
                        'source': a.get('source', {}).get('name', 'Unknown'),
                        'published': a.get('publishedAt', '')[:10],
                        'url': a.get('url', '')
                    })
            return news, None
        else:
            return [], data.get('message', 'API error')
    except Exception as e:
        return [], str(e)


def generate_demo_news(query, days=15):
    """Generate realistic demo news when no API key provided"""
    import random
    random.seed(42)

    # Extract stock name from query
    stock = query.split()[0] if query else "Stock"

    templates = [
        (f"{stock} shares surge {random.randint(2,8)}% after strong quarterly earnings beat", "bull", 0.82),
        (f"Analysts raise {stock} target price, cite strong fundamentals", "bull", 0.74),
        (f"{stock} announces major expansion, investor confidence grows", "bull", 0.69),
        (f"{stock} reports record revenue, margin improvement surprises market", "bull", 0.85),
        (f"Foreign institutional investors increase holdings in {stock}", "bull", 0.61),
        (f"{stock} secures large contract, stock rises on positive outlook", "bull", 0.73),
        (f"Brokerage upgrades {stock} to Buy, sees 25% upside potential", "bull", 0.78),
        (f"{stock} stock dips {random.randint(1,4)}% amid global market pressure", "bear", -0.52),
        (f"Concerns over {stock} valuation as stock trades at premium multiples", "bear", -0.44),
        (f"{stock} faces headwinds in key segment, analysts turn cautious", "bear", -0.61),
        (f"Regulatory scrutiny weighs on {stock} amid sector-wide concerns", "bear", -0.55),
        (f"{stock} Q{random.randint(1,4)} margins disappoint, cost pressure visible", "bear", -0.58),
        (f"{stock} management change raises governance questions for investors", "bear", -0.47),
        (f"Market volatility impacts {stock}, short-term outlook uncertain", "neut", 0.05),
        (f"{stock} consolidates near key support levels, awaiting trigger", "neut", 0.08),
        (f"Analysts divided on {stock} prospects ahead of policy announcement", "neut", -0.03),
        (f"{stock} volume picks up, technical indicators show mixed signals", "neut", 0.12),
    ]

    base_date = datetime.now()
    sources = ["Economic Times", "Moneycontrol", "Business Standard",
               "LiveMint", "NDTV Profit", "Reuters", "Bloomberg", "CNBC"]
    news = []
    random.shuffle(templates)

    for i, (title, sentiment, score) in enumerate(templates[:15]):
        pub_date = base_date - timedelta(days=random.randint(0, days-1))
        news.append({
            'title': title,
            'description': f"Market analysts and investors closely watched {stock} as {title.lower()}.",
            'source': random.choice(sources),
            'published': pub_date.strftime('%Y-%m-%d'),
            'url': '#',
            '_demo_sentiment': sentiment,
            '_demo_score': score
        })

    news.sort(key=lambda x: x['published'], reverse=True)
    return news


# ─── SENTIMENT ENGINE ──────────────────────────────────────────────────────────
@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()

def analyze_sentiment(news_items):
    analyzer = get_analyzer()
    results = []

    # Finance-specific keyword boosters
    bullish_words = {
        'surge', 'soar', 'rally', 'jump', 'rise', 'gain', 'profit', 'beat',
        'record', 'upgrade', 'buy', 'bullish', 'growth', 'strong', 'positive',
        'expand', 'outperform', 'upside', 'boost', 'high', 'breakthrough'
    }
    bearish_words = {
        'fall', 'drop', 'decline', 'plunge', 'loss', 'miss', 'downgrade',
        'sell', 'bearish', 'weak', 'concern', 'risk', 'pressure', 'headwind',
        'disappoints', 'cut', 'warning', 'negative', 'slump', 'crash', 'fear'
    }

    for item in news_items:
        text = item['title'] + ' ' + item.get('description', '')
        text_lower = text.lower()

        # Base VADER score
        base_score = analyzer.polarity_scores(text)['compound']

        # Finance keyword adjustment
        bull_hits = sum(1 for w in bullish_words if w in text_lower)
        bear_hits = sum(1 for w in bearish_words if w in text_lower)
        adjustment = (bull_hits - bear_hits) * 0.05

        # Final score (clamped to -1, +1)
        final_score = max(-1.0, min(1.0, base_score + adjustment))

        # Override with demo scores if available
        if '_demo_score' in item:
            final_score = item['_demo_score']

        # Classify
        if final_score >= 0.15:
            label = 'Bullish'
            css = 'bull'
        elif final_score <= -0.15:
            label = 'Bearish'
            css = 'bear'
        else:
            label = 'Neutral'
            css = 'neut'

        results.append({
            **{k: v for k, v in item.items() if not k.startswith('_')},
            'score': round(final_score, 4),
            'label': label,
            'css': css
        })

    return pd.DataFrame(results)


def get_overall_signal(df):
    avg = df['score'].mean()
    bullish_pct = (df['label'] == 'Bullish').mean() * 100
    bearish_pct = (df['label'] == 'Bearish').mean() * 100
    neutral_pct = (df['label'] == 'Neutral').mean() * 100

    if avg >= 0.25:
        signal = "BULLISH"
        css = "signal-bullish"
        color = "#00D278"
        emoji = "🟢"
        advice = "Positive news momentum — market sentiment favors upside"
    elif avg <= -0.25:
        signal = "BEARISH"
        css = "signal-bearish"
        color = "#FF4B4B"
        emoji = "🔴"
        advice = "Negative news momentum — caution advised, watch for reversal"
    elif avg >= 0.05:
        signal = "MILDLY BULLISH"
        css = "signal-bullish"
        color = "#00D278"
        emoji = "🟡"
        advice = "Slightly positive tone — monitor for confirmation signal"
    elif avg <= -0.05:
        signal = "MILDLY BEARISH"
        css = "signal-bearish"
        color = "#FF4B4B"
        emoji = "🟡"
        advice = "Slightly negative tone — watch key support levels"
    else:
        signal = "NEUTRAL"
        css = "signal-neutral"
        color = "#C8B432"
        emoji = "⚪"
        advice = "Mixed sentiment — no strong directional bias in news"

    return {
        'signal': signal, 'css': css, 'color': color,
        'emoji': emoji, 'advice': advice,
        'avg_score': avg,
        'bullish_pct': bullish_pct,
        'bearish_pct': bearish_pct,
        'neutral_pct': neutral_pct
    }


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 StockPulse")
    st.markdown("---")

    # Stock selector
    st.markdown("**Select Stock**")
    stock_choice = st.selectbox("", list(STOCK_PRESETS.keys()), label_visibility="collapsed")

    if stock_choice == "Custom":
        custom_query = st.text_input("Enter stock name or ticker", placeholder="e.g. Infosys NSE stock")
        search_query = custom_query
    else:
        search_query = STOCK_PRESETS[stock_choice]

    st.markdown("---")

    # API Key
    st.markdown("**NewsAPI Key** *(optional)*")
    api_key = st.text_input(
        "",
        type="password",
        placeholder="Paste your key here",
        label_visibility="collapsed"
    )
    st.markdown("""
    <div class="api-box">
        🔑 Get a free key at <b>newsapi.org</b><br>
        Takes 30 seconds, no credit card.<br><br>
        Without key: runs on realistic demo data so you can still see all features.
    </div>
    """, unsafe_allow_html=True)

    days = st.slider("Days of news history", 3, 15, 15)
    max_articles = st.slider("Max articles to analyze", 5, 30, 15)

    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze Now", use_container_width=True)

    st.markdown("---")
    st.markdown("**Disclaimer**")
    st.caption("This tool is for educational purposes only. Not financial advice. Always do your own research before investing.")


# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">📈 Stock<span style="color:#00D278">Pulse</span></p>
    <p class="hero-sub">News Sentiment Analyzer · Bullish / Bearish Signal from Last 15 Days of Headlines</p>
</div>
""", unsafe_allow_html=True)

# ─── MAIN LOGIC ────────────────────────────────────────────────────────────────
if not analyze_btn and 'df_results' not in st.session_state:
    # Landing state
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#4A5568">
        <div style="font-size:3rem;margin-bottom:1rem">🔍</div>
        <div style="font-size:1.1rem;font-weight:500;color:#6B7A8D;margin-bottom:0.5rem">Select a stock and click Analyze Now</div>
        <div style="font-size:0.85rem">Fetches latest news headlines · Runs VADER sentiment analysis · Generates Bullish/Bearish signal</div>
    </div>
    """, unsafe_allow_html=True)

else:
    if analyze_btn or 'df_results' in st.session_state:

        # Fetch or use cached results
        if analyze_btn:
            with st.spinner(f"Fetching news for **{search_query}**..."):
                if api_key and len(api_key) > 10:
                    news, error = fetch_news_newsapi(search_query, api_key, days)
                    if error:
                        st.warning(f"⚠️ API error: {error}. Using demo data instead.")
                        news = generate_demo_news(search_query, days)
                    elif len(news) == 0:
                        st.warning("No articles found for this query. Using demo data.")
                        news = generate_demo_news(search_query, days)
                    else:
                        st.success(f"✅ Fetched {len(news)} real articles from NewsAPI")
                else:
                    news = generate_demo_news(search_query, days)

                news = news[:max_articles]
                df = analyze_sentiment(news)
                st.session_state['df_results'] = df
                st.session_state['stock_name'] = stock_choice
                st.session_state['search_query'] = search_query

        df = st.session_state['df_results']
        signal = get_overall_signal(df)

        # ── SIGNAL ROW ──────────────────────────────────────────────────────────
        col_sig, col_stats = st.columns([1, 2])

        with col_sig:
            st.markdown(f"""
            <div class="signal-box {signal['css']}">
                <div class="signal-label" style="color:{signal['color']}">Overall Signal</div>
                <div class="signal-value" style="color:{signal['color']}">{signal['emoji']} {signal['signal']}</div>
                <div class="signal-sub">{signal['advice']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_stats:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-val" style="color:#C8D6E5">{len(df)}</div>
                    <div class="stat-lbl">Articles</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-val" style="color:#00D278">{signal['bullish_pct']:.0f}%</div>
                    <div class="stat-lbl">Bullish</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-val" style="color:#FF4B4B">{signal['bearish_pct']:.0f}%</div>
                    <div class="stat-lbl">Bearish</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                score_color = "#00D278" if signal['avg_score'] > 0 else "#FF4B4B" if signal['avg_score'] < 0 else "#C8B432"
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-val" style="color:{score_color}">{signal['avg_score']:+.3f}</div>
                    <div class="stat-lbl">Avg Score</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── CHARTS ROW ──────────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["📰 News Feed", "📊 Charts", "📋 Data Table"])

        with tab1:
            st.markdown('<div class="section-hdr">Latest Headlines with Sentiment</div>',
                        unsafe_allow_html=True)

            # Filter
            filter_col1, filter_col2 = st.columns([2, 1])
            with filter_col1:
                filter_sent = st.radio("Filter", ["All", "Bullish", "Bearish", "Neutral"],
                                       horizontal=True)
            with filter_col2:
                sort_by = st.selectbox("Sort by", ["Date (newest)", "Score (highest)", "Score (lowest)"])

            # Apply filter and sort
            df_view = df.copy()
            if filter_sent != "All":
                df_view = df_view[df_view['label'] == filter_sent]

            if sort_by == "Score (highest)":
                df_view = df_view.sort_values('score', ascending=False)
            elif sort_by == "Score (lowest)":
                df_view = df_view.sort_values('score', ascending=True)
            else:
                df_view = df_view.sort_values('published', ascending=False)

            # Render news cards
            for _, row in df_view.iterrows():
                pill_cls = f"pill-{row['css']}"
                card_cls = f"news-{row['css']}"
                score_str = f"{row['score']:+.3f}"
                label_emoji = "🟢" if row['label'] == 'Bullish' else "🔴" if row['label'] == 'Bearish' else "⚪"

                st.markdown(f"""
                <div class="news-card {card_cls}">
                    <div class="news-title">{row['title']}</div>
                    <div class="news-meta">
                        <span>📅 {row['published']}</span>
                        <span>📰 {row['source']}</span>
                        <span><span class="score-pill {pill_cls}">{label_emoji} {row['label']} {score_str}</span></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            c1, c2 = st.columns(2)

            with c1:
                # Sentiment distribution donut
                fig, ax = plt.subplots(figsize=(6, 4))
                counts = df['label'].value_counts()
                colors_map = {'Bullish': '#00D278', 'Bearish': '#FF4B4B', 'Neutral': '#C8B432'}
                pie_colors = [colors_map.get(l, '#888') for l in counts.index]
                wedges, texts, autotexts = ax.pie(
                    counts.values, labels=counts.index, colors=pie_colors,
                    autopct='%1.0f%%', startangle=90,
                    wedgeprops=dict(width=0.55, edgecolor='#0D1117', linewidth=2),
                    pctdistance=0.75
                )
                for t in texts: t.set_color('#4A5568'); t.set_fontsize(10)
                for at in autotexts: at.set_color('#E8F4F0'); at.set_fontweight('700')
                ax.set_title('Sentiment Breakdown', fontsize=12, pad=15, color='#C8D6E5')
                fig.patch.set_facecolor('#0D1117')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with c2:
                # Score distribution histogram
                fig, ax = plt.subplots(figsize=(6, 4))
                bull_scores = df[df['label'] == 'Bullish']['score']
                bear_scores = df[df['label'] == 'Bearish']['score']
                neut_scores = df[df['label'] == 'Neutral']['score']

                if len(bull_scores): ax.hist(bull_scores, bins=8, color='#00D278', alpha=0.7, label='Bullish', edgecolor='none')
                if len(bear_scores): ax.hist(bear_scores, bins=8, color='#FF4B4B', alpha=0.7, label='Bearish', edgecolor='none')
                if len(neut_scores): ax.hist(neut_scores, bins=5, color='#C8B432', alpha=0.7, label='Neutral', edgecolor='none')

                ax.axvline(x=0, color='#4A5568', linestyle='--', alpha=0.6)
                ax.axvline(x=signal['avg_score'], color='#FFFFFF', linestyle='-', alpha=0.8, linewidth=1.5, label=f"Avg: {signal['avg_score']:+.3f}")
                ax.set_title('Score Distribution', fontsize=12, pad=15, color='#C8D6E5')
                ax.set_xlabel('VADER Sentiment Score', fontsize=10)
                ax.set_ylabel('Articles', fontsize=10)
                ax.legend(fontsize=9, framealpha=0, labelcolor='#C8D6E5')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.patch.set_facecolor('#0D1117')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Sentiment timeline
            st.markdown('<div class="section-hdr" style="margin-top:1rem">Sentiment Timeline</div>',
                        unsafe_allow_html=True)
            try:
                df['pub_dt'] = pd.to_datetime(df['published'])
                daily = df.groupby('pub_dt')['score'].mean().reset_index()
                daily = daily.sort_values('pub_dt')

                fig, ax = plt.subplots(figsize=(12, 3.5))
                colors_line = ['#00D278' if s >= 0.15 else '#FF4B4B' if s <= -0.15 else '#C8B432'
                               for s in daily['score']]

                ax.fill_between(daily['pub_dt'], daily['score'], 0,
                                where=(daily['score'] >= 0),
                                alpha=0.15, color='#00D278', interpolate=True)
                ax.fill_between(daily['pub_dt'], daily['score'], 0,
                                where=(daily['score'] < 0),
                                alpha=0.15, color='#FF4B4B', interpolate=True)
                ax.plot(daily['pub_dt'], daily['score'], color='#C8D6E5',
                        linewidth=1.5, zorder=3)
                ax.scatter(daily['pub_dt'], daily['score'],
                           c=colors_line, s=50, zorder=4, edgecolors='#0D1117', linewidth=1)
                ax.axhline(y=0, color='#4A5568', linestyle='--', alpha=0.5)
                ax.axhline(y=0.25, color='#00D278', linestyle=':', alpha=0.3, label='Bullish threshold')
                ax.axhline(y=-0.25, color='#FF4B4B', linestyle=':', alpha=0.3, label='Bearish threshold')
                ax.set_ylabel('Avg Daily Sentiment', fontsize=10)
                ax.set_ylim(-1.1, 1.1)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                plt.xticks(rotation=30, ha='right')
                ax.legend(fontsize=9, framealpha=0, labelcolor='#4A5568')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', alpha=0.3)
                fig.patch.set_facecolor('#0D1117')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.caption(f"Timeline chart unavailable: {e}")

        with tab3:
            st.markdown('<div class="section-hdr">Full Data Export</div>', unsafe_allow_html=True)
            display_df = df[['published', 'title', 'source', 'label', 'score']].copy()
            display_df.columns = ['Date', 'Headline', 'Source', 'Signal', 'Score']
            display_df['Score'] = display_df['Score'].round(4)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            csv = display_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download as CSV",
                csv,
                f"stockpulse_{search_query.replace(' ','_')[:20]}.csv",
                "text/csv"
            )

        # ── DISCLAIMER ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;color:#2A3A50;font-size:11px;padding:0.5rem">'
            '⚠️ StockPulse is a sentiment analysis tool for educational and research purposes only. '
            'Not financial advice. Past news sentiment does not guarantee future price movement. '
            'Always consult a SEBI-registered financial advisor before investing.'
            '</div>',
            unsafe_allow_html=True
        )
