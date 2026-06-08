import os
import json
import urllib.parse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from transformers import pipeline
import warnings
import streamlit as st
import feedparser


warnings.filterwarnings('ignore')

# Set up Streamlit Page Configuration
st.set_page_config(page_title="Sector Sentiment Analyzer", layout="wide", page_icon="📈")

# ─────────────────────────────────────────────────────────────────────────────
# Name / alias → canonical NSE/NYSE ticker
# ─────────────────────────────────────────────────────────────────────────────
NAME_TO_TICKER = {
    # ── IT / Technology ──────────────────────────────────────────────────────
    "tcs": "TCS.NS", "tata consultancy": "TCS.NS", "tata consultancy services": "TCS.NS",
    "infosys": "INFY.NS", "infy": "INFY.NS", "wipro": "WIPRO.NS", "hcl": "HCLTECH.NS",
    "hcltech": "HCLTECH.NS", "hcl technologies": "HCLTECH.NS", "tech mahindra": "TECHM.NS",
    "techm": "TECHM.NS", "ltm": "LTM.NS", "ltimindtree": "LTM.NS", "persistent": "PERSISTENT.NS",
    "coforge": "COFORGE.NS", "mphasis": "MPHASIS.NS", "oracle financial": "OFSS.NS",
    "ofss": "OFSS.NS", "kpit": "KPITTECH.NS", "kpit technologies": "KPITTECH.NS",
    # ── Banking / Financial ───────────────────────────────────────────────────
    "hdfc bank": "HDFCBANK.NS", "hdfcbank": "HDFCBANK.NS", "hdfc": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS", "icicibank": "ICICIBANK.NS", "icici": "ICICIBANK.NS",
    "kotak": "KOTAKBANK.NS", "kotakbank": "KOTAKBANK.NS", "kotak mahindra bank": "KOTAKBANK.NS",
    "sbi": "SBIN.NS", "state bank": "SBIN.NS", "axis bank": "AXISBANK.NS", "axisbank": "AXISBANK.NS",
    "bajaj finance": "BAJFINANCE.NS", "bajajfinance": "BAJFINANCE.NS", "bajaj finserv": "BAJAJFINSV.NS",
    "indusind bank": "INDUSINDBK.NS", "indusindbk": "INDUSINDBK.NS", "yes bank": "YESBANK.NS",
    "yesbank": "YESBANK.NS", "pnb": "PNB.NS", "punjab national bank": "PNB.NS", "bank of baroda": "BANKBARODA.NS", "bob": "BANKBARODA.NS",
    # ── FMCG / Consumer Defensive ────────────────────────────────────────────
    "hul": "HINDUNILVR.NS", "hindustan unilever": "HINDUNILVR.NS", "itc": "ITC.NS", "nestle": "NESTLEIND.NS",
    "nestle india": "NESTLEIND.NS", "dabur": "DABUR.NS", "marico": "MARICO.NS", "britannia": "BRITANNIA.NS",
    "godrej consumer": "GODREJCP.NS", "godrejcp": "GODREJCP.NS", "emami": "EMAMILTD.NS", "colgate": "COLPAL.NS", "colgate palmolive": "COLPAL.NS",
    # ── Energy ───────────────────────────────────────────────────────────────
    "reliance": "RELIANCE.NS", "ril": "RELIANCE.NS", "ongc": "ONGC.NS", "bpcl": "BPCL.NS", "ioc": "IOC.NS",
    "indian oil": "IOC.NS", "gail": "GAIL.NS", "oil india": "OIL.NS", "adani total gas": "ATGL.NS",
    "indraprastha gas": "IGL.NS", "igl": "IGL.NS", "petronet lng": "PETRONET.NS",
    # ── Healthcare / Pharma ──────────────────────────────────────────────────
    "sun pharma": "SUNPHARMA.NS", "sunpharma": "SUNPHARMA.NS", "dr reddy": "DRREDDY.NS", "drreddy": "DRREDDY.NS",
    "cipla": "CIPLA.NS", "divis": "DIVISLAB.NS", "divislab": "DIVISLAB.NS", "apollo hospitals": "APOLLOHOSP.NS",
    "lupin": "LUPIN.NS", "biocon": "BIOCON.NS", "zydus": "ZYDUSLIFE.NS", "zydus lifesciences": "ZYDUSLIFE.NS",
    "torrent pharma": "TORNTPHARM.NS", "ipca": "IPCALAB.NS", "alkem": "ALKEM.NS",
    # ── Auto / Consumer Cyclical ──────────────────────────────────────────────
    "maruti": "MARUTI.NS", "maruti suzuki": "MARUTI.NS", "mahindra": "M&M.NS", "m&m": "M&M.NS",
    "bajaj auto": "BAJAJ-AUTO.NS", "eicher": "EICHERMOT.NS", "royal enfield": "EICHERMOT.NS",
    "hero motocorp": "HEROMOTOCO.NS", "titan": "TITAN.NS", "tata motors": "TATAMOTORS.NS",
    "tatamotors": "TATAMOTORS.NS", "tvs motor": "TVSMOTOR.NS", "tvsmotor": "TVSMOTOR.NS", "mrf": "MRF.NS",
    # ── Industrials ──────────────────────────────────────────────────────────
    "l&t": "LT.NS", "larsen": "LT.NS", "larsen and toubro": "LT.NS", "siemens": "SIEMENS.NS",
    "bhel": "BHEL.NS", "hal": "HAL.NS", "hindustan aeronautics": "HAL.NS", "bel": "BEL.NS",
    "bharat electronics": "BEL.NS", "cummins": "CUMMINSIND.NS", "thermax": "THERMAX.NS", "abb india": "ABB.NS", "abb": "ABB.NS",
    # ── Metals / Materials ───────────────────────────────────────────────────
    "tata steel": "TATASTEEL.NS", "tatasteel": "TATASTEEL.NS", "hindalco": "HINDALCO.NS", "jsw steel": "JSWSTEEL.NS",
    "jswsteel": "JSWSTEEL.NS", "vedanta": "VEDL.NS", "nmdc": "NMDC.NS", "sail": "SAIL.NS",
    "steel authority": "SAIL.NS", "jindal steel": "JINDALSTEL.NS", "jindalstel": "JINDALSTEL.NS", "national aluminium": "NATIONALUM.NS", "nalco": "NATIONALUM.NS",
    # ── Real Estate ──────────────────────────────────────────────────────────
    "dlf": "DLF.NS", "godrej properties": "GODREJPROP.NS", "godrejprop": "GODREJPROP.NS", "oberoi realty": "OBEROIRLTY.NS",
    "prestige": "PRESTIGE.NS", "brigade": "BRIGADE.NS", "lodha": "MACROTECH.NS", "macrotech": "MACROTECH.NS", "sobha": "SOBHA.NS",
    # ── Utilities ────────────────────────────────────────────────────────────
    "ntpc": "NTPC.NS", "powergrid": "POWERGRID.NS", "power grid": "POWERGRID.NS", "tata power": "TATAPOWER.NS",
    "adani power": "ADANIPOWER.NS", "torrent power": "TORNTPOWER.NS", "jsw energy": "JSWENERGY.NS",
    # ── Telecom ──────────────────────────────────────────────────────────────
    "airtel": "BHARTIARTL.NS", "bharti airtel": "BHARTIARTL.NS", "vodafone idea": "IDEA.NS", "vi": "IDEA.NS", "idea": "IDEA.NS", "indus towers": "INDUSTOWER.NS",
    # ── Cement ───────────────────────────────────────────────────────────────
    "ultratech": "ULTRACEMCO.NS", "ultratech cement": "ULTRACEMCO.NS", "shree cement": "SHREECEM.NS", "acc": "ACC.NS",
    "ambuja": "AMBUJACEM.NS", "ambuja cements": "AMBUJACEM.NS", "dalmia bharat": "DALBHARAT.NS", "jk cement": "JKCEMENT.NS",
}

SECTOR_PEERS = {
    "Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "OFSS.NS", "KPITTECH.NS"],
    "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "YESBANK.NS", "PNB.NS", "BANKBARODA.NS"],
    "Consumer Defensive": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "DABUR.NS", "MARICO.NS", "BRITANNIA.NS", "GODREJCP.NS", "EMAMILTD.NS", "COLPAL.NS"],
    "Consumer Cyclical": ["TITAN.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TATAMOTORS.NS", "TVSMOTOR.NS", "MRF.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "OIL.NS", "ATGL.NS", "IGL.NS", "PETRONET.NS"],
    "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "LUPIN.NS", "BIOCON.NS", "ZYDUSLIFE.NS", "TORNTPHARM.NS"],
    "Industrials": ["LT.NS", "SIEMENS.NS", "ABB.NS", "BHEL.NS", "HAL.NS", "BEL.NS", "CUMMINSIND.NS", "THERMAX.NS"],
    "Basic Materials": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "NMDC.NS", "SAIL.NS", "JINDALSTEL.NS", "NATIONALUM.NS"],
    "Real Estate": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "BRIGADE.NS", "MACROTECH.NS", "SOBHA.NS"],
    "Utilities": ["NTPC.NS", "POWERGRID.NS", "ADANIPOWER.NS", "TATAPOWER.NS", "TORNTPOWER.NS", "JSWENERGY.NS"],
    "Communication Services": ["BHARTIARTL.NS", "IDEA.NS", "INDUSTOWER.NS"],
    "Basic Materials_Cement": ["ULTRACEMCO.NS", "SHREECEM.NS", "ACC.NS", "AMBUJACEM.NS", "DALBHARAT.NS", "JKCEMENT.NS"],
}

def resolve_ticker(user_input: str) -> str | None:
    normalised = user_input.strip().lower()
    if normalised in NAME_TO_TICKER:
        return NAME_TO_TICKER[normalised]
    for key, ticker in NAME_TO_TICKER.items():
        if normalised in key or key in normalised:
            return ticker
    candidate = user_input.strip().upper()
    if _ticker_valid(candidate):
        return candidate
    candidate_ns = candidate + ".NS"
    if _ticker_valid(candidate_ns):
        return candidate_ns
    return None

def _ticker_valid(ticker: str) -> bool:
    try:
        info = yf.Ticker(ticker).info
        return bool(info.get("sector") or info.get("longName"))
    except Exception:
        return False

def _get_info(ticker: str) -> dict | None:
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName")
        if not name:
            return None
        return {
            "name":       name,
            "ticker":     ticker.removesuffix(".NS"),
            "sector":     info.get("sector", ""),
            "market_cap": info.get("marketCap") or 0,
        }
    except Exception:
        return None

def get_similar_stocks(resolved_ticker: str, base_info: dict, top_n: int = 4) -> list[dict]:
    sector = base_info["sector"]
    peer_tickers = []
    
    for sec_key, tickers in SECTOR_PEERS.items():
        if sec_key.lower() in sector.lower() or sector.lower() in sec_key.lower():
            peer_tickers = [t for t in tickers if t.upper() != resolved_ticker.upper()]
            break

    if not peer_tickers:
        return [{"name": base_info["name"], "ticker": base_info["ticker"]}]

    peers = []
    for t in peer_tickers:
        info = _get_info(t)
        if info:
            peers.append(info)

    peers.sort(key=lambda x: x["market_cap"], reverse=True)
    results = [{"name": p["name"], "ticker": p["ticker"]} for p in peers[:top_n]]
    
    # Append target stock safely
    results.append({"name": base_info["name"], "ticker": base_info["ticker"]})
    return results

# Cache the AI Model so it doesn't reload on every button click
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI Interface layout
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Cross-Company Sector Sentiment Dashboard")
st.markdown("Analyze current news sentiment trends for any major Indian stock and its immediate sector peers using **FinBERT**.")

# User inputs
col1, col2 = st.columns([3, 1])
with col1:
    target_stock = st.text_input("Enter Company Name or Ticker Symbol:", value="TCS", placeholder="e.g., Infosys, Reliance, HDFC, SBIN")
with col2:
    days_lookback = st.slider("Lookback Period (Days)", min_value=5, max_value=30, value=15)

if st.button("Run Market Intelligence Analysis", type="primary"):
    with st.spinner("Resolving asset identifiers and mapping sector peers..."):
        ticker = resolve_ticker(target_stock)
        
        if not ticker:
            st.error(f"Could not automatically resolve '{target_stock}' to an eligible market ticker.")
        else:
            base = _get_info(ticker)
            if not base or not base["sector"]:
                st.error(f"Failed to fetch market sector classifications for structural comparison on '{ticker}'.")
            else:
                peers = get_similar_stocks(ticker, base)
                
                st.info(f"Target Resolved: **{base['name']}** | Sector Bucket: **{base['sector']}**")
                
                # Load FinBERT model from cache
                analyzer = load_sentiment_model()
                
                results_summary = []
                all_raw_headlines = {}

                # Iterating over peer entities
                for company in peers:
                    name = company["name"]
                    comp_ticker = company["ticker"]
                    
                    st.write(f"🔍 Analyzing sentiment footprints for **{name}**...")
                    
                    search_term = f"{name} {comp_ticker}".strip()
                    query = urllib.parse.quote(f"{search_term} stock when:{days_lookback}d")
                    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
                    
                    feed = feedparser.parse(rss_url)
                    entries = feed.entries[:20]
                    
                    bullish_wt, bearish_wt, neutral_wt = 0, 0, 0
                    company_headlines = []
                    
                    for entry in entries:
                        title = entry.title
                        sentiment = analyzer(title)[0]
                        label = sentiment['label']
                        score = sentiment['score']
                        
                        if score < 0.8:  # Drop unstable/low confidence metrics
                            continue
                            
                        if label == 'positive':
                            bullish_wt += score
                        elif label == 'negative':
                            bearish_wt += score
                        elif label == 'neutral':
                            neutral_wt += score
                            
                        company_headlines.append({
                            "Date": getattr(entry, 'published', 'N/A'),
                            "Headline": title,
                            "Sentiment": label.upper(),
                            "Confidence": round(score, 3)
                        })
                    
                    all_raw_headlines[name] = pd.DataFrame(company_headlines)
                    
                    total = bullish_wt + bearish_wt + neutral_wt
                    bull_pct = (bullish_wt / total * 100) if total > 0 else 0
                    bear_pct = (bearish_wt / total * 100) if total > 0 else 0
                    float_neut_pct = (neutral_wt / total * 100) if total > 0 else 0
                    
                    results_summary.append({
                        "Company": name,
                        "Bullish (%)": round(bull_pct, 1),
                        "Neutral (%)": round(float_neut_pct, 1),
                        "Bearish (%)": round(bear_pct, 1)
                    })
                
                df_peers = pd.DataFrame(results_summary)
                
                # Layout UI splits for output
                st.subheader("=== GENERATED SECTOR SENTIMENT MATRIX ===")
                st.dataframe(df_peers, use_container_width=True)
                
                # Matplotlib visualization rendering direct to app
                fig, ax = plt.subplots(figsize=(12, 6))
                x = np.arange(len(df_peers["Company"]))
                width = 0.25
                
                bars_bull = ax.bar(x - width, df_peers["Bullish (%)"], width, label='Bullish', color='#2ca02c', edgecolor='black', zorder=3)
                bars_neut = ax.bar(x, df_peers["Neutral (%)"], width, label='Neutral', color='#ff7f0e', edgecolor='black', zorder=3)
                bars_bear = ax.bar(x + width, df_peers["Bearish (%)"], width, label='Bearish', color='#d62728', edgecolor='black', zorder=3)
                
                for bars in (bars_bull, bars_neut, bars_bear):
                    for bar in bars:
                        h = bar.get_height()
                        if h > 0:
                            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                for i, row in df_peers.iterrows():
                    scores = {"Bullish": row["Bullish (%)"], "Neutral": row["Neutral (%)"], "Bearish": row["Bearish (%)"]}
                    dominant = max(scores, key=scores.get)
                    colour = {'Bullish': '#2ca02c', 'Neutral': '#ff7f0e', 'Bearish': '#d62728'}[dominant]
                    ax.text(x[i], -5, f"▲ {dominant}" if dominant == "Bullish" else f"● {dominant}" if dominant == "Neutral" else f"▼ {dominant}", ha='center', va='top', fontsize=8, fontweight='bold', color=colour)
                
                ax.set_ylabel('Sentiment Score (%)', fontweight='bold')
                ax.set_title(f'Peer Group Sentiment Map — Last {days_lookback} Days', fontsize=13, fontweight='bold', pad=14)
                ax.set_xticks(x)
                ax.set_xticklabels(df_peers["Company"], rotation=15, ha='right', fontsize=9)
                ax.set_ylim(0, max(df_peers[["Bullish (%)", "Neutral (%)", "Bearish (%)"]].max().max() + 15, 30))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
                ax.legend(loc='upper right', framealpha=0.9)
                ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
                
                st.pyplot(fig)
                
                # Dropdowns to see the exact scraped data per peer
                st.subheader("📋 Drill-down Structural News Analysis")
                for comp_name, df_headlines in all_raw_headlines.items():
                    with st.expander(f"View Scraped Headlines for {comp_name}"):
                        if not df_headlines.empty:
                            st.dataframe(df_headlines, use_container_width=True)
                        else:
                            st.write("No headlines with high model confidence (≥ 0.80) discovered.")
