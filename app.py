import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Monitor Avanzato", layout="wide")
st.title("📈 La mia piattaforma azioni — versione avanzata")

# ----------------------------
# Mappa NOME → TICKER (inclusi i tuoi + ETF/ETC esempi)
# ----------------------------
COMPANIES = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Alphabet (Google)": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Enel": "ENEL.MI",
    "ENI": "ENI.MI",
    "Intesa Sanpaolo": "ISP.MI",
    "STMicroelectronics": "STM.MI",
    "Red Cat": "RCAT",
    "Kratos Defense": "KTOS",
    "Rigetti": "RGTI",
    "AIRO": "AIRO",
    "Mexedia": "ALMEX",
    "DroneShield": "DRO",   # se non dà dati prova 'DRO.AX'
    # ETF/ETC popolari (puoi aggiungerne altri)
    "SPDR Gold (US)": "GLD",
    "iShares Gold (US)": "IAU",
    "WisdomTree Physical Gold (UK)": "PHAU.L",
    "XETRA-Gold (DE)": "4GLD.DE",
    "iShares Silver (US)": "SLV",
    "WisdomTree Physical Silver (UK)": "PHAG.L",
    "MSCI World (Xetra)": "EUNL.DE",
    "MSCI World (LSE)": "SWDA.L",
    "MSCI World (AMS)": "IWDA.AS",
    "MSCI World (US)": "URTH",
}

# ----------------------------
# Helpers & Cache
# ----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _normalize_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, level=-1, axis=1)
        except Exception:
            df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=300, show_spinner=True)
def load_prices(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, group_by="column")
    return _normalize_df(df, ticker)

@st.cache_data(ttl=300, show_spinner=False)
def load_daily(ticker: str, days: int = 10) -> pd.DataFrame:
    """Scarica ultimi N giorni (daily) e normalizza colonne."""
    df = yf.download(ticker, period=f"{days}d", interval="1d", auto_adjust=True, group_by="column")
    return _normalize_df(df, ticker)

@st.cache_data(ttl=300, show_spinner=False)
def load_batch_last_close(tickers: list[str]) -> dict:
    """Ultimo close per ciascun ticker. Prova batch, poi fallback per-simbolo."""
    out = {t: float("nan") for t in tickers}
    if not tickers:
        return out
    try:
        df = yf.download(tickers, period="5d", interval="1d", auto_adjust=True, group_by="ticker")
        if isinstance(df.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    s = df["Close"][t].dropna()
                    if len(s):
                        out[t] = float(s.iloc[-1])
                except Exception:
                    pass
        else:
            s = df["Close"].dropna()
            if len(s) and len(tickers) == 1:
                out[tickers[0]] = float(s.iloc[-1])
    except Exception:
        pass
    # Fallback singolo
    for t in tickers:
        if not pd.notna(out.get(t)):
            try:
                h = yf.Ticker(t).history(period="5d", interval="1d", auto_adjust=True)
                if not h.empty:
                    out[t] = float(h["Close"].dropna().iloc[-1])
            except Exception:
                pass
    return out

@st.cache_data(ttl=600, show_spinner=False)
def load_info(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.get_info()
    except Exception:
        try:
            info = t.info
        except Exception:
            info = {}
    return info

@st.cache_data(ttl=600, show_spinner=False)
def load_actions(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = yf.Ticker(ticker)
    div = pd.DataFrame()
    spl = pd.DataFrame()
    try:
        div = t.dividends.to_frame("Dividend")
    except Exception:
        pass
    try:
        spl = t.splits.to_frame("Split")
    except Exception:
        pass
    return div, spl

def fmt_num(v, dec=2):
    try:
        n = float(v)
        if abs(n) >= 1_000_000_000_000:
            return f"{n/1_000_000_000_000:,.{dec}f}T"
        if abs(n) >= 1_000_000_000:
            return f"{n/1_000_000_000:,.{dec}f}B"
        if abs(n) >= 1_000_000:
            return f"{n/1_000_000:,.{dec}f}M"
        return f"{n:,.{dec}f}"
    except Exception:
        return "—"

# ----------------------------
# Sidebar (azienda + periodo + opzioni)
# ----------------------------
st.sidebar.subheader("Impostazioni")
company_name = st.sidebar.selectbox("Scegli azienda/ETF", list(COMPANIES.keys()), index=0)
ticker = COMPANIES[company_name]

period_labels = ["1 giorno", "1 settimana", "3 mesi", "6 mesi", "1 anno", "All"]
period_map = {
    "1 giorno": "1d",
    "1 settimana": "7d",
    "3 mesi": "3mo",
    "6 mesi": "6mo",
    "1 anno": "1y",
    "All": "max",
}
period_label = st.sidebar.selectbox("Periodo di analisi", period_labels, index=3)
period = period_map[period_label]
interval = "15m" if period == "1d" else ("30m" if period == "7d" else "1d")

# Toggle grafici/indicatori
show_sma20      = st.sidebar.checkbox("Mostra SMA20", value=True)
show_sma50      = st.sidebar.checkbox("Mostra SMA50", value=True)
show_bbands     = st.sidebar.checkbox("Mostra Bande di Bollinger", value=False)
show_close_line = st.sidebar.checkbox("Mostra linea Close", value=True)
show_candles    = st.sidebar.checkbox("Mostra Candlestick", value=True)
show_boxplot    = st.sidebar.checkbox("Mostra Boxplot variazioni %", value=False)

st.subheader(f"{company_name}  ·  {ticker}")

# ----------------------------
# Dati prezzo + indicatori
# ----------------------------
df = load_prices(ticker, period, interval)
if df.empty or "Close" not in df.columns:
    st.warning(f"Nessun dato valido per {company_name} ({ticker}). Se è 'DRO', prova 'DRO.AX'.")
    st.stop()

df["SMA20"] = ta.sma(df["Close"], length=20)
df["SMA50"] = ta.sma(df["Close"], length=50)
if show_bbands:
    bb = ta.bbands(df["Close"], length=20)
    if bb is not None and not bb.empty:
        df["BBL"] = bb.iloc[:, 0]
        df["BBM"] = bb.iloc[:, 1]
        df["BBU"] = bb.iloc[:, 2]

# ----------------------------
# Variazioni (giornaliera + periodo)
# ----------------------------
daily = load_daily(ticker, days=10)
if len(daily) >= 2 and pd.notna(daily["Close"].iloc[-1]) and pd.notna(daily["Close"].iloc[-2]):
    last_close_daily = float(daily["Close"].iloc[-1])
    prev_close_daily = float(daily["Close"].iloc[-2])
    day_abs = last_close_daily - prev_close_daily
    day_pct = (day_abs / prev_close_daily) * 100 if prev_close_daily else 0.0
else:
    last_close_daily, day_abs, day_pct = float("nan"), float("nan"), float("nan")

show_period_var = (period != "1d")
if show_period_var:
    close_series = df["Close"].dropna()
    if len(close_series) >= 2:
        last_close_period = float(close_series.iloc[-1])
        start_close_period = float(close_series.iloc[0])
        per_abs = last_close_period - start_close_period
        per_pct = (per_abs / start_close_period) * 100 if start_close_period else 0.0
    else:
        last_close_period, per_abs, per_pct = float("nan"), float("nan"), float("nan")

# ----------------------------
# Metric in alto
# ----------------------------
if show_period_var:
    col1, col2, col3, col4 = st.columns([1.2, 1.4, 1, 1])
else:
    col1, col3, col4 = st.columns([1.6, 1, 1])

with col1:
    st.metric("Variazione giornaliera (ultimo close)", f"{last_close_daily:,.2f}",
              delta=f"{day_abs:+.2f} ({day_pct:+.2f}%)")
if show_period_var:
    with col2:
        st.metric(f"Da inizio periodo ({period_label})", f"{last_close_period:,.2f}",
                  delta=f"{per_abs:+.2f} ({per_pct:+.2f}%)")
with (col3 if show_period_var else col3):
    st.metric("SMA20", f"{df['SMA20'].iloc[-1]:,.2f}" if pd.notna(df['SMA20'].iloc[-1]) else "—")
with (col4 if show_period_var else col4):
    st.metric("SMA50", f"{df['SMA50'].iloc[-1]:,.2f}" if pd.notna(df['SMA50'].iloc[-1]) else "—")

# ----------------------------
# Tabs principali (senza Benchmark)
# ----------------------------
tab_price, tab_portfolio, tab_fund, tab_events, tab_raw = st.tabs(
    ["Prezzo", "Portafoglio", "Fondamentali", "Eventi", "Dati grezzi"]
)

# Prezzo: Close line + Candlestick opzionale + indicatori + boxplot + spiegazione SMA
with tab_price:
    if not df.empty:
        fig = go.Figure()

        # Candlestick opzionale
        if show_candles:
            fig.add_candlestick(
                x=df.index,
                open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name="Prezzo"
            )

        # Linea del Close (default ON)
        if show_close_line:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Close"], mode="lines",
                name="Close", line=dict(width=2)
            ))

        # Indicatori opzionali
        if show_sma20 and "SMA20" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], mode="lines", name="SMA20"))
        if show_sma50 and "SMA50" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], mode="lines", name="SMA50"))
        if show_bbands and {"BBL","BBM","BBU"} <= set(df.columns):
            fig.add_trace(go.Scatter(x=df.index, y=df["BBU"], mode="lines", name="BBand Upper", line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df["BBM"], mode="lines", name="BBand Middle", line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df["BBL"], mode="lines", name="BBand Lower", line=dict(width=1)))

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=520,
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Boxplot delle variazioni % nel periodo selezionato (opzionale)
        if show_boxplot:
            returns_pct = df["Close"].pct_change().dropna() * 100.0
            if not returns_pct.empty:
                box = go.Figure()
                box.add_trace(go.Box(y=returns_pct, name=f"Rendimenti % ({period_label})", boxmean=True))
                box.update_layout(height=320)
                st.plotly_chart(box, use_container_width=True)

        # Spiegazione SMA (solo qui)
        with st.expander("Cos’è la SMA20 / SMA50?"):
            st.markdown("""
**SMA** (*Simple Moving Average*) = **media mobile semplice** del **prezzo di chiusura** degli ultimi *N* periodi.

- **SMA20**: media degli ultimi **20** periodi (20 giorni su daily, 20 barre su intraday).
- **SMA50**: media degli ultimi **50** periodi.
- A cosa serve: **smussa** il rumore, aiuta a leggere il **trend**.  
  Una SMA corta (20) reagisce più in fretta; una lunga (50) è più lenta/stabile.
- Incroci frequenti: **SMA20 sopra SMA50** (*golden cross*, trend in rafforzamento);  
  **SMA20 sotto SMA50** (*death cross*, indebolimento).
""")

# Portafoglio
with tab_portfolio:
    st.caption("Inserisci/rivedi le tue posizioni. I prezzi sono ultimo close via Yahoo Finance.")
    if "portfolio_df" not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame(
            [{"Ticker": ticker, "Qtà": 10, "PMC": 100.00}]
        )

    pf = st.data_editor(
        st.session_state.portfolio_df,
        num_rows="dynamic",
        use_container_width=True,
        key="pf_editor"
    )

    tickers_in_pf = [str(x).upper().strip() for x in pf["Ticker"].dropna().unique().tolist()]
    last_prices = load_batch_last_close(tickers_in_pf)

    def row_pl(row):
        t = str(row.get("Ticker","")).upper().strip()
        qty = float(row.get("Qtà", 0) or 0)
        pmc = float(row.get("PMC", 0) or 0)
        px = float(last_prices.get(t, float("nan")))
        value = qty * px if pd.notna(px) else float("nan")
        pnl = (px - pmc) * qty if (pd.notna(px) and pmc) else float("nan")
        pnl_pct = ((px - pmc) / pmc * 100) if (pmc and pd.notna(px)) else float("nan")
        return pd.Series({"Ultimo Prezzo": px, "Valore": value, "P&L": pnl, "P&L %" : pnl_pct})

    if not pf.empty:
        calc = pf.apply(row_pl, axis=1)
        result = pd.concat([pf, calc], axis=1)

        # Vista “pulita” con trattini al posto dei NaN
        result_display = result.copy()
        for col in ["Ultimo Prezzo", "Valore", "P&L", "P&L %"]:
            if col in result_display:
                result_display[col] = result_display[col].apply(lambda x: "—" if pd.isna(x) else round(float(x), 2))

        st.dataframe(result_display, use_container_width=True)

        # Metriche totali calcolate sui numeri reali
        tot_value = result["Valore"].sum(skipna=True)
        tot_pnl = result["P&L"].sum(skipna=True)
        st.metric("Valore Portafoglio", fmt_num(tot_value, 2))
        st.metric("P&L Totale", fmt_num(tot_pnl, 2))

        # salva in sessione
        st.session_state.portfolio_df = pf

# Fondamentali
with tab_fund:
    info = load_info(ticker)
    def safe_get(k, default="—"):
        v = info.get(k, None)
        return v if v is not None else default

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("P/E (Trailing)", safe_get("trailingPE", "—"))
        st.metric("EPS (Trailing)", safe_get("trailingEps", "—"))
        st.metric("Beta", safe_get("beta", "—"))
    with c2:
        st.metric("P/E (Forward)", safe_get("forwardPE", "—"))
        dy = safe_get("dividendYield", None)
        st.metric("Dividend Yield", f"{float(dy)*100:.2f}%" if isinstance(dy, (int,float)) else "—")
        st.metric("Market Cap", fmt_num(safe_get("marketCap", None)))
    with c3:
        st.write(f"**Settore**: {safe_get('sector')}")
        st.write(f"**Industria**: {safe_get('industry')}")
        st.caption("Dati da Yahoo Finance via yfinance; per alcuni titoli/borse alcuni campi possono mancare.")

# Eventi
with tab_events:
    div, spl = load_actions(ticker)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Dividendi (recenti)")
        if div is not None and not div.empty:
            st.dataframe(div.tail(10))
        else:
            st.write("—")
    with c2:
        st.subheader("Split (recenti)")
        if spl is not None and not spl.empty:
            st.dataframe(spl.tail(10))
        else:
            st.write("—")

# Dati grezzi
with tab_raw:
    st.dataframe(df.tail(30))
