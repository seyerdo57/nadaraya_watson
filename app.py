import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ssl
import time
import concurrent.futures

ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="LuxAlgo Final Precision", layout="wide")

# Reduce top whitespace
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

def calculate_lux_ultra_precise(symbol, df_slice, info, live_price=None):
    try:
        # 1. Veri Hazırlığı
        if df_slice.empty: return "Veri Boş"
        
        # Batch indirmede veri Series veya DataFrame (tek sütun) olabilir
        if isinstance(df_slice, pd.DataFrame):
            # Eğer DataFrame gelirse ve Close sütunu varsa
            if "Close" in df_slice.columns:
                close_series = df_slice["Close"]
            else:
                # Tek sütunlu DataFrame ise
                 close_series = df_slice.iloc[:, 0]
        else:
            # Series ise
            close_series = df_slice
            
        close_series = close_series.dropna()
        if close_series.empty: return "Temizlenen Veri Boş"
        
        # 2. Vektörel Hazırlık
        src = close_series.values.flatten().astype(float)
        
        # CANLI VERİ GÜNCELLEMESİ (Override)
        # Eğer live_price varsa, grafikteki son kapanış fiyatını bununla değiştir.
        # Bu sayede indikatör anlık fiyata göre yeniden hesaplanır.
        if live_price is not None and live_price > 0:
            src[-1] = float(live_price)
            
        n = len(src)
        
        # Pine Script Parametreleri
        h = 8.0
        mult = 3.0
        
        # 3. Pine Script 'Repaint' Algoritması (Birebir Simülasyon)
        # i: Hedef nokta, j: Ağırlık veren nokta
        indices = np.arange(n)
        nwe_line = np.zeros(n)
        
        # Performans için son 500 mumu (veya veri azsa tamamını) hesaplayalım
        # TradingView bu döngüyü max_bars_back=500 ile yapar
        lookback = 500
        start_idx = max(0, n - lookback)
        
        for i in range(start_idx, n):
            # Gauss çekirdeği ağırlıkları
            weights = np.exp(-((i - indices)**2) / (2 * h**2))
            nwe_line[i] = np.sum(src * weights) / np.sum(weights)

        # 4. SAE (Bant Genişliği) - Pine Script'teki sae := sae / n mantığı
        # Sadece hesaplanan bölgedeki hataları baz alıyoruz
        calc_range_src = src[start_idx:]
        calc_range_nwe = nwe_line[start_idx:]
        sae = np.mean(np.abs(calc_range_src - calc_range_nwe)) * mult
        
        current_price = float(src[-1])
        upper_band = float(nwe_line[-1] + sae)
        lower_band = float(nwe_line[-1] - sae)
        
        # 5. Temel Veriler (Info dict'ten al)
        pe_ratio = info.get('trailingPE', None)
        fwd_pe_ratio = info.get('forwardPE', None)
        market_cap = info.get('marketCap', None)
        industry = info.get('industry', 'N/A')
        
        return {
            "Sembol": str(symbol),
            "Piyasa Değeri": market_cap, 
            "Sektör": industry,
            "Fiyat": round(current_price, 2),
            "Üst Bant": round(upper_band, 2),
            "Alt Bant": round(lower_band, 2),
            "Üst Uzaklık %": round(((upper_band - current_price) / current_price) * 100, 2),
            "Alt Uzaklık %": round(((current_price - lower_band) / current_price) * 100, 2),
            "PE": round(pe_ratio, 2) if pe_ratio else None,
            "PE (FWD)": round(fwd_pe_ratio, 2) if fwd_pe_ratio else None
        }
    except Exception as e:
        return str(e)

# UI
st.title("Nadaraya-Watson Envelope (LuxAlgo)")

if 'results' not in st.session_state:
    st.session_state.results = []

# Market Seçimi (Butondan önce olmalı)
market_option = st.radio("Market Seçiniz:", ("NASDAQ 100", "S&P 100", "ASYA", "BİST 30"), horizontal=True)

if st.button('İncele'):
    start_time = time.time()
    status_place = st.empty()
    
    if market_option == "NASDAQ 100":
        symbols = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "AVGO", "TSLA", "COST",
            "ASML", "ADBE", "NFLX", "AMD", "PEP", "AZN", "LIN", "TMUS", "CSCO", "INTU",
            "QCOM", "TXN", "AMAT", "ISRG", "AMGN", "INTC", "HON", "VRTX", "BKNG", "BK",
            "ADP", "REGN", "MDLZ", "LRCX", "ADI", "PANW", "SNPS", "MU", "KLAC", "CDNS",
            "MELI", "PDD", "MAR", "PYPL", "CSX", "CRWD", "ORCL", "MNST", "ABNB", "LULU",
            "ADSK", "IDXX", "CTAS", "AEP", "BKR", "KDP", "MCHP", "CPRT", "DXCM", "NXPI",
            "KLA", "EXC", "PAYX", "GILD", "ROP", "LRCX", "TEAM", "WDAY", "MRVL", "ADP",
            "ODFL", "ADSK", "FAST", "GEHC", "DASH", "CDW", "MDB", "MSTR", "FANG", "ANSS",
            "BIIB", "TTD", "CSGP", "DDOG", "ON", "DLTR", "ILMN", "EXPE", "WBD", "ZM",
            "ZSC", "GFS", "ENPH", "ARM", "PDD", "EBAY", "SIRI", "VRSK", "ALGN", "WBA"
        ]
        # Remove duplicates
        symbols = list(set(symbols))
    elif market_option == "S&P 100":
        # S&P 100 (OEF) Components - Representative List
        symbols = [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "LLY", "AVGO",
            "JPM", "TSLA", "XOM", "UNH", "V", "PG", "MA", "COST", "JNJ", "HD",
            "MRK", "ABBV", "KO", "CRM", "BAC", "PEP", "WMT", "RACE", "ACN", "MCD",
            "LIN", "CSCO", "NFLX", "AMD", "AZN", "SAP", "INTU", "QCOM", "IBM", "TXN",
            "GE", "VZ", "NOW", "AMAT", "UBER", "DIS", "ISRG", "PFE", "AMGN", "INTC",
            "CAT", "GS", "CMCSA", "DHR", "NEE", "RTX", "T", "HON", "UNP", "AXP",
            "LOW", "SPGI", "PM", "PGR", "BLK", "BKNG", "SYK", "TJX", "ELV", "C",
            "VRTX", "MDT", "ADP", "MMC", "GILD", "DE", "LMT", "BSX", "CI", "ADI",
            "PANW", "BA", "MDLZ", "REGN", "MU", "KLAC", "FI", "PLD", "TMUS", "LRCX",
            "ETN", "SNPS", "CDNS", "CB", "SO", "DUK", "MO", "CL", "ZTS", "ITW"
        ]
        symbols = list(set(symbols))
    elif market_option == "ASYA":
        # Asya Marketi
        symbols = [
            "BABA", "MCHI", "BIDU", "JD", "NTDOY", "TCEHY", "XIACY", "FXI", "VO", "VOO", 
            "PDD", "SONY"
        ]
    else:
        # BİST 30 (Yahoo Finance için .IS uzantısı gerekir)
        symbols = [
            "AKBNK.IS", "ALARK.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS",
            "BRSAN.IS", "DOAS.IS", "EKGYO.IS", "ENJSA.IS", "EREGL.IS", "FROTO.IS",
            "GARAN.IS", "GUBRF.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS",
            "KOZAL.IS", "KRDMD.IS", "ODAS.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS",
            "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS",
            "TUPRS.IS", "YKBNK.IS", "VESTL.IS", "EUPWR.IS" 
        ]
    results = []
    
    status_place.markdown(f"⏳ **Veriler indiriliyor...**")
    
    # 1. Batch Fiyat İndirme (Hızlı ve Güvenli)
    # threads=True (veya default): Artık kendi thread pool'umuz başlamadan önce çalıştığı için güvenli.
    try:
        batch_data = yf.download(symbols, period="10y", interval="1wk", group_by='ticker', progress=False, auto_adjust=True)
    except Exception as e:
        st.error(f"Toplu veri indirme hatası: {e}")
        batch_data = pd.DataFrame()

    # 2. Temel Verileri (Info) Paralel Çekme
    status_place.markdown(f"⏳ **Temel analiz verileri toplanıyor...**")
    
    ticker_infos = {}
    def fetch_info(s):
        try:
            return s, yf.Ticker(s).info
        except:
            return s, {}
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {executor.submit(fetch_info, s): s for s in symbols}
        count = 0
        for future in concurrent.futures.as_completed(future_map):
            count += 1
            s, info = future.result()
            ticker_infos[s] = info
            status_place.markdown(f"⏳ **Analiz ediliyor... ({count}/{len(symbols)})**")

    for s in symbols:
        # Batch data'dan ilgili hissenin verisini al
        try:
            stock_df = pd.DataFrame()
            if not batch_data.empty:
                # MultiIndex kontrolü
                if isinstance(batch_data.columns, pd.MultiIndex):
                    if s in batch_data.columns.get_level_values(0):
                        stock_df = batch_data[s]
                else:
                    # Tek seviyeli index ise ve kolonlarda semboller varsa (group_by=ticker bazen böyle yapabilir)
                    if s in batch_data.columns:
                         stock_df = batch_data[s] # Bu genelde Series döner
                    else:
                        # Belki tek hisse indi ve kolonlar direk Open, Close vs.
                        if len(symbols) == 1:
                            stock_df = batch_data
            
            if stock_df.empty:
                continue

            info = ticker_infos.get(s, {})
            # Canlı Fiyatı al (Öncelik: Pre-Market > Post-Market > Current)
            # Piyasa kapalıyken currentPrice bazen son kapanışı gösterebilir.
            live_price = info.get('preMarketPrice')
            if live_price is None:
                live_price = info.get('postMarketPrice')
            if live_price is None:
                live_price = info.get('currentPrice')
                
            res = calculate_lux_ultra_precise(s, stock_df, info, live_price)
            
            # Hata ayıklama: Eğer sözlük değilse (None veya hata mesajı stringi)
            if res and isinstance(res, dict):
                results.append(res)
            elif isinstance(res, str):
                # İlk hatayı kullanıcıya gösterelim ki sorunu anlayalım
                if not results: 
                    st.error(f"Hesaplama Hatası ({s}): {res}")
                
        except Exception as e:
            if not results: st.error(f"Döngü Hatası ({s}): {e}")
            pass
            
    status_place.markdown(
        f"<div style='background-color: #d4edda; color: #155724; padding: 5px; border-radius: 5px; font-size: 12px; text-align: center; border: 1px solid #c3e6cb;'>"
        f"✅ Tamamlandı! Toplam Süre: {time.time() - start_time:.1f} sn"
        f"</div>", 
        unsafe_allow_html=True
    )
    st.session_state.results = results
    
    if not results:
        st.error("Sonuç tablosu oluşturulamadı.")
        # Debug için son hatayı gösterelim (eğer varsa)
        if not batch_data.empty:
            st.warning(f"Veri Yapısı: {batch_data.shape}. Örnek Kolonlar: {batch_data.columns[:5]}")

if st.session_state.results:
    df = pd.DataFrame(st.session_state.results)
    df = df.sort_values(by="Piyasa Değeri", ascending=False).reset_index(drop=True)
    
    # Styling
    def highlight_cols(x, col_name, threshold, color):
        return [f'background-color: {color}' if (val < threshold) else '' for val in x]
        
    def format_market_cap(val):
        if not val or pd.isna(val): return "N/A"
        if val >= 1_000_000_000_000:
            return f"{val / 1_000_000_000_000:.2f}T"
        elif val >= 1_000_000_000:
            return f"{val / 1_000_000_000:.2f}B"
        elif val >= 1_000_000:
            return f"{val / 1_000_000:.2f}M"
        else:
            return f"{val:,.0f}"

    styler = df.style.format({
        "Piyasa Değeri": format_market_cap,
        "Fiyat": "{:.2f}",
        "Üst Bant": "{:.2f}",
        "Alt Bant": "{:.2f}",
        "Üst Uzaklık %": "{:.2f}",
        "Alt Uzaklık %": "{:.2f}",
        "PE": "{:.2f}", 
        "PE (FWD)": "{:.2f}"
    }, na_rep="N/A")
    
    # Consolidated Styling Function to handle Selection Blending
    selected_rows = st.session_state.get("stocks_table", {}).get("selection", {}).get("rows", [])
    
    def apply_styles(row):
        styles = [''] * len(row)
        is_selected = row.name in selected_rows
        
        for i, col in enumerate(row.index):
            val = row[col]
            bg_color = None
            text_color = None
            
            # 1. Determine Conditional Base Color
            if col == 'Üst Uzaklık %':
                if pd.notna(val):
                    if val < 0: 
                        bg_color = '#8b0000'; text_color = 'white' # Dark Red
                    elif 0 <= val < 10: 
                        bg_color = '#ff9999'; text_color = 'black' # Light Red
                        
            elif col == 'Alt Uzaklık %':
                if pd.notna(val):
                    if val < 0: 
                        bg_color = '#006400'; text_color = 'white' # Dark Green
                    elif 0 <= val < 10: 
                        bg_color = '#99ff99'; text_color = 'black' # Light Green
            
            # 2. Apply Selection Overlay (Blending Logic)
            if is_selected:
                if bg_color is None:
                    bg_color = '#ffffcc' # Pale Yellow (Transparent-ish effect on white)
                    text_color = 'black'
                else:
                    # Blend with Yellow approximation
                    if bg_color == '#8b0000':   bg_color = '#ad4c00' # Brown/Rust
                    elif bg_color == '#ff9999': bg_color = '#ffb76b' # Peach/Orange
                    elif bg_color == '#006400': bg_color = '#4c9200' # Olive
                    elif bg_color == '#99ff99': bg_color = '#b7fe6b' # Lime
                    text_color = 'black' # Force black text for readability on yellow mix
                    
            if bg_color:
                s = f'background-color: {bg_color}'
                if text_color: s += f'; color: {text_color}'
                styles[i] = s
            elif is_selected:
                # Text color correction for plainly selected rows if needed
                styles[i] = 'color: black'
                
        return styles

    styler.apply(apply_styles, axis=1)

    event = st.dataframe(
        styler,
        on_select="rerun",
        selection_mode="multi-row",
        hide_index=True,
        key="stocks_table",
        use_container_width=True
    )
    
    # Optional: Display selected rows or actions if needed
    if event.selection.rows:
        st.write(f"Seçilen satır sayısı: {len(event.selection.rows)}")