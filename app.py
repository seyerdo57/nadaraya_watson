import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ssl
import time
import concurrent.futures

ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="LuxAlgo Final Precision", layout="wide")

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
market_option = st.radio("Market Seçiniz:", ("NASDAQ 100", "Asya Marketi"), horizontal=True)
if st.button('Hassas Taramayı Başlat'):
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
        # Remove duplicates just in case the user list had any
        symbols = list(set(symbols))
    else:
        # Asya Marketi
        symbols = [
            "BABA", "MCHI", "BIDU", "JD", "NTDOY", "TCEHY", "XIACY", "FXI", "VO", "VOO", 
            "PDD", "SONY"
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
            
    status_place.success(f"Tamamlandı! Toplam Süre: {time.time() - start_time:.1f} sn")
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
    
    # Apply Logic: 
    # Üst Uzaklık: 0-10 -> Açık Kırmızı (#ff9999), <0 -> Koyu Kırmızı (#8b0000)
    def style_upper(val):
        if pd.isna(val): return ''
        if val < 0: return 'background-color: #8b0000; color: white'
        elif 0 <= val < 10: return 'background-color: #ff9999; color: black'
        return ''

    # Alt Uzaklık: 0-10 -> Açık Yeşil (#99ff99), <0 -> Koyu Yeşil (#006400)
    def style_lower(val):
        if pd.isna(val): return ''
        if val < 0: return 'background-color: #006400; color: white'
        elif 0 <= val < 10: return 'background-color: #99ff99; color: black'
        return ''

    styler.map(style_upper, subset=['Üst Uzaklık %'])
    styler.map(style_lower, subset=['Alt Uzaklık %'])

    # Apply Selection Highlight (Yellow) - Overrides previous colors
    selected_rows = st.session_state.get("stocks_table", {}).get("selection", {}).get("rows", [])
    
    def highlight_selected_rows(row):
        if row.name in selected_rows:
            return ['background-color: #ffff00; color: black'] * len(row)
        return [''] * len(row)
        
    styler.apply(highlight_selected_rows, axis=1)

    event = st.dataframe(
        styler,
        on_select="rerun",
        selection_mode="multi-row",
        hide_index=True,
        key="stocks_table"
    )
    
    # Optional: Display selected rows or actions if needed
    if event.selection.rows:
        st.write(f"Seçilen satır sayısı: {len(event.selection.rows)}")