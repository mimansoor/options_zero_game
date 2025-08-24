import yfinance as yf
import pandas as pd
import os

# --- Configuration ---
# A diverse list of symbols representing the top 500 South American stocks
SYMBOLS_TO_CACHE = [
    # Brazil (B3 Exchange - .SA)
    'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'PETR3.SA', 'B3SA3.SA', 'ABEV3.SA',
    'WEGE3.SA', 'JBSS3.SA', 'SUZB3.SA', 'RENT3.SA', 'RADL3.SA', 'RDOR3.SA', 'GGBR4.SA',
    'PRIO3.SA', 'LREN3.SA', 'ITSA4.SA', 'BBAS3.SA', 'EQTL3.SA', 'CSAN3.SA', 'UGPA3.SA',
    'HAPV3.SA', 'VIVT3.SA', 'VBBR3.SA', 'TOTS3.SA', 'RAIZ4.SA', 'MGLU3.SA', 'BBSE3.SA',
    'EMBR3.SA', 'ELET3.SA', 'CMIG4.SA', 'CCRO3.SA', 'SBSP3.SA', 'CPLE6.SA', 'ENEV3.SA',
    'HYPE3.SA', 'TIMS3.SA', 'BRFS3.SA', 'CSNA3.SA', 'RAIL3.SA', 'ALPA4.SA', 'MULT3.SA',
    'CYRE3.SA', 'EGIE3.SA', 'ASAI3.SA', 'PCAR3.SA', 'AMER3.SA', 'MRFG3.SA', 'VAMO3.SA',
    'USIM5.SA', 'ENGI11.SA', 'TAEE11.SA', 'BEEF3.SA', 'EZTC3.SA', 'DXCO3.SA', 'LWSA3.SA',
    'CVCB3.SA', 'ARZZ3.SA', 'SMTO3.SA', 'MRVE3.SA', 'GOAU4.SA', 'NTCO3.SA', 'RRRP3.SA',
    'YDUQ3.SA', 'FLRY3.SA', 'QUAL3.SA', 'COGN3.SA', 'IRBR3.SA', 'AZUL4.SA', 'GOLL4.SA',
    'ECOR3.SA', 'BRAP4.SA', 'GMAT3.SA', 'JHSF3.SA', 'LOGG3.SA', 'TRPL4.SA', 'SOJA3.SA',
    'MELI34.SA', 'AURE3.SA', 'IGTI11.SA', 'CPFE3.SA', 'OIBR3.SA', 'CIEL3.SA', 'BPAN4.SA',
    'CASH3.SA', 'ENAT3.SA', 'PETZ3.SA', 'TEND3.SA', 'MOVI3.SA', 'SIMH3.SA', 'SLCE3.SA',
    'SMFT3.SA', 'PORT3.SA', 'RECV3.SA', 'AGRO3.SA', 'BRKM5.SA', 'CMIN3.SA', 'CURY3.SA',
    'GRND3.SA', 'LIGT3.SA', 'MDIA3.SA', 'MILS3.SA', 'MYPK3.SA', 'NEOE3.SA', 'ONCO3.SA',
    'PSSA3.SA', 'SBFG3.SA', 'SEQL3.SA', 'TUPY3.SA', 'ANIM3.SA', 'CAML3.SA', 'CEAB3.SA',
    'CGAS5.SA', 'CRFB3.SA', 'CSMG3.SA', 'DIRR3.SA', 'FESA4.SA', 'GFSA3.SA', 'HBOR3.SA',
    'INEP3.SA', 'JALL3.SA', 'KLBN11.SA', 'LAVV3.SA', 'LEVE3.SA', 'LJQQ3.SA', 'MLAS3.SA',
    'ODPV3.SA', 'OFSA3.SA', 'OPCT3.SA', 'PFRM3.SA', 'PLPL3.SA', 'PNVL3.SA', 'POMO4.SA',
    'POSI3.SA', 'RAPT4.SA', 'RCSL4.SA', 'RDNI3.SA', 'ROMI3.SA', 'RSID3.SA', 'SANB11.SA',
    'SAPR11.SA', 'SGPS3.SA', 'SHOW3.SA', 'STBP3.SA', 'SYNE3.SA', 'TRIS3.SA', 'TTEN3.SA',
    'UNIP6.SA', 'VIVA3.SA', 'VLID3.SA', 'VULC3.SA', 'WEST3.SA', 'ZAMP3.SA', 'AALR3.SA',
    'ABCB4.SA', 'AESB3.SA', 'AGXY3.SA', 'ALLD3.SA', 'ALUP11.SA', 'AMBP3.SA', 'ARML3.SA',
    'ATOM3.SA', 'AZEV4.SA', 'BAHI3.SA', 'BALM4.SA', 'BAZA3.SA', 'BDLL4.SA', 'BGIP4.SA',
    'BIOM3.SA', 'BLAU3.SA', 'BMGB4.SA', 'BMIN4.SA', 'BMOB3.SA', 'BNBR3.SA', 'BOBR4.SA',
    'BPAC11.SA', 'BRAP3.SA', 'BRBI11.SA', 'BRIT3.SA', 'BRIV4.SA', 'BRPR3.SA', 'BSLI4.SA',
    'CALI3.SA', 'CARD3.SA', 'CBAV3.SA', 'CCPR3.SA', 'CEDO4.SA', 'CEEB3.SA', 'CGRA4.SA',
    'CLSA3.SA', 'COCE5.SA', 'CPLE3.SA', 'CRDE3.SA', 'CRPG5.SA', 'CTKA4.SA', 'CTNM4.SA',
    'CTSA4.SA', 'DASA3.SA', 'DESK3.SA', 'DEXP3.SA', 'DOHL4.SA', 'DOTZ3.SA', 'ELET6.SA',
    'EMAE4.SA', 'ENMT4.SA', 'ESPA3.SA', 'ESTR4.SA', 'ETER3.SA', 'EUCA4.SA', 'EVEN3.SA',
    'FIQE3.SA', 'FHER3.SA', 'FRAS3.SA', 'FRIO3.SA', 'GEPA4.SA', 'GGBR3.SA', 'GGPS3.SA',
    'GNDI3.SA', 'GOAU3.SA', 'GSHP3.SA', 'GUAR3.SA', 'HAGA4.SA', 'HBRE3.SA', 'HBSA3.SA',
    'HETA4.SA', 'HOOT4.SA', 'IFCM3.SA', 'IGBR3.SA', 'JFEN3.SA', 'JOPA3.SA', 'KEPL3.SA',
    'KRSA3.SA', 'LAND3.SA', 'LCAM3.SA', 'LIPR3.SA', 'LLIS3.SA', 'LOGN3.SA', 'LPSB3.SA',
    'LUPA3.SA', 'LUXM4.SA', 'LVTC3.SA', 'MATD3.SA', 'MEAL3.SA', 'MEGA3.SA', 'MGEL4.SA',
    'MNPR3.SA', 'MODL11.SA', 'MTRE3.SA', 'MTSA4.SA', 'MWET4.SA', 'NGRD3.SA', 'NINJ3.SA',
    'ORVR3.SA', 'PARD3.SA', 'PATI3.SA', 'PDGR3.SA', 'PDTC3.SA', 'PEAB3.SA', 'PGMN3.SA',
    'PINE4.SA', 'PMAM3.SA', 'PNVL4.SA', 'PRBC4.SA', 'PRNR3.SA', 'PTBL3.SA', 'PTNT4.SA',
    'RANI3.SA', 'RCSL3.SA', 'RDOR3.SA', 'REDE3.SA', 'RNEW11.SA', 'RPAD5.SA', 'RSUL4.SA',
    'SANB3.SA', 'SAPR4.SA', 'SCAR3.SA', 'SEER3.SA', 'SHUL4.SA', 'SLED4.SA', 'SMLS3.SA',
    'SOND5.SA', 'SQIA3.SA', 'SULA11.SA', 'TASA4.SA', 'TECN3.SA', 'TELB4.SA', 'TFCO4.SA',
    'TGMA3.SA', 'TRAD3.SA', 'TRPN3.SA', 'UCAS3.SA', 'UNIP3.SA', 'UPSS3.SA', 'USIM3.SA',
    'VITT3.SA', 'VIVR3.SA', 'WEGE3.SA', 'WHRL4.SA', 'WIZS3.SA',

    # Chile (Santiago Stock Exchange - .SN)
    'SQM-B.SN', 'FALABELLA.SN', 'CMPC.SN', 'COPEC.SN', 'ENELAM.SN', 'CHILE.SN', 'BSANTANDER.SN',
    'BCI.SN', 'LTM.SN', 'ENTEL.SN', 'COLBUN.SN', 'SM-CHILE_B.SN', 'ANDINA-B.SN', 'CONCHATORO.SN',
    'RIPLEY.SN', 'PARAUCO.SN', 'SALFACORP.SN', 'VAPORES.SN', 'IAM.SN', 'CCU.SN', 'ENELCHILE.SN',
    'AESGENER.SN', 'AGUAS-A.SN', 'ANTARCHILE.SN', 'BANMEDICA.SN', 'CAP.SN', 'CENCOSUD.SN',
    'CENCOSHOPP.SN', 'ECL.SN', 'EMBONOR-B.SN', 'ENELGXCH.SN', 'FORUS.SN', 'HABITAT.SN',
    'HITES.SN', 'ILC.SN', 'INVERCAP.SN', 'ITAUCORP.SN', 'LAPOLAR.SN', 'MALLPLAZA.SN',
    'MASISA.SN', 'ORO-BLANCO.SN', 'QUINENCO.SN', 'SECURITY.SN', 'SK.SN', 'SM-CHILE_A.SN',
    'SMSAAM.SN', 'SONDA.SN', 'SQUIMICH-A.SN',

    # Argentina (Buenos Aires Stock Exchange - .BA)
    'GGAL.BA', 'YPFD.BA', 'PAM.BA', 'BMA.BA', 'TECO2.BA', 'LOMA.BA', 'CRES.BA', 'TX.BA',
    'SUPV.BA', 'CEPU.BA', 'EDN.BA', 'IRS.BA', 'PAMP.BA', 'TGSU2.BA', 'ALUA.BA', 'BBAR.BA',
    'BYMA.BA', 'CADO.BA', 'COME.BA', 'CTIO.BA', 'CVH.BA', 'DGCU2.BA', 'FERR.BA', 'FIPL.BA',
    'GCLA.BA', 'GRIM.BA', 'HARG.BA', 'INTR.BA', 'LEDE.BA', 'METR.BA', 'MIRG.BA', 'MOLA.BA',
    'MOLI.BA', 'MORI.BA', 'OEST.BA', 'PATA.BA', 'POLL.BA', 'RICH.BA', 'ROSE.BA', 'SAMI.BA',
    'SEMI.BA', 'TGNO4.BA', 'TRAN.BA', 'VALO.BA',

    # Colombia (Colombia Stock Exchange - .CN)
    'ECOPETROL.CN', 'BCOLOMBIA.CN', 'GRUPOSURA.CN', 'ISA.CN', 'PFBCOLOM.CN', 'GRUPOARGOS.CN',
    'CEMARGOS.CN', 'NUTRESA.CN', 'GEB.CN', 'CORFICOLCF.CN', 'PFGRUPSURA.CN', 'CELSIA.CN',
    'PFAVAL.CN', 'BOGOTA.CN', 'EXITO.CN', 'PFGRUPOARG.CN', 'PFCEMARGOS.CN', 'GRUPOAVAL.CN',
    'ETB.CN', 'MINEROS.CN', 'PROMIGAS.CN', 'TERPEL.CN', 'CANOALC.CN', 'CARACOLTV.CN',
    'CARTON.CN', 'CASALIMPIA.CN', 'CONCONCRET.CN', 'CORFERIAS.CN', 'DAHIATSU.CN', 'ENKA.CN',
    'FABRICATO.CN', 'FAMILIA.CN', 'FAVI.CN', 'GDL.CN', 'ICO.CN', 'IMO.CN', 'INVERARGOS.CN',
    'MANUFTS.CN', 'MEDICINA.CN', 'OCCIDENTE.CN', 'PAZRIO.CN', 'PETROMIN.CN', 'PRODUMIL.CN',
    'SAI.CN', 'SIDAUTO.CN', 'SIMIT.CN', 'TEJICONDOR.CN', 'TLA.CN', 'VALOREM.CN',

    # Peru (Bolsa de Valores de Lima - .LM) - smaller selection
    'SCCO.LM', 'BVN.LM', 'CPACASC1.LM', 'BAP.LM', 'IFS.LM', 'VOLCABC1.LM', 'ALICORC1.LM',
    'FERREYC1.LM', 'AENZAC1.LM', 'ENDISPC1.LM', 'CASAGRC1.LM', 'CORAREI1.LM', 'CVERDEC1.LM',
    'ENGIEC1.LM', 'GRAMONC1.LM', 'INRETC1.LM', 'LUSURC1.LM', 'MINSURI1.LM', 'MPLC1.LM', 'RELAPAC1.LM',
    'SIDERC1.LM', 'SNJUNC1.LM', 'TV.LM', 'UNACEMC1.LM',
]


# Use a path relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIRECTORY = os.path.join(SCRIPT_DIR, "market_data_cache")
DATA_PERIOD = "10y"
DATA_INTERVAL = "1d"

def create_market_data_cache():
    """
    Downloads historical daily price data, cleans it by removing non-positive prices,
    and saves it to a CSV file in a cache directory.
    This version uses a robust method to prevent metadata corruption in the CSV.
    """
    print(f"--- Creating Market Data Cache in '{CACHE_DIRECTORY}' ---")
    os.makedirs(CACHE_DIRECTORY, exist_ok=True)

    for ticker in SYMBOLS_TO_CACHE:
        print(f"Downloading {DATA_PERIOD} of {DATA_INTERVAL} data for {ticker}...")
        try:
            data = yf.download(ticker, period=DATA_PERIOD, interval=DATA_INTERVAL, progress=False, auto_adjust=True)

            if data.empty:
                print(f"Warning: No data found for ticker {ticker}. Skipping.")
                continue

            # --- ROBUST DATA CLEANING AND RECONSTRUCTION ---

            # 1. Isolate the 'Close' column into a pandas Series.
            clean_series = data['Close'].dropna()

            # 2. Filter out any non-positive prices to prevent numerical errors.
            clean_series = clean_series[clean_series > 0]

            # If after cleaning, the data is empty, skip.
            if clean_series.empty:
                print(f"Warning: No valid data remains for ticker {ticker} after cleaning. Skipping.")
                continue

            # 3. Create a BRAND NEW, clean DataFrame from the isolated Series.
            #    This is the key step to strip any unwanted metadata or complex headers.
            output_df = pd.DataFrame(clean_series)

            # 4. Explicitly name the index column 'Date'. This will become the first column in the CSV.
            output_df.index.name = 'Date'

            # 5. Save the newly created clean DataFrame.
            #    Replace characters in ticker that are invalid for filenames, like '^'.
            safe_ticker_name = ticker.replace('^', '')
            output_path = os.path.join(CACHE_DIRECTORY, f"{safe_ticker_name}.csv")
            output_df.to_csv(output_path) # Now the simplest to_csv call will work perfectly.

            print(f"Successfully saved {len(output_df)} rows of data for {ticker} to {output_path}")

        except Exception as e:
            print(f"An error occurred while downloading data for {ticker}: {e}")

    print("\n--- Market Data Cache Creation Complete! ---")

if __name__ == "__main__":
    create_market_data_cache()
