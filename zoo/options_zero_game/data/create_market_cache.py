import yfinance as yf
import pandas as pd
import os

# --- Configuration ---
# A diverse list of symbols representing different market types and regimes
SYMBOLS_TO_CACHE = [
    '360ONE.NS', '3MINDIA.NS', 'ABB.NS', 'ACC.NS', 'AIAENG.NS', 'APLAPOLLO.NS', 
    'AUBANK.NS', 'AADHARHFC.NS', 'AARTIIND.NS', 'AAVAS.NS', 'ABBOTINDIA.NS', 
    'ACE.NS', 'ADANIENSOL.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 
    'ADANIPOWER.NS', 'ATGL.NS', 'AWL.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'ABREL.NS', 
    'ABSLAMC.NS', 'AEGISLOG.NS', 'AFFLE.NS', 'AJANTPHARM.NS', 'AKUMS.NS', 
    'APLLTD.NS', 'ALKEM.NS', 'ALKYLAMINE.NS', 'ALOKINDS.NS', 'ARE&M.NS', 
    'AMBER.NS', 'AMBUJACEM.NS', 'ANANDRATHI.NS', 'ANANTRAJ.NS', 'ANGELONE.NS', 
    'APARINDS.NS', 'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'APTUS.NS', 'ACI.NS', 
    'ASAHIINDIA.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTERDM.NS', 'ASTRAZEN.NS', 
    'ASTRAL.NS', 'ATUL.NS', 'AUROPHARMA.NS', 'AVANTIFEED.NS', 'DMART.NS', 
    'AXISBANK.NS', 'BASF.NS', 'BEML.NS', 'BLS.NS', 'BSE.NS', 'BAJAJ-AUTO.NS', 
    'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BALAMINES.NS', 
    'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 
    'BANKINDIA.NS', 'MAHABANK.NS', 'BATAINDIA.NS', 'BAYERCROP.NS', 'BERGEPAINT.NS', 
    'BDL.NS', 'BEL.NS', 'BHARATFORG.NS', 'BHEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 
    'BHARTIHEXA.NS', 'BIKAJI.NS', 'BIOCON.NS', 'BIRLACORPN.NS', 'BSOFT.NS', 
    'BLUEDART.NS', 'BLUESTARCO.NS', 'BBTC.NS', 'BOSCHLTD.NS', 'BRIGADE.NS', 
    'BRITANNIA.NS', 'MAPMYINDIA.NS', 'CCL.NS', 'CESC.NS', 'CGPOWER.NS', 
    'CIEINDIA.NS', 'CRISIL.NS', 'CAMPUS.NS', 'CANFINHOME.NS', 'CANBK.NS', 
    'CAPLIPOINT.NS', 'CGCL.NS', 'CARBORUNIV.NS', 'CASTROLIND.NS', 'CEATLTD.NS', 
    'CELLO.NS', 'CENTRALBK.NS', 'CDSL.NS', 'CENTURYPLY.NS', 'CERA.NS', 
    'CHALET.NS', 'CHAMBLFERT.NS', 'CHEMPLASTS.NS', 'CHENNPETRO.NS', 
    'CHOLAHLDNG.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'CUB.NS', 'CLEAN.NS', 
    'COALINDIA.NS', 'COCHINSHIP.NS', 'COFORGE.NS', 'COLPAL.NS', 'CAMS.NS', 
    'CONCORDBIO.NS', 'CONCOR.NS', 'COROMANDEL.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 
    'CROMPTON.NS', 'CUMMINSIND.NS', 'CYIENT.NS', 'DLF.NS', 'DOMS.NS', 'DABUR.NS', 
    'DALBHARAT.NS', 'DATAPATTNS.NS', 'DEEPAKFERT.NS', 'DEEPAKNTR.NS', 'DELHIVERY.NS', 
    'DEVYANI.NS', 'DIVISLAB.NS', 'DIXON.NS', 'LALPATHLAB.NS', 'DRREDDY.NS', 
    'EIDPARRY.NS', 'EIHOTEL.NS', 'EASEMYTRIP.NS', 'EICHERMOT.NS', 'ELECON.NS', 
    'ELGIEQUIP.NS', 'EMAMILTD.NS', 'EMCURE.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS', 
    'EQUITASBNK.NS', 'ERIS.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'NYKAA.NS', 
    'FEDERALBNK.NS', 'FACT.NS', 'FINEORG.NS', 'FINCABLES.NS', 'FINPIPE.NS', 
    'FSL.NS', 'FIVESTAR.NS', 'FORTIS.NS', 'GRINFRA.NS', 'GAIL.NS', 'GVT&D.NS', 
    'GMRAIRPORT.NS', 'GRSE.NS', 'GICRE.NS', 'GILLETTE.NS', 'GLAND.NS', 'GLAXO.NS', 
    'GLENMARK.NS', 'MEDANTA.NS', 'GODIGIT.NS', 'GPIL.NS', 'GODFRYPHLP.NS', 
    'GODREJAGRO.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GODREJPROP.NS', 'GRANULES.NS', 
    'GRAPHITE.NS', 'GRASIM.NS', 'GESHIP.NS', 'GRINDWELL.NS', 'GAEL.NS', 
    'FLUOROCHEM.NS', 'GUJGASLTD.NS', 'GMDCLTD.NS', 'GNFC.NS', 'GPPL.NS', 'GSFC.NS', 
    'GSPL.NS', 'HEG.NS', 'HBLENGINE.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 
    'HDFCLIFE.NS', 'HFCL.NS', 'HAPPSTMNDS.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 
    'HSCL.NS', 'HINDALCO.NS', 'HAL.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS', 
    'HINDUNILVR.NS', 'HINDZINC.NS', 'POWERINDIA.NS', 'HOMEFIRST.NS', 'HONASA.NS', 
    'HONAUT.NS', 'HUDCO.NS', 'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 
    'ISEC.NS', 'IDBI.NS', 'IDFCFIRSTB.NS', 'IFCI.NS', 'IIFL.NS', 'INOXINDIA.NS', 
    'IRB.NS', 'IRCON.NS', 'ITC.NS', 'ITI.NS', 'INDGN.NS', 'INDIACEM.NS', 
    'INDIAMART.NS', 'INDIANB.NS', 'IEX.NS', 'INDHOTEL.NS', 'IOC.NS', 'IOB.NS', 
    'IRCTC.NS', 'IRFC.NS', 'IREDA.NS', 'IGL.NS', 'INDUSTOWER.NS', 'INDUSINDBK.NS', 
    'NAUKRI.NS', 'INFY.NS', 'INOXWIND.NS', 'INTELLECT.NS', 'INDIGO.NS', 'IPCALAB.NS', 
    'JBCHEPHARM.NS', 'JKCEMENT.NS', 'JBMA.NS', 'JKLAKSHMI.NS', 'JKTYRE.NS', 
    'JMFINANCIL.NS', 'JSWENERGY.NS', 'JSWINFRA.NS', 'JSWSTEEL.NS', 'JPPOWER.NS', 
    'J&KBANK.NS', 'JINDALSAW.NS', 'JSL.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 
    'JUBLFOOD.NS', 'JUBLINGREA.NS', 'JUBLPHARMA.NS', 'JWL.NS', 'JUSTDIAL.NS', 
    'JYOTHYLAB.NS', 'JYOTICNC.NS', 'KPRMILL.NS', 'KEI.NS', 'KNRCON.NS', 'KPITTECH.NS', 
    'KSB.NS', 'KAJARIACER.NS', 'KPIL.NS', 'KALYANKJIL.NS', 'KANSAINER.NS', 
    'KARURVYSYA.NS', 'KAYNES.NS', 'KEC.NS', 'KFINTECH.NS', 'KIRLOSBROS.NS', 
    'KIRLOSENG.NS', 'KOTAKBANK.NS', 'KIMS.NS', 'LTF.NS', 'LTTS.NS', 'LICHSGFIN.NS', 
    'LTIM.NS', 'LT.NS', 'LATENTVIEW.NS', 'LAURUSLABS.NS', 'LEMONTREE.NS', 'LICI.NS', 
    'LINDEINDIA.NS', 'LLOYDSME.NS', 'LUPIN.NS', 'MMTC.NS', 'MRF.NS', 'LODHA.NS', 
    'MGL.NS', 'MAHSEAMLES.NS', 'M&MFIN.NS', 'M&M.NS', 'MAHLIFE.NS', 'MANAPPURAM.NS', 
    'MRPL.NS', 'MANKIND.NS', 'MARICO.NS', 'MARUTI.NS', 'MASTEK.NS', 'MFSL.NS', 
    'MAXHEALTH.NS', 'MAZDOCK.NS', 'METROBRAND.NS', 'METROPOLIS.NS', 'MINDACORP.NS', 
    'MSUMI.NS', 'MOTILALOFS.NS', 'MPHASIS.NS', 'MCX.NS', 'MUTHOOTFIN.NS', 
    'NATCOPHARM.NS', 'NBCC.NS', 'NCC.NS', 'NHPC.NS', 'NLCINDIA.NS', 'NMDC.NS', 
    'NSLNISP.NS', 'NTPC.NS', 'NH.NS', 'NATIONALUM.NS', 'NAVINFLUOR.NS', 
    'NESTLEIND.NS', 'NETWEB.NS', 'NETWORK18.NS', 'NEWGEN.NS', 'NAM-INDIA.NS', 
    'NUVAMA.NS', 'NUVOCO.NS', 'OBEROIRLTY.NS', 'ONGC.NS', 'OIL.NS', 'OLECTRA.NS', 
    'PAYTM.NS', 'OFSS.NS', 'POLICYBZR.NS', 'PCBL.NS', 'PIIND.NS', 'PNBHOUSING.NS', 
    'PNCINFRA.NS', 'PTCIL.NS', 'PVRINOX.NS', 'PAGEIND.NS', 'PATANJALI.NS', 
    'PERSISTENT.NS', 'PETRONET.NS', 'PFIZER.NS', 'PHOENIXLTD.NS', 'PIDILITIND.NS', 
    'PEL.NS', 'PPLPHARMA.NS', 'POLYMED.NS', 'POLYCAB.NS', 'POONAWALLA.NS', 'PFC.NS', 
    'POWERGRID.NS', 'PRAJIND.NS', 'PRESTIGE.NS', 'PGHH.NS', 'PNB.NS', 'QUESS.NS', 
    'RRKABEL.NS', 'RBLBANK.NS', 'RECLTD.NS', 'RHIM.NS', 'RITES.NS', 'RADICO.NS', 
    'RVNL.NS', 'RAILTEL.NS', 'RAINBOW.NS', 'RAJESHEXPO.NS', 'RKFORGE.NS', 'RCF.NS', 
    'RATNAMANI.NS', 'RTNINDIA.NS', 'RAYMOND.NS', 'REDINGTON.NS', 'RELIANCE.NS', 
    'ROUTE.NS', 'SBFC.NS', 'SBICARD.NS', 'SBILIFE.NS', 'SJVN.NS', 'SKFINDIA.NS', 
    'SRF.NS', 'SAMMAANCAP.NS', 'MOTHERSON.NS', 'SANOFI.NS', 'SAPPHIRE.NS', 
    'SAREGAMA.NS', 'SCHAEFFLER.NS', 'SCHNEIDER.NS', 'SCI.NS', 'SHREECEM.NS', 
    'RENUKA.NS', 'SHRIRAMFIN.NS', 'SHYAMMETL.NS', 'SIEMENS.NS', 'SIGNATURE.NS', 
    'SOBHA.NS', 'SOLARINDS.NS', 'SONACOMS.NS', 'SONATSOFTW.NS', 'STARHEALTH.NS', 
    'SBIN.NS', 'SAIL.NS', 'SWSOLAR.NS', 'SUMICHEM.NS', 'SPARC.NS', 'SUNPHARMA.NS', 
    'SUNTV.NS', 'SUNDARMFIN.NS', 'SUNDRMFAST.NS', 'SUPREMEIND.NS', 'SUVENPHAR.NS', 
    'SUZLON.NS', 'SWANENERGY.NS', 'SYNGENE.NS', 'SYRMA.NS', 'TBOTEK.NS', 
    'TVSMOTOR.NS', 'TVSSCS.NS', 'TANLA.NS', 'TATACHEM.NS', 'TATACOMM.NS', 'TCS.NS', 
    'TATACONSUM.NS', 'TATAELXSI.NS', 'TATAINVEST.NS', 'TATAMOTORS.NS', 'TATAPOWER.NS', 
    'TATASTEEL.NS', 'TATATECH.NS', 'TTML.NS', 'TECHM.NS', 'TECHNOE.NS', 'TEJASNET.NS', 
    'NIACL.NS', 'RAMCOCEM.NS', 'THERMAX.NS', 'TIMKEN.NS', 'TITAGARH.NS', 'TITAN.NS', 
    'TORNTPHARM.NS', 'TORNTPOWER.NS', 'TRENT.NS', 'TRIDENT.NS', 'TRIVENI.NS', 
    'TRITURBINE.NS', 'TIINDIA.NS', 'UCOBANK.NS', 'UNOMINDA.NS', 'UPL.NS', 
    'UTIAMC.NS', 'UJJIVANSFB.NS', 'ULTRACEMCO.NS', 'UNIONBANK.NS', 'UBL.NS', 
    'UNITDSPR.NS', 'USHAMART.NS', 'VGUARD.NS', 'VIPIND.NS', 'DBREALTY.NS', 'VTL.NS', 
    'VARROC.NS', 'VBL.NS', 'MANYAVAR.NS', 'VEDL.NS', 'VIJAYA.NS', 'VINATIORGA.NS', 
    'IDEA.NS', 'VOLTAS.NS', 'WELCORP.NS', 'WELSPUNLIV.NS', 'WESTLIFE.NS', 
    'WHIRLPOOL.NS', 'WIPRO.NS', 'YESBANK.NS', 'ZFCVINDIA.NS', 'ZEEL.NS', 
    'ZENSARTECH.NS', 'ZOMATO.NS', 'ZYDUSLIFE.NS', 'ECLERX.NS',
    '^NSEBANK',
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
