'''Complete list of Vanguard ETFs trading on US stock exchanges.

Provides a categorized dictionary and helper functions for use in
ticker selection dropdowns throughout the application.
'''

# All 103 Vanguard ETFs organized by category.
# Each entry maps ticker -> full ETF name.

VANGUARD_ETFS = {
    # --- Broad U.S. Equity ---
    "VOO": "Vanguard S&P 500 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "VUG": "Vanguard Growth ETF",
    "VTV": "Vanguard Value ETF",
    "VIG": "Vanguard Dividend Appreciation ETF",
    "VO": "Vanguard Mid-Cap ETF",
    "VYM": "Vanguard High Dividend Yield Index ETF",
    "VB": "Vanguard Small-Cap ETF",
    "VT": "Vanguard Total World Stock ETF",
    "VV": "Vanguard Large-Cap ETF",
    "VNQ": "Vanguard Real Estate ETF",
    "VBR": "Vanguard Small Cap Value ETF",
    "MGK": "Vanguard Mega Cap Growth ETF",
    "VXF": "Vanguard Extended Market ETF",
    "VOE": "Vanguard Mid-Cap Value ETF",
    "VBK": "Vanguard Small-Cap Growth ETF",
    "VOT": "Vanguard Mid-Cap Growth ETF",
    "MGV": "Vanguard Mega Cap Value ETF",
    "MGC": "Vanguard Mega Cap ETF",
    "VCR": "Vanguard Consumer Discretionary ETF",
    "VONE": "Vanguard Russell 1000 ETF",
    "VONG": "Vanguard Russell 1000 Growth ETF",
    "VONV": "Vanguard Russell 1000 Value ETF",
    "VTWO": "Vanguard Russell 2000 ETF",
    "VTWG": "Vanguard Russell 2000 Growth ETF",
    "VTWV": "Vanguard Russell 2000 Value ETF",
    "VTHR": "Vanguard Russell 3000 ETF",
    "VOOG": "Vanguard S&P 500 Growth ETF",
    "VOOV": "Vanguard S&P 500 Value ETF",
    "IVOO": "Vanguard S&P Mid-Cap 400 ETF",
    "IVOG": "Vanguard S&P Mid-Cap 400 Growth ETF",
    "IVOV": "Vanguard S&P Mid-Cap 400 Value ETF",
    "VIOO": "Vanguard S&P Small-Cap 600 ETF",
    "VIOG": "Vanguard S&P Small-Cap 600 Growth ETF",
    "VIOV": "Vanguard S&P Small-Cap 600 Value ETF",
    # --- International Equity ---
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "VXUS": "Vanguard Total International Stock ETF",
    "VWO": "Vanguard FTSE Emerging Markets ETF",
    "VEU": "Vanguard FTSE All-World ex-US Index Fund",
    "VGK": "Vanguard FTSE Europe ETF",
    "VYMI": "Vanguard International High Dividend Yield ETF",
    "VSS": "Vanguard FTSE All-World ex-US Small-Cap ETF",
    "VPL": "Vanguard FTSE Pacific ETF",
    "VIGI": "Vanguard International Dividend Appreciation ETF",
    "VNQI": "Vanguard Global ex-U.S. Real Estate ETF",
    "VEXC": "Vanguard Emerging Markets Ex-China ETF",
    # --- Sector ETFs ---
    "VGT": "Vanguard Information Technology ETF",
    "VHT": "Vanguard Health Care ETF",
    "VFH": "Vanguard Financials ETF",
    "VDE": "Vanguard Energy ETF",
    "VPU": "Vanguard Utilities ETF",
    "VDC": "Vanguard Consumer Staples ETF",
    "VIS": "Vanguard Industrials ETF",
    "VOX": "Vanguard Communication Services ETF",
    "VAW": "Vanguard Materials ETF",
    # --- U.S. Bond / Fixed Income ---
    "BND": "Vanguard Total Bond Market ETF",
    "VCIT": "Vanguard Intermediate-Term Corporate Bond ETF",
    "BSV": "Vanguard Short-Term Bond ETF",
    "VTEB": "Vanguard Tax-Exempt Bond ETF",
    "VCSH": "Vanguard Short-Term Corporate Bond ETF",
    "VGIT": "Vanguard Intermediate-Term Treasury ETF",
    "BIV": "Vanguard Intermediate-Term Bond ETF",
    "VGSH": "Vanguard Short-Term Treasury ETF",
    "VTIP": "Vanguard Short-Term Inflation-Protected Securities ETF",
    "VMBS": "Vanguard Mortgage-Backed Securities ETF",
    "VGLT": "Vanguard Long-Term Treasury ETF",
    "VUSB": "Vanguard Ultra-Short Bond ETF",
    "VCLT": "Vanguard Long-Term Corporate Bond ETF",
    "VCRB": "Vanguard Core Bond ETF",
    "VWOB": "Vanguard Emerging Markets Government Bond ETF",
    "BLV": "Vanguard Long-Term Bond ETF",
    "VBIL": "Vanguard 0-3 Month Treasury Bill ETF",
    "EDV": "Vanguard Extended Duration Treasury ETF",
    "VTC": "Vanguard Total Corporate Bond ETF",
    "VTES": "Vanguard Short-Term Tax Exempt Bond ETF",
    "VTEI": "Vanguard Intermediate-Term Tax-Exempt Bond ETF",
    "VPLS": "Vanguard Core Plus Bond ETF",
    "VCRM": "Vanguard Core Tax-Exempt Bond ETF",
    "VCEB": "Vanguard ESG U.S. Corporate Bond ETF",
    "VGUS": "Vanguard Ultra-Short Treasury ETF",
    "VSDM": "Vanguard Short Duration Tax-Exempt Bond ETF",
    "VSDB": "Vanguard Short Duration Bond ETF",
    "VTEL": "Vanguard Long-Term Tax-Exempt Bond ETF",
    "VGHY": "Vanguard High-Yield Active ETF",
    "VGMS": "Vanguard Multi-Sector Income Bond ETF",
    "BNDP": "Vanguard Core-Plus Bond Index ETF",
    "VTP": "Vanguard Total Inflation-Protected Securities ETF",
    "VTG": "Vanguard Total Treasury ETF",
    "VGVT": "Vanguard Government Securities Active ETF",
    # --- International Bond ---
    "BNDX": "Vanguard Total International Bond ETF",
    "BNDW": "Vanguard Total World Bond ETF",
    # --- ESG ---
    "ESGV": "Vanguard ESG U.S. Stock ETF",
    "VSGX": "Vanguard ESG International Stock ETF",
    # --- Factor / Smart Beta ---
    "VFMO": "Vanguard U.S. Momentum Factor ETF",
    "VFVA": "Vanguard U.S. Value Factor ETF",
    "VFMF": "Vanguard U.S. Multifactor ETF",
    "VFQY": "Vanguard U.S. Quality Factor ETF",
    "VFMV": "Vanguard U.S. Minimum Volatility ETF",
    # --- State-Specific Muni ---
    "VTEC": "Vanguard California Tax-Exempt Bond ETF",
    "MUNY": "Vanguard New York Tax-Exempt Bond ETF",
    # --- Active ETFs ---
    "VUSV": "Vanguard Wellington U.S. Value Active ETF",
    "VDIG": "Vanguard Wellington Dividend Growth Active ETF",
    "VUSG": "Vanguard Wellington U.S. Growth Active ETF",
}


def get_etf_options():
    '''Return a list of formatted "TICKER - Name" strings for dropdown display,
    sorted alphabetically by ticker.

    OUTPUTS:
    list of str'''
    return [
        f"{ticker} - {name}"
        for ticker, name in sorted(VANGUARD_ETFS.items())
    ]


def get_ticker_from_option(option):
    '''Extract the ticker symbol from a formatted dropdown option string.

    INPUTS:
    option - str like "VTI - Vanguard Total Stock Market ETF"

    OUTPUTS:
    str - the ticker symbol (e.g. "VTI")'''
    return option.split(" - ")[0].strip()


def get_all_tickers():
    '''Return a sorted list of all Vanguard ETF ticker symbols.

    OUTPUTS:
    list of str'''
    return sorted(VANGUARD_ETFS.keys())
