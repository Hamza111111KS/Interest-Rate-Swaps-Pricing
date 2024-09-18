import requests
import base64
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import os

# Set Streamlit to wide layout
st.set_page_config(layout="wide")
# Helper function to convert Streamlit date to the required format
def convert_streamlit_date_to_str(date_obj):
    return date_obj.strftime('%d/%m/%Y')
import pandas as pd
from datetime import datetime




def construire_url_tmp(date):
    """Construct the URL for the specified date to download the CSV data."""
    if isinstance(date, str):
        date = pd.to_datetime(date, format='%d/%m/%Y')
    
    date_formatee = date.strftime('%d/%m/%Y').replace('/', '%2F')
    base_url = 'https://www.bkam.ma/fr/export/blockcsv/2340/c3367fcefc5f524397748201aee5dab8/e1d6b9bbf87f86f8ba53e8518e882982'
    params = f'?date={date_formatee}&block=e1d6b9bbf87f86f8ba53e8518e882982?t=1726615279'
    
    return f'{base_url}{params}'




# Function to download, process, and save the CSV file
def telecharger_csv_tmp(date, save_directory="downloads"):
    """Download, clean, and save the CSV data for the specified date."""
    try:
        url = construire_url_tmp(date)
        response = requests.get(url)
        response.raise_for_status()

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Set the paths for raw and processed files
        raw_filename = f"New_BKAM_Data_{date.strftime('%Y-%m-%d')}.csv"
        processed_filename = f"BKAM_Data_{date.strftime('%Y-%m-%d')}.csv"
        
        raw_save_path = os.path.join(save_directory, raw_filename)
        processed_save_path = os.path.join(save_directory, processed_filename)

        # Save the raw CSV file
        with open(raw_save_path, 'wb') as file:
            file.write(response.content)

        # Read the CSV file with the correct delimiter (semicolon) and skip the first 3 lines
        df = pd.read_csv(raw_save_path, skiprows=3, delimiter=';')

        # Check the columns after reading the file
        if len(df.columns) == 4:
            df.columns = ['Date déchéance', 'Transaction', 'Taux moyen pondéré', 'Date de la valeur']
        else:
            print(f"Unexpected number of columns: {len(df.columns)}")
            return pd.DataFrame()

        # Remove rows with 'Total' and null values
        df = df[~df['Date déchéance'].str.contains("Total", na=False)]
        df = df.dropna()

        # Convert the date columns to datetime
        df['Date déchéance'] = pd.to_datetime(df['Date déchéance'], format='%d/%m/%Y')
        df['Date de la valeur'] = pd.to_datetime(df['Date de la valeur'], format='%d/%m/%Y')

        # Clean the interest rate column and convert it to numeric, then format it with French notation
        df['Taux moyen pondéré'] = df['Taux moyen pondéré'].str.replace('%', '').replace(',', '.', regex=True).astype(float)

        # Format the 'Taux moyen pondéré' back to a string with the required French formatting
        df['Taux moyen pondéré'] = df['Taux moyen pondéré'].apply(lambda x: f"{x:,.2f}".replace(",", " ").replace(".", ",") + " %")

        # Drop the 'Transaction' column if it's not needed
        df = df.drop(columns=['Transaction'])

        # Convert the 'Date d\'échéance' and 'Date de la valeur' back to the required date format
        df['Date déchéance'] = df['Date déchéance'].dt.strftime('%d/%m/%Y')
        df['Date de la valeur'] = df['Date de la valeur'].dt.strftime('%d/%m/%Y')

        # Select relevant columns and save the cleaned data to a new CSV file
        df[['Taux moyen pondéré', 'Date de la valeur', 'Date déchéance']].to_csv(processed_save_path, index=False, sep=';')

        print(f"Processed data saved to {processed_save_path}")

        return df[['Taux moyen pondéré', 'Date de la valeur', 'Date déchéance']]

    except requests.HTTPError as e:
        print(f"HTTP Error: {e}")
        return pd.DataFrame()
    except requests.RequestException as e:
        print(f"Request Exception: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def process_bkam_data(df):
    """Process the BKAM data by cleaning and converting columns."""
    df = df.rename(columns={"Taux moyen pondéré": "TMP"})
    df["TMP"] = df['TMP'].str.replace('%', '').replace(',', '.', regex=True).astype(float)

    # Convert dates to datetime format
    df["Date de la valeur"] = pd.to_datetime(df["Date de la valeur"], format="%d/%m/%Y")
    df["Date déchéance"] = pd.to_datetime(df["Date déchéance"], format="%d/%m/%Y")

    # Calculate the maturity in days
    df["Maturite jour"] = (df['Date déchéance'] - df["Date de la valeur"]).dt.days
    
    # Return only relevant columns for interpolation
    bond_data = df[['Maturite jour', 'TMP']].dropna()
    
    return bond_data


def get_zero_coupon_rates(date):
    """Fetch and process zero-coupon rates for the given date."""
    df = telecharger_csv_tmp(date)
    if df.empty:
        return pd.DataFrame()
    
    bond_data = process_bkam_data(df)
    return bond_data


def interpolate_rates(bond_data, zero):
    """Interpolate rates using bond data and fill the zero dataframe."""
    interp_func = interp1d(bond_data['Maturite jour'], bond_data['TMP'], kind='linear', fill_value="extrapolate")

    zero['Taux'] = interp_func(zero['Maturite jour'])
    return zero


def setup(date):
    """Main function to set up and process data for a given date."""
    df = telecharger_csv_tmp(date)
    
    if df.empty:
        return None

    bond_data = process_bkam_data(df)
    
    zero = pd.DataFrame({
        'Maturite': ["13 Semaines", "26 Semaine", "1 AN", "2 ANS", "5 ANS", "10 ANS", "15 ANS", "20 ANS", "25 ANS", "30 ANS"],
        'Maturite jour': [91, 182, 365, 730, 1825, 3650, 5475, 7300, 9125, 10950]
    })
    
    zero_filled = interpolate_rates(bond_data, zero)
    return zero_filled


def bootstrap(zero_coupon):
    zero_coupon['Taux'] = zero_coupon['Taux'] / 100

    for j in range(len(zero_coupon)):
        if zero_coupon.loc[j, 'Maturite jour']<366:
            zero_coupon.loc[j, 'Taux ZC'] = zero_coupon.loc[j, 'Taux']
            
    for n in range(j, len(zero_coupon)):
        tN = zero_coupon.loc[n, 'Taux']
        summation = 0
        
        for i in range(1, n):
            tN_i = zero_coupon.loc[n, 'Taux']
            ZC_i = zero_coupon.loc[i, 'Taux ZC']
            discounted_taux = (1 / (1 + tN_i) ** (i ))
            discounted_zc = (1 / (1 + ZC_i) ** (i))
            summation += discounted_taux - discounted_zc
        
        final_term = 1 / (1 + tN) ** (n)
        ZC_N = (1 / (summation + final_term)) ** (1 / (n)) - 1
        zero_coupon.loc[n, 'Taux ZC'] = ZC_N

    return zero_coupon


def construire_url_monia(date):
    if isinstance(date, str):
        date = pd.to_datetime(date, format='%d/%m/%Y')
    date_formatee = date.strftime('%d/%m/%Y').replace('/', '%2F')
    base_url = 'https://www.bkam.ma/fr/export/blockcsv/566622/30551c1667f5f2004fb0019220d41795/4734c7b73113d8d72895a19090974066'
    params = f'?date={date_formatee}&block=4734c7b73113d8d72895a19090974066'
    return f'{base_url}{params}'


# Function to download and clean the MONIA CSV file
def get_monia_rate(date, save_directory="downloads"):
    """Fetch the MONIA rate for a given date by downloading and cleaning a CSV file."""
    try:
        # Construct the MONIA CSV download URL
        url = construire_url_monia(date)

        # Send request to download the CSV
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Create the save directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the raw CSV file for debugging or future reference
        raw_filename = f"MONIA_Raw_Data_{date.strftime('%Y-%m-%d')}.csv"
        raw_save_path = os.path.join(save_directory, raw_filename)
        with open(raw_save_path, 'wb') as file:
            file.write(response.content)

        # Read the CSV file (assuming it's semicolon-separated with headers starting at row 2)
        df = pd.read_csv(raw_save_path, sep=';', skiprows=2)

        # Check if the necessary columns are available
        if 'Indice MONIA' in df.columns and 'Date de référence' in df.columns:
            # Clean 'Indice MONIA' by replacing commas with dots and removing percentage signs
            df['Indice MONIA'] = df['Indice MONIA'].str.replace('%', '').str.replace(',', '.').astype(float)

            # Filter by the exact date provided
            date_str = date.strftime('%d/%m/%Y')
            df_filtered = df[df['Date de référence'] == date_str]

            # If we find MONIA for the date, return it
            if not df_filtered.empty:
                monia_rate = df_filtered['Indice MONIA'].values[0] / 100  # Convert to decimal
                return monia_rate
            else:
                print(f"No MONIA data available for {date_str}.")
                return None
        else:
            print("Required columns ('Indice MONIA', 'Date de référence') not found in the CSV file.")
            return None

    except requests.HTTPError as e:
        print(f"HTTP Error: {e}")
        return None
    except requests.RequestException as e:
        print(f"Request Exception: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def interpolate_zero_coupon_rates(maturities, rates, new_maturities):
    interpolation_func = interp1d(maturities, rates, kind='linear', fill_value="extrapolate")
    interpolated_rates = interpolation_func(new_maturities)
    return interpolated_rates


def calculate_forward_rate(spot_rate_1, spot_rate_2, maturity_1, maturity_2):
    forward_rate = (((1 + spot_rate_2) ** maturity_2) / ((1 + spot_rate_1) ** maturity_1)) ** (1 / (maturity_2 - maturity_1)) - 1
    return forward_rate


def calculate_daily_forward_rates_with_monia(maturities, spot_rates, monia_rate):
    forward_rates = []
    
    forward_rates.append({
        'Maturity_1': maturities[0],
        'Maturity_2': maturities[1],
        'Spot_Rate_1': spot_rates[0],
        'Spot_Rate_2': spot_rates[1],
        'Forward_Rate': monia_rate*100
    })
    
    for i in range(1, len(maturities)):
        maturity_1 = maturities[i - 1]
        maturity_2 = maturities[i]
        spot_rate_1 = spot_rates[i - 1]
        spot_rate_2 = spot_rates[i]
        
        forward_rate = calculate_forward_rate(spot_rate_1, spot_rate_2, maturity_1, maturity_2)
        forward_rates.append({
            'Maturity_1': maturity_1,
            'Maturity_2': maturity_2,
            'Spot_Rate_1': spot_rate_1,
            'Spot_Rate_2': spot_rate_2,
            'Forward_Rate': forward_rate
        })
    
    return pd.DataFrame(forward_rates)

# Function to calculate the floating leg with compounded discount factors
def floatleg(daily_forward_rates_with_monia_df, start, end, frequence=30, notionnel=10000):
    # Convert start and end dates
    start = pd.to_datetime(start, format="%d/%m/%Y")
    end = pd.to_datetime(end, format="%d/%m/%Y")
    
    # Calculate the number of days between start and end
    nombres_jour = (end - start).days
    
    # Number of periods (based on frequence)
    num_periods = nombres_jour // frequence

    # Initialize the list for cash flows
    float_leg_values = []
    
    # Initialize cumulative discount factor (start at 1)
    cumulative_discount_factor = 1.0
    p = 1.0
    # Loop through each period (frequence)
    for j in range(0, num_periods):
        # Get the start and end of the period
        period_start_day = (j) * frequence
        period_end_day = (j+1) * frequence
        
        # Initialize p (discount factor) for this period
        
        
        # Loop through each day in the period
        for i in range(period_start_day, period_end_day):
            # Retrieve the forward rate for day i
            forward_rate = daily_forward_rates_with_monia_df.loc[i, "Forward_Rate"]
            
            # Apply the discount factor for each day
            p *= (1 / (1 + (forward_rate/100) * (1 / 365)))
        
        # Update the cumulative discount factor
        cumulative_discount_factor *= p
        
        # Calculate the cash flow for this period using the cumulative discount factor
        float_leg_values.append(cumulative_discount_factor * notionnel*(forward_rate/100)*frequence/360)
    
    print(f'Float leg cash flows: {float_leg_values}')
    
    # Return the sum of all the float leg cash flows
    return sum(float_leg_values)



def fixedleg(daily_forward_rates_with_monia_df, start, end, frequence=30, fixed_rate=0.03, notionnel=10000):
    start = pd.to_datetime(start, format="%d/%m/%Y")
    end = pd.to_datetime(end, format="%d/%m/%Y")
    
    nombres_jour = (end - start).days
    num_periods = nombres_jour // frequence
    
    fixed_leg_values = []
    p = 1.0
    cumulative_discount_factor = 1.0

    for j in range(0, num_periods):
        period_start_day = (j) * frequence
        period_end_day = (j+1) * frequence
        
        for i in range(period_start_day, period_end_day):
            forward_rate = daily_forward_rates_with_monia_df.loc[i, "Forward_Rate"]
            p *= (1 / (1 + (forward_rate/100) * (1 / 365)))
        
        cumulative_discount_factor *= p
        fixed_payment = notionnel * fixed_rate * (frequence / 365)
        fixed_leg_values.append(fixed_payment * cumulative_discount_factor)
    
    print(f'Fixedt leg cash flows: {fixed_leg_values}')
    return sum(fixed_leg_values)



# Adjusted calculate_fixed_rate function to target specific NPV
def calculate_fixed_rate(daily_forward_rates_with_monia_df, start, end, frequence=30, notionnel=10000, target_npv=0, tolerance=1e-4, max_iterations=1000):
    lower_bound = 0.00
    upper_bound = 0.4
    
    for _ in range(max_iterations):
        fixed_rate_guess = (lower_bound + upper_bound) / 2
        fixed_leg_value = fixedleg(daily_forward_rates_with_monia_df, start, end, frequence, fixed_rate=fixed_rate_guess, notionnel=notionnel)
        floating_leg_value = floatleg(daily_forward_rates_with_monia_df, start, end, frequence, notionnel)

        difference = fixed_leg_value - floating_leg_value - target_npv
        if abs(difference) < tolerance:
            return fixed_rate_guess
        
        if difference > 0:
            upper_bound = fixed_rate_guess
        else:
            lower_bound = fixed_rate_guess
    
    return None

def calculate_npv(fixed_leg_value, float_leg_value, is_fixed_payer):
    if is_fixed_payer:
        return float_leg_value - fixed_leg_value
    else:
        return fixed_leg_value - float_leg_value


# Function to encode image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# CSS for positioning the logo and the title
st.markdown(
    """
    <style>
    .header {
        display: flex;
        justify-content: flex-start;
        align-items: center;
    }
    .header img {
        width: 150px;  /* Adjusted width to enlarge the logo */
        margin-right: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Get the base64 encoded version of the logo
# Correct path to your logo image (without duplication)
logo_base64 = get_base64_image("E:/AF/Internships/Application/Crédit du Maroc/Pricer Application/logo.png")  # Use single correct path

# Embed the logo and the title together in the header
st.markdown(
    f"""
    <div class="header">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo">
        <h1>IRS Swap Pricer with Dynamic MONIA and Zero-Coupon Rates</h1>
    </div>
    """,
    unsafe_allow_html=True
)
import streamlit as st
from datetime import datetime
import numpy as np
import os
import streamlit as st
from datetime import datetime
import numpy as np
import os

import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import os
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import os






# Get the base64 encoded version of the logo (ensure path is correct)
logo_base64 = get_base64_image("E:\AF\Internships\Application\Crédit du Maroc\Pricer Application\logo.png")  # Adjust path as needed



# Dropdown options for frequency
frequency_options = {
    "1 semaine": 7,
    "2 semaines": 14,
    "1 mois": 30,
    "3 mois": 90,
    "6 mois": 180,
    "1 an": 365,
    "5 ans": 1825
}


# Function to read the bond data from Excel
def read_bond_data(file_path):
    df = pd.read_excel(file_path)
    return df
# Function to clean and prepare the bond data
def prepare_bond_data(df):
    """
    Clean and prepare the bond data for further calculations.
    This includes removing commas, converting percentages, and parsing dates.

    Parameters:
    df: DataFrame containing the bond data
    
    Returns:
    Cleaned DataFrame ready for processing
    """
    # Remove commas from number columns (e.g., prices and nominal values)

    # Ensure that the column 'Valeur Nominale' exists or find the correct name
    if 'Valeur Nominale ' in df.columns:
        df['Valeur Nominale'] = df['Valeur Nominale '].replace({',': ''}, regex=True).astype(float)
    else:
        st.error("'Valeur Nominale' column not found!")

    df['Prix'] = df['Prix'].replace({',': ''}, regex=True).astype(float)
    df['Taux Nominal %'] = df['Taux Nominal %'].replace({',': ''}, regex=True).astype(float)

    # Convert percentage columns to decimals
    df['Taux Nominal %'] = df['Taux Nominal %'] / 100

    # Parse date columns
    df['Date Courbe'] = pd.to_datetime(df['Date Courbe'], format='%d/%m/%Y', dayfirst=True)
    df['Date d\'échéance'] = pd.to_datetime(df["Date d'&eacute;ch&eacute;ance"], format='%d/%m/%Y', dayfirst=True)

    # Return the cleaned DataFrame
    return df

# Function to calculate the bond durations
def calculate_bond_duration(df,start_date):
    df['Coupon Rate'] = df['Taux Nominal %'] / 100
    # Correct the date parsing with dayfirst=True
    df['Maturity in Years'] = (pd.to_datetime(df["Date d'&eacute;ch&eacute;ance"], dayfirst=True) - start_date).dt.days / 365
    durations = []
    for _, row in df.iterrows():
        coupon_rate = row['Coupon Rate']
        maturity_years = row['Maturity in Years']
        price = row['Prix']
        nominal_value = row['Valeur Nominale']
        coupon_payments = coupon_rate * nominal_value
        macaulay_duration = sum([(t * coupon_payments / (1 + coupon_rate)**t) for t in range(1, int(maturity_years) + 1)]) + \
                            (maturity_years * nominal_value / (1 + coupon_rate)**maturity_years)
        macaulay_duration /= price
        durations.append(macaulay_duration)
    df['Bond Duration'] = durations
    return df

# Function to match bond duration with swap duration
def match_bond_duration_with_swap(df, swap_duration):
    df['Duration Difference'] = abs(df['Bond Duration'] - swap_duration)
    closest_bond = df.loc[df['Duration Difference'].idxmin()]
    return closest_bond['Code ISIN']

# Function to calculate Modified Duration and DV01 of IRS
def calculate_modified_duration_and_dv01(daily_forward_rates_with_monia_df, start, end, frequence=30, fixed_rate=0.03, notionnel=10000, is_fixed_payer=True, rate_shift=1e-4):
    # Step 1: Calculate the original NPV at the current fixed rate
    fixed_leg_value = fixedleg(daily_forward_rates_with_monia_df, start, end, frequence, fixed_rate, notionnel)
    float_leg_value = floatleg(daily_forward_rates_with_monia_df, start, end, frequence, notionnel=notionnel)
    npv_original = calculate_npv(fixed_leg_value, float_leg_value, is_fixed_payer)

    # Step 2: Perturb the fixed rate by a small amount (rate_shift) and recalculate NPV
    perturbed_fixed_rate = fixed_rate + rate_shift
    fixed_leg_value_perturbed = fixedleg(daily_forward_rates_with_monia_df, start, end, frequence, perturbed_fixed_rate, notionnel)
    npv_perturbed = calculate_npv(fixed_leg_value_perturbed, float_leg_value, is_fixed_payer)

    # Step 3: Calculate Modified Duration
    modified_duration = (npv_perturbed - npv_original) / (rate_shift * npv_original)

    # Step 4: Calculate DV01
    dv01 = -modified_duration * npv_original / 10000  # 1 basis point = 1/10000 of 1%

    return modified_duration, dv01
# Main function for the pricer
def main():
    # Split the page into 3 vertical columns
    col1, col2, col3 = st.columns(3)

    # Column 1: Input Section
    with col1:
        st.header("Inputs")
        valuation_date = st.date_input("Valuation Date", datetime.today())
        start_date = pd.to_datetime(valuation_date)

        end_date = st.date_input("End Date", datetime.today())
        end_date = pd.to_datetime(end_date)

        notional = st.number_input("Notional", value=1000000, min_value=0)
        # Format the notional with spaces for thousands
        formatted_notional = f"{int(notional):,}".replace(",", " ")

        # Display the formatted notional
        st.write(f"Notional : {formatted_notional} MAD")
        # Frequency dropdown
        frequency_selection = st.selectbox("Frequency", list(frequency_options.keys()))
        frequency = frequency_options[frequency_selection]  # Convert to days based on the selection

        fixed_rate_input = 3.0000/100
        is_fixed_payer = st.radio("Are you the Fixed Payer?", ["Yes", "No"]) == "Yes"

        # Fetch MONIA rate based on valuation date
        monia_rate = get_monia_rate(start_date)
        if monia_rate is None:
            st.error("Failed to retrieve MONIA rate for the selected date.")
            return

        st.write(f"MONIA Rate on {start_date.strftime('%Y-%m-%d')}: {monia_rate * 100:.4f}%")

    # Column 2: Pricing Results
    with col2:
        st.header("Pricing Results")
        # Fetch zero-coupon rates based on valuation date
        bond_data = get_zero_coupon_rates(start_date)
        if bond_data.empty:
            st.error("Failed to retrieve zero-coupon rates for the selected date.")
            return

        # Prepare zero-coupon rates and maturities for interpolation
        maturities = bond_data['Maturite jour'].values / 365  # Convert days to years
        spot_rates = bond_data['TMP'].values

        # Interpolate zero-coupon rates for daily maturities
        new_maturities = np.linspace(1 / 365, maturities[-1], int(maturities[-1] * 365))
        interpolated_spot_rates = interpolate_zero_coupon_rates(maturities, spot_rates, new_maturities)

        # Calculate daily forward rates with MONIA
        daily_forward_rates_with_monia_df = calculate_daily_forward_rates_with_monia(new_maturities, interpolated_spot_rates, monia_rate)

        # Calculate floating and fixed legs with the inputted fixed rate
        fixed_leg_value = fixedleg(daily_forward_rates_with_monia_df, start_date, end_date, frequency, fixed_rate_input, notional)
        float_leg_value = floatleg(daily_forward_rates_with_monia_df, start_date, end_date, frequency, notionnel=notional)

        # Calculate NPV with the current fixed rate
        npv_current = calculate_npv(fixed_leg_value, float_leg_value, is_fixed_payer)

        # Calculate the fair fixed rate required to achieve the target NPV
        fair_fixed_rate = calculate_fixed_rate(daily_forward_rates_with_monia_df, start_date, end_date, frequency, notionnel=notional, target_npv=0, tolerance=1e-4, max_iterations=1000)

        # Display the NPV, fixed rate, and fair fixed rate
        st.write(f"**Current Fixed Rate:** {fixed_rate_input * 100:.4f}%")
        st.write(f"**Fair Fixed Rate:** {fair_fixed_rate * 100:.4f}%")
        st.write(f"**Fixed Leg Value:** {fixed_leg_value:.2f}")
        st.write(f"**Floating Leg Value:** {float_leg_value:.2f}")
        st.write(f"**Net Present Value (NPV):** {npv_current:.2f}")

        # Calculate Modified Duration and DV01
        modified_duration, dv01 = calculate_modified_duration_and_dv01(
            daily_forward_rates_with_monia_df, start_date, end_date, frequency, fixed_rate_input, notional, is_fixed_payer
        )
                # Display the modified duration and DV01
        st.write(f"**Modified Duration:** {modified_duration:.6f}")
        st.write(f"**DV01:** {dv01:.6f}")
        # Initialize the variable df_bonds_with_duration to None to avoid UnboundLocalError
        df_bonds_with_duration = None

        # Step 2: After pricing, allow the user to upload a bond file
        st.subheader("Upload Bond File and Match Duration")
        
        # Add a file uploader for the user to upload the bond file
        bond_file = st.file_uploader("Upload Bond File (Excel format)", type=["xlsx", "xls"])

        if bond_file:
            # Step 3: Read the bond data
            df_bonds = read_bond_data(bond_file)
            
            if df_bonds is not None:
                # Step 4: Prepare and clean the bond data
                df_bonds_cleaned = prepare_bond_data(df_bonds)

                # Step 5: Calculate bond durations and add a new column
                df_bonds_with_duration = calculate_bond_duration(df_bonds_cleaned, start_date)

                # Match bond duration with swap duration
                matching_bond_isin = match_bond_duration_with_swap(df_bonds_with_duration, modified_duration)

                # Display the matching bond ISIN
                st.write(f"Matching Bond ISIN with closest duration: {matching_bond_isin}")

    # Column 3: Data and Plots
    with col3:
        st.header("Forward Rates Visualization")

        # Filter the DataFrame for the selected period
        start_years_diff = (start_date - start_date).days / 365
        end_years_diff = (end_date - start_date).days / 365
        filtered_forward_rates_df = daily_forward_rates_with_monia_df[
            (daily_forward_rates_with_monia_df['Maturity_2'] >= start_years_diff) &
            (daily_forward_rates_with_monia_df['Maturity_2'] <= end_years_diff)
        ]

        # Display forward rates data
        with st.expander("Show Forward Rates Data"):
            st.dataframe(filtered_forward_rates_df)

        # Display zero-coupon rates data
        with st.expander("Show Zero-Coupon Rates Data"):
            st.dataframe(bond_data)

        # Only display the bond DataFrame with the duration column if df_bonds_with_duration exists
        if df_bonds_with_duration is not None:
            with st.expander("Bond Data with Calculated Duration"):
                st.dataframe(df_bonds_with_duration)



if __name__ == "__main__":
    # Check if the script has already been run
    if "STREAMLIT_RUN" not in os.environ:
        os.environ["STREAMLIT_RUN"] = "true"
        os.system('streamlit run A.py')
    else:
        main()
