import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import os
# Helper function to convert Streamlit date to the required format
def convert_streamlit_date_to_str(date_obj):
    return date_obj.strftime('%d/%m/%Y')


def construire_url_tmp(date):
    """Construct the URL for the specified date to download the CSV data."""
    if isinstance(date, str):
        date = pd.to_datetime(date, format='%d/%m/%Y')
    
    date_formatee = date.strftime('%d/%m/%Y').replace('/', '%2F')
    base_url = 'https://www.bkam.ma/Marches/Principaux-indicateurs/Marche-obligataire/Marche-des-bons-de-tresor/Marche-secondaire/Taux-de-reference-des-bons-du-tresor'
    params = f'?date={date_formatee}&block=e1d6b9bbf87f86f8ba53e8518e882982#address-c3367fcefc5f524397748201aee5dab8-e1d6b9bbf87f86f8ba53e8518e882982'
    
    return f'{base_url}{params}'


def telecharger_csv_tmp(date, save_directory="downloads"):
    """Download the CSV data for the specified date and save it locally."""
    try:
        if isinstance(date, str):
            date = pd.to_datetime(date, format='%d/%m/%Y')
        
        url_page = construire_url_tmp(date)
        response = requests.get(url_page)
        response.raise_for_status()  # Check for HTTP errors

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')
        
        if table is not None:
            headers = [header.text.strip() for header in table.find_all('th')]
            rows = []
            for row in table.find_all('tr')[1:]:
                row_data = [value.text.strip() for value in row.find_all('td')]
                if "Total" not in row_data:
                    rows.append(row_data)
            
            df = pd.DataFrame(rows, columns=headers)
            
            # Rename and clean columns
            df['Date déchéance'] = df['Date d\'échéance']
            df = df.drop(['Transaction', 'Date d\'échéance'], axis=1)
            
            # Create the directory if it does not exist
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # Define file name and save path
            filename = f"BKAM_Data_{date.strftime('%Y-%m-%d')}.csv"
            save_path = os.path.join(save_directory, filename)
            
            # Save the DataFrame to a CSV file
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            return df  # Return the DataFrame

        else:
            return pd.DataFrame()

    except requests.HTTPError as e:
        return pd.DataFrame()
    except requests.RequestException as e:
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()


def process_bkam_data(df):
    """Process the BKAM data by cleaning and converting columns."""
    df = df.rename(columns={"Taux moyen pondéré": "TMP"})
    df["TMP"] = df['TMP'].str.replace('%', '').replace(',', '.', regex=True).astype(float)

    # Convert dates to datetime format
    df["Date de la valeur"] = pd.to_datetime(df["Date de la valeur"], format="%d/%m/%Y")
    df["Date déchéance"] = pd.to_datetime(df["Date déchéance"], format="%d/%m/%Y")

    # Calculate the maturity in days
    df["Maturite jour"] = (df['Date déchéance'] - pd.Timestamp('today')).dt.days
    
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
    interp_func = interp1d(bond_data['Maturite jour'], bond_data['TMP'], kind='cubic', fill_value="extrapolate")

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

    for i in range(3):
        zero_coupon.loc[i, 'Taux ZC'] = zero_coupon.loc[i, 'Taux']

    for n in range(3, len(zero_coupon)):
        tN = zero_coupon.loc[n, 'Taux']
        summation = 0
        
        for i in range(1, n):
            tN_i = zero_coupon.loc[n, 'Taux']
            ZC_i = zero_coupon.loc[i, 'Taux ZC']
            discounted_taux = (1 / (1 + tN_i) ** (i + 1))
            discounted_zc = (1 / (1 + ZC_i) ** (i + 1))
            summation += discounted_taux - discounted_zc
        
        final_term = 1 / (1 + tN) ** (n + 1)
        ZC_N = (1 / (summation + final_term)) ** (1 / (n + 1)) - 1
        zero_coupon.loc[n, 'Taux ZC'] = ZC_N

    return zero_coupon


def construire_url_monia(date):
    if isinstance(date, str):
        date = pd.to_datetime(date, format='%d/%m/%Y')
    date_formatee = date.strftime('%d/%m/%Y').replace('/', '%2F')
    base_url = 'https://www.bkam.ma/Marches/Principaux-indicateurs/Marche-monetaire/Indice-monia-moroccan-overnight-index-average'
    params = f'?date={date_formatee}&block=4734c7b73113d8d72895a19090974066#address-30551c1667f5f2004fb0019220d41795-4734c7b73113d8d72895a19090974066'
    return f'{base_url}{params}'


def telecharger_csv_monia(date):
    try:
        if isinstance(date, str):
            date = pd.to_datetime(date, format='%d/%m/%Y')
        
        url_page = construire_url_monia(date)

        response = requests.get(url_page)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find('table')
        if table is not None:
            headers = [header.text.strip() for header in table.find_all('th')]
            rows = []
            for row in table.find_all('tr')[1:]:
                rows.append([value.text.strip() for value in row.find_all('td')])

            df = pd.DataFrame(rows, columns=headers)
            df['Date'] = date.strftime('%Y-%m-%d')
            return df
        else:
            return pd.DataFrame()

    except requests.HTTPError as e:
        return pd.DataFrame()
    except requests.RequestException as e:
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

import requests

def get_monia_rate(date):
    """Fetch the MONIA rate for a given date."""
    try:
        date_str = date.strftime('%d/%m/%Y')
        url_page = construire_url_monia(date)

        # Using a session
        session = requests.Session()

        # Adding custom headers to imitate a browser request
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://www.bkam.ma",
        })

        # Making the request with the session
        response = session.get(url_page)
        response.raise_for_status()  # Check for HTTP errors

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')

        if table is not None:
            headers = [header.text.strip() for header in table.find_all('th')]
            rows = []
            for row in table.find_all('tr')[1:]:
                rows.append([value.text.strip() for value in row.find_all('td')])

            df = pd.DataFrame(rows, columns=headers)
            df['Date'] = date.strftime('%Y-%m-%d')
            monia_rate_str = df.loc[df['Date de référence'] == date_str, 'Indice MONIA'].values[0]
            monia_rate_real = float(monia_rate_str.replace(',', '.').replace('%', '').strip()) / 100
            return monia_rate_real
        else:
            return None

    except Exception as e:
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
        'Maturity_1': 0,
        'Maturity_2': maturities[0],
        'Forward_Rate': monia_rate
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


def calculate_fixed_rate(daily_forward_rates_with_monia_df, start, end, frequence=30, notionnel=10000, tolerance=1e-4, max_iterations=1000):
    lower_bound = 0.00
    upper_bound = 0.4
    
    for _ in range(max_iterations):
        fixed_rate_guess = (lower_bound + upper_bound) / 2
        fixed_leg_value = fixedleg(daily_forward_rates_with_monia_df, start, end, frequence, fixed_rate=fixed_rate_guess, notionnel=notionnel)
        floating_leg_value = floatleg(daily_forward_rates_with_monia_df, start, end, frequence, notionnel)

        difference = fixed_leg_value - floating_leg_value
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

def main():
    st.title("IRS Swap Pricer with Dynamic MONIA and Zero-Coupon Rates")
    
    st.sidebar.header("Input Parameters")
    
    # Streamlit date picker returns a datetime.date object, convert to datetime format
    valuation_date = st.sidebar.date_input("Valuation Date", datetime.today())
    start_date = pd.to_datetime(valuation_date)
    end_date = st.sidebar.date_input("End Date", datetime.today())
    end_date = pd.to_datetime(end_date)
    
    notional = st.sidebar.number_input("Notional", value=10000, min_value=0)
    frequency = st.sidebar.number_input("Frequency (days)", value=30, min_value=1)
    fixed_rate_input = st.sidebar.number_input("Fixed Rate (%)", value=3.0, format="%.4f") / 100
    is_fixed_payer = st.sidebar.radio("Are you the Fixed Payer?", ["Yes", "No"]) == "Yes"
    
    # Fetch MONIA rate based on valuation date
    monia_rate = get_monia_rate(start_date)
    if monia_rate is None:
        st.error("Failed to retrieve MONIA rate for the selected date.")
        return
    
    st.sidebar.write(f"MONIA Rate on {start_date.strftime('%Y-%m-%d')}: {monia_rate * 100:.4f}%")
    
    # Fetch zero-coupon rates based on valuation date
    bond_data = get_zero_coupon_rates(start_date)
    if bond_data.empty:
        st.error("Failed to retrieve zero-coupon rates for the selected date.")
        return
    
    st.sidebar.write("Zero-Coupon Rates Retrieved:")
    st.sidebar.dataframe(bond_data)
    
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
    
    # Display the NPV calculated from the fixed rate
    st.header("Pricing Results")
    st.write(f"**Current Fixed Rate:** {fixed_rate_input * 100:.4f}%")
    st.write(f"**Fixed Leg Value:** {fixed_leg_value:.2f}")
    st.write(f"**Floating Leg Value:** {float_leg_value:.2f}")
    st.write(f"**Net Present Value (NPV):** {npv_current:.2f}")
    
    # Allow the user to input a new fixed rate and recalculate the NPV
    st.sidebar.subheader("Modify Fixed Rate to Recalculate NPV")
    user_fixed_rate = st.sidebar.slider("Fixed Rate (%)", min_value=0.0, max_value=10.0, value=fixed_rate_input * 100, step=0.01) / 100
    
    # Recalculate NPV with the user-specified fixed rate
    fixed_leg_value_user_rate = fixedleg(daily_forward_rates_with_monia_df, start_date, end_date, frequency, user_fixed_rate, notional)
    npv_user_fixed_rate = calculate_npv(fixed_leg_value_user_rate, float_leg_value, is_fixed_payer)
    
    # Display the recalculated NPV
    st.write(f"**NPV with {user_fixed_rate * 100:.4f}% Fixed Rate:** {npv_user_fixed_rate:.2f}")
    
    # Allow the user to input a target NPV and calculate the required fixed rate
    st.sidebar.subheader("Target NPV to Calculate Fair Rate")
    target_npv = st.sidebar.number_input("Target NPV", value=0.0)
    
    # Calculate the fair fixed rate required to achieve the target NPV
    fair_fixed_rate = calculate_fixed_rate(daily_forward_rates_with_monia_df, start_date, end_date, frequency, notionnel=notional, tolerance=1e-4, max_iterations=1000)
    
    # Display the calculated fair fixed rate
    st.write(f"**Fair Fixed Rate for NPV of {target_npv:.2f}:** {fair_fixed_rate * 100:.4f}%")
    
    # Visualize the forward rates for the selected period
    st.subheader("Forward Rates Visualization")
    
    # Filter the forward rates to only show data for the period between start_date and end_date
    filtered_forward_rates_df = daily_forward_rates_with_monia_df[
        (daily_forward_rates_with_monia_df['Maturity_2'] >= (start_date - pd.Timestamp('today')).days) &
        (daily_forward_rates_with_monia_df['Maturity_2'] <= (end_date - pd.Timestamp('today')).days)
    ]
    
    st.line_chart(filtered_forward_rates_df["Forward_Rate"])
    
    # Optional: Display detailed forward rates
    with st.expander("Show Forward Rates Data"):
        st.dataframe(filtered_forward_rates_df)


if __name__ == "__main__":
    main()

