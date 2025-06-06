import numpy as np
import pandas as pd
#from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
import pickle

def fed_yield_curve(fed_yield_curve,date_truncation="2021-11-01",date_end=False):
    
    fed_sheet=pd.read_csv(fed_yield_curve)
    
    date_row_index = fed_sheet[fed_sheet.isin(["Date"])].stack().index[0][0]
    useful_information_list=["Date","BETA0","BETA1","BETA2","BETA3","TAU1","TAU2","SVENY01","SVENY02","SVENY03","SVENY04","SVENY05","SVENY06",
                             "SVENY07"]
    ##
    fed_sheet_truncated=fed_sheet.iloc[8:,].reset_index(drop=True)
    fed_sheet_truncated.columns = fed_sheet_truncated.iloc[0]  # Assign the first row as the column names
    
    ##Remove unnecessary information
    # useful_information_list=["Date","BETA0","BETA1","BETA2","BETA3","TAU1","TAU2","SVENF01","SVENF02","SVENF03","SVENF04","SVENF05","SVENF06",
    #                          "SVENF07"]
    useful_information_list=["Date","BETA0","BETA1","BETA2","BETA3","TAU1","TAU2","SVENY01","SVENY02","SVENY03","SVENY04","SVENY05","SVENY06",
                             "SVENY07","SVENF01","SVENF02","SVENF03","SVENF04","SVENF05","SVENF06","SVENF07"]
    fed_sheet=fed_sheet_truncated[useful_information_list]
    fed_sheet=fed_sheet[1:]
    
    ##keep only after 1980
    fed_sheet["Date"]=pd.to_datetime(fed_sheet["Date"])
    fed_sheet = fed_sheet[fed_sheet["Date"] >= "1980-01-02"]  # Filter for the specific date
    fed_sheet = fed_sheet[fed_sheet["Date"] >= date_truncation]  # Filter for the specific date
    if date_end!=False:
        fed_sheet = fed_sheet[fed_sheet["Date"] <= date_end]  # Filter for the specific date

    fed_sheet = fed_sheet.dropna()  # Drop rows with any NaN values
    return fed_sheet
    

def svensson_model(t, beta0, beta1, beta2, beta3, tau1, tau2):
    """
    Svensson interest rate model function.

    Parameters:
    t (float or np.ndarray): Time to maturity (can be a scalar or an array).
    beta0 (float): Long-term level parameter.
    beta1 (float): Short-term slope parameter.
    beta2 (float): Medium-term curvature parameter.
    beta3 (float): Second medium/long-term curvature parameter.
    tau1 (float): Decay factor for the first exponential term.
    tau2 (float): Decay factor for the second exponential term.

    Returns:
    float or np.ndarray: Instantaneous forward rate(s) at time t.
    """
    term1 = beta0
    term2 = beta1 * np.exp(-t / tau1)
    term3 = beta2 * (t / tau1) * np.exp(-t / tau1)
    term4 = beta3 * (t / tau2) * np.exp(-t / tau2)
    return term1 + term2 + term3 + term4

def svensson_spot_rate(T, beta0, beta1, beta2, beta3, tau1, tau2):
    """
    Calculate spot rate for a given maturity T using the Svensson interest rate model.
    
    Parameters:
    T (float or np.ndarray): Maturity (in years) for which the spot rate is calculated.
    beta0, beta1, beta2, beta3, tau1, tau2 (float): Svensson model parameters.

    Returns:
    float or np.ndarray: Spot rate for the given maturity T.
    """
    # Create a grid of t values from 0 to T (1000 points)
    t_values = np.linspace(0, T, 1000)  # Correct use of T (not t)
    
    # Compute forward rate at each t in the interval [0, T]
    f_values = svensson_model(t_values, beta0, beta1, beta2, beta3, tau1, tau2) 
    
    # Compute the integral of f(t) over [0, T] using trapezoidal integration
    integral = np.trapz(f_values, t_values)  
    
    # Calculate the spot rate r(T) as the average forward rate
    spot_rate = integral / T
    
    return spot_rate

def compute_yields(fed_sheet,plotting=True):
    
    fed_sheet=fed_sheet.reset_index(drop=True)

    #collect parameters
    param_list = np.array(fed_sheet.iloc[0, 1:7])
    param_list = pd.to_numeric(param_list, errors='coerce')  # Replace non-numeric values with NaN
    
    
    #Predicting maturities
    predicted_maturities = np.array([i / 365 for i in range(0, 2555)])  # Daily maturities (1 day to ~7 years)
    predicted_forward_rates = [svensson_model(T, *param_list) for T in predicted_maturities]
    spot_rates = np.array([svensson_spot_rate(T, *param_list) for T in predicted_maturities])
    spot_rates=np.clip(spot_rates, 0, a_max=None)
    spot_rates[0]=spot_rates[1] ##Assume the zero day risk free rate is just the 1 day risk free rate
    
    ##
    observed_spot=fed_sheet.iloc[0,7:14]
    observed_spot =np.array( pd.to_numeric(observed_spot, errors='coerce'))  # Replace non-numeric values with NaN

    observed_forward=fed_sheet.iloc[0,14:]
    observed_forward =np.array( pd.to_numeric(observed_forward, errors='coerce'))  # Replace non-numeric values with NaN

    observed_maturities=np.array([1,2,3,4,5,6,7])
    date = fed_sheet.loc[0, "Date"].strftime("%Y_%m_%d")
    if plotting==True:
        current_directory = os.getcwd()
        figure1_folder = os.path.join(current_directory, "yield_curves")
        if not os.path.exists(figure1_folder):
            os.makedirs(figure1_folder)
            print(f"Created folder: {figure1_folder}")
        
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax1.scatter(observed_maturities, observed_spot, label="Observed spot Yield Curve", color="red", s=10)
        ax1.plot(predicted_maturities, spot_rates, label=f"Svensson curve on {date}", linestyle="--")
        ax1.set_title(f"Svenson spot yield curve on {date}")
        ax1.set_xlabel("Maturity{years}")
        ax1.set_ylabel("Yield (%)")
        ax1.legend()

        ax2.scatter(observed_maturities, observed_forward, label="Observed foward Yield Curve", color="red", s=10)
        ax2.plot(predicted_maturities, predicted_forward_rates, label=f"Svensson forward curve on {date}", linestyle="--")
        ax2.set_title(f"Svenson yield curve on {date}")
        ax2.set_xlabel("Maturity{years}")
        ax2.set_ylabel("Yield (%)")
        ax2.legend()

        
        save_path = os.path.join(figure1_folder, f"yield_curve on {date}")
        plt.savefig(save_path)
        plt.close()
    date = pd.to_datetime(fed_sheet.loc[0, "Date"])  # Convert to actual datetime

    # Create DataFrames for spot rates and forward rates
    spot_df = pd.DataFrame({
        "maturity_days": np.arange(len(spot_rates)),
        "spot_rate": spot_rates / 100  # Convert to percentage format if needed
    })
    
    predicted_forward_rates=np.array(predicted_forward_rates)
    forward_df = pd.DataFrame({
        "maturity_days": np.arange(len(predicted_forward_rates)),
        "forward_rate": predicted_forward_rates / 100  # Convert to percentage format if needed
    })
    print(f"this is {date}")
    if np.any(spot_rates)>10:
        print(date)
    if np.any(spot_rates)<0:
        print(date)
    return date,spot_df,forward_df

def create_yield_curves(fed_yield_curve_filename,date_truncation="1980-01-02",date_end=False):
    # Example usage
    fed_sheet = fed_yield_curve(fed_yield_curve_filename,date_truncation,date_end)
    grouped = fed_sheet.groupby("Date")
    
    # Initialize dictionaries for spot and forward rate curves
    spot_curves = {}
    forward_curves = {}
    
    for _, group in grouped:
        date, spot_df, forward_df = compute_yields(group, plotting=True)
        spot_curves[date] = spot_df["spot_rate"].values
        forward_curves[date] = forward_df["forward_rate"].values
    
    # Create DataFrames for spot and forward rate curves
    spot_curves_df = pd.DataFrame.from_dict(spot_curves, orient="index")
    forward_curves_df = pd.DataFrame.from_dict(forward_curves, orient="index")
    
    # Ensure datetime index for both DataFrames
    spot_curves_df.index = pd.to_datetime(spot_curves_df.index)
    forward_curves_df.index = pd.to_datetime(forward_curves_df.index)
    
    # Fill missing dates with forward fill
    spot_curves_df = spot_curves_df.resample('D').ffill()
    forward_curves_df = forward_curves_df.resample('D').ffill()
    return spot_curves_df,forward_curves_df

# spot,forward=create_yield_curves("fed_yield_curve.csv","2022-08-25","2022-09-05")
# yields_dict={"spot":spot,"forward":forward}
# filename = "yields_dict.pkl"

# # Save the dictionary to a file
# with open(filename, "wb") as file:
#     pickle.dump(yields_dict, file)
    
    
# def pasting_yields_option_df(option_df,yields_df,new_column_name):
#     """
#     Description of function: Pastes interpolated risk free rates from the yields_df onto the options df

#     Parameters
#     ----------
#     option_df : TYPE
#         DESCRIPTION.
#     yields_df : TYPE
#         DESCRIPTION.
#     new_column_name : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     option_df : TYPE
#         DESCRIPTION.

#     """
#     # Reshape spot_curves_df to long format
#     yields_df_long = yields_df.reset_index().melt(
#         id_vars=["index"], var_name="rounded_maturity", value_name=new_column_name
#     )
#     yields_df_long.rename(columns={"index": "qoute_dt"}, inplace=True)
    
#     # Ensure the data types align for merging
#     yields_df_long["qoute_dt"] = pd.to_datetime(yields_df_long["qoute_dt"])
#     yields_df_long["rounded_maturity"] = yields_df_long["rounded_maturity"].astype(float)/365  # Convert maturity to float for matching
    
#     # Assuming 'option_df' is your option dataframe
#     option_df["qoute_dt"] = pd.to_datetime(option_df["qoute_dt"])  # Ensure 'date' is datetime
#     option_df["maturity"] = option_df["maturity"].astype(float)  # Ensure 'maturity' is float for matching
    
#     # Merge the option dataframe with the reshaped spot curves
#     option_df = option_df.merge(
#         yields_df_long,
#         how="left",  # Use 'left' join to keep all rows in the option dataframe
#         on=["qoute_dt", "rounded_maturity"]
#     )
#     return option_df

# new_options_df=pasting_yields_option_df(option_df,spot,"my_risk_free_rate")
# new_options_df=pasting_yields_option_df(option_df,forward,"my_forward_rate")

