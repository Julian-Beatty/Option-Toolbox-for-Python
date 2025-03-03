import os
import pandas as pd
import warnings
import py_vollib.black_scholes_merton.implied_volatility
from arbitragerepair import constraints, repair
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
import py_vollib
from py_vollib_vectorized import implied_volatility
from KDEpy import*
import time  # Import the time module
import sys
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneOut



class OptionMarket:
    def __init__(self, option_data_prefix, stock_price_data_prefix,spot_curve_df,forward_curve_df):
        """
        Initializes the OptionMarket class with option data, par yield curve, and stock price data.

        Parameters:
        option_data_prefix (str): Prefix for the option data files.
        par_yield_curve_prefix (str): Prefix for the yield curve data files.
        stock_price_data_prefix (str): Prefix for the csv file containing stock price data.
        """
        self.option_data_prefix = option_data_prefix  # Store the prefix for option data files
        self.stock_price_data_prefix = stock_price_data_prefix  # Store stock price data
        
        self.spot_curve_df=spot_curve_df
        self.forward_curve_df=forward_curve_df
        self.option_df = self.load_option_data()  # Load the option data
        self.stock_df = self.load_stock_price_data()

        self.original_market_df = self.merge_option_stock_yields()
        
    def load_option_data(self):
        """
        Loads option data from CSV files in the current directory.

        Returns:
        DataFrame containing combined option data.
        """
        option_data_prefix=self.option_data_prefix
        try:
            current_directory_str = os.getcwd()  # Current working directory

            # List all files starting with the option_data_prefix
            options_csv_files = [f for f in os.listdir(current_directory_str) if f.startswith(option_data_prefix)]

            # Create full paths for the files
            options_filepaths_list = [os.path.join(current_directory_str, f) for f in options_csv_files]

            # Read the files into a DataFrame
            option_df = pd.concat(map(pd.read_csv, options_filepaths_list), ignore_index=True)
            print(f"We have successfully loaded and merged all CSV files beginning with '{option_data_prefix}' in your working directory. Storing in object.")
            return option_df
        except Exception as e:
            print(f"An error occurred while loading option data: {e}. Check that files beginning with {option_data_prefix} are in {current_directory_str}")
    def load_fed_sheet(self):
        fed_sheet_name=self.par_yield_curve_prefix
        fed_sheet=fed_yield_curve_maker(fed_sheet_name)
        
        yield_curve = {}
        group=fed_sheet.groupby(["Date"])
        group.apply(lambda x: yield_curve.update(compute_yields(x,plotting=False)))
        return yield_curve
    
    def load_stock_price_data(self):
        """
        Loads the stock data into a dataframe.
        Returns:
        Dataframe containing the stock price.
        """
        stock_price_data_prefix=self.stock_price_data_prefix
        current_directory_str = os.getcwd()  # Current working directory
            
            # Attempt to load stock data files
        try:
                # List all files starting with the stock_price_data_prefix
            stock_csv_files = [f for f in os.listdir(current_directory_str) if f.startswith(stock_price_data_prefix)]
                
            if not stock_csv_files:
                raise FileNotFoundError(f"No files found starting with '{stock_price_data_prefix}'.")
    
                # Create full paths for the files and read them into a DataFrame
            stock_filepaths_list = [os.path.join(current_directory_str, f) for f in stock_csv_files]
            stock_df = pd.concat(map(pd.read_csv, stock_filepaths_list), ignore_index=True)
            
            stock_header=stock_df.columns
            if any("ate" in column for column in stock_header): #finds date column name
                date_column_name = list(filter(lambda word: "ate" in word, stock_header))
            else:
                raise FileNotFoundError(f"The stock data file must contain a column labeled date. We could not find it")


                # Convert 'dates' column to datetime
            stock_df['Date_column'] = pd.to_datetime(stock_df[date_column_name[0]])
            print("--" * 20)  # Print a line of dashes for separation
            print(f"We have successfully loaded the stock data files beginning with '{stock_price_data_prefix}' in your working directory.")

                # Check for the required 'price' column
            if 'price' not in stock_df.columns:
                raise ValueError("The stock data file must contain a column labeled 'price'.")
            if 'price' in stock_df.columns:
                stock_df=stock_df[['Date_column','price']]
                
                all_dates = pd.date_range(start=stock_df['Date_column'].min(), 
                                          end=stock_df['Date_column'].max(), freq='D')
        
                # Find missing dates (i.e., days that should have data but don't)
                missing_dates = all_dates.difference(stock_df['Date_column'])
        
                # Reindex the DataFrame to include missing dates, then fill with the previous day's yield curve
                stock_df = stock_df.set_index('Date_column').reindex(all_dates).ffill().reset_index()
        
                # Rename index column back to Date
                stock_df = stock_df.rename(columns={'index': 'Date_column'})
            return stock_df
    
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return None  # Return None if there's an error
    
        except Exception as e:
            print(f"An error occurred while loading the stock data: {e}. Check that files beginning with '{stock_price_data_prefix}' are in {current_directory_str}")
            return None  # Return None if there's a different error

    def merge_option_stock_yields(self):
        """
        Merges the stock data and the option dataframe together, and appends the risk free rate, as calibrated from niegel-svenson model to another column beside the maturity.
        Additionally, only keeps OTM options by replacing ITM calls with OTM puts using the put-call parity.
        Returns:
        Dataframe containing options, stock data and risk free rate.
        
        """
        print("--"*20)
        print("We are now merging option, yield curve and stock data")
        option_df=self.load_option_data()
        #stock_df=self.load_stock_price_data()
        #self.stock_df=stock_df
        stock_df=self.stock_df
        spot_curve_df=self.spot_curve_df
        forward_curve_df=self.forward_curve_df

        #yield_curve_df=self.load_par_yield_curve()
        #self.yield_curve=yield_curve_df
        option_df.columns = option_df.columns.str.strip()
        header_list=option_df.columns.tolist()
        
        if "[BASE_CURRENCY]" in header_list:
            source_identification="Options_dx_bitcoin_eod_quotes"
            option_df=clean_btc_optionsdx(option_df,stock_df,spot_curve_df,forward_curve_df)

            print("--" * 20)  # Print a line of dashes for separation
            print("We believe this data is end of day quotes for bitcoin options from Options_DX")
        if "cp_flag" in header_list:
            option_df=clean_optionmetrics(option_df,stock_df,spot_curve_df,forward_curve_df)
            print("Options_metrics")
        
        option_df=option_df.reset_index(drop=True)
        #self.original_market_df=option_df
        return option_df
    def set_master_dataframe(self,volume=-1,maturity=[-1,5000],moneyness=[-1000,1000],rolling=False,date="all"):
        """
        

        Parameters
        ----------
        volume : float
            Removes all options with volume less than or equal to this number.
        maturity : list
            removes all options with maturity less than maturity[0] or greater than maturity[1]. If list has only 1 entry, only options with maturity exactly equal to maturity[0] are kept. Maturity is in DAYs.
            i.e maturity=[1,2] keeps all options with maturities of 1 or 2 days.
        Rolling: float
            if True Keeps all options with the maturity closest to float.
            
        Returns
        -------
        None.

        """
        original_dataframe=self.original_market_df.copy()
        
        original_dataframe=original_dataframe[original_dataframe["volume"]>volume]
        original_dataframe=original_dataframe[original_dataframe["bid_size"]>volume]
        original_dataframe=original_dataframe[original_dataframe["offer_size"]>volume]

        ###
        original_dataframe=original_dataframe[(original_dataframe["rounded_maturity"]>=maturity[0]/365) & (original_dataframe["rounded_maturity"]<=maturity[1]/365)]
        ##
        original_dataframe=original_dataframe[(original_dataframe["moneyness"]>=moneyness[0]) & (original_dataframe["moneyness"]<=moneyness[1])]
        ##
        if rolling != False:
            date_group=original_dataframe.groupby(["date"])
            original_dataframe = date_group.apply(lambda x: rolling_window(x,rolling))
        
        original_dataframe=original_dataframe.reset_index(drop=True)
        if date != "all":
            start_date = pd.to_datetime(date[0])
            end_date = pd.to_datetime(date[1])
            original_dataframe=original_dataframe[ (original_dataframe["qoute_dt"]>=start_date) & (original_dataframe["qoute_dt"]<=end_date)]
        self.master_dataframe=original_dataframe
        return None
    def compute_option_pdfs(self,bw_setting="cv_ml",kde_setting="ISJ",kde_scale=1,plot_pdf=True,truncate_pdf=False,plot_raw_pdf=True,foldername="figureplots"):
        """
        Computes option implied PDFs of the master-dataframe.

        Parameters
        ----------
        bw_setting : STR or [float], optional
            Interpolation setting for IV curve: Pick either "spline","cv_ml","cv_ls" or enter a manual number [float]. The default is "cv_ml".
        kde_setting : STR, optional
            BW method for KDE. Pick either "ISJ","scott","silverman. The default is "ISJ".
        kde_scale : float, optional
            Divides silverman or scotts bandwidth by kde_scale. The default is 1.
        plotting : True/False boolean, optional
            Saves plots. The default is True.
        truncate : True/False boolean, optional
            Truncates until CDF is 99.5%, starting from center. The default is False.
        plot_raw : true/False boolean, optional
            Also plots the raw pdf. The default is True.

        Returns
        -------
        result_dict : dict
            dictionary containing the dates/expiration as keys, and as a value, a dataframe containing strike,return and pdf axes.

        """
        option_market=self.master_dataframe
        date_group=option_market.groupby(["date",'exdate'])
        result_dict = {}
        date_group.apply(lambda x: result_dict.update(compute_pdf(x,bw_setting,kde_setting,kde_scale,plotting=plot_pdf,truncate=truncate_pdf,
                                                                  plot_raw=plot_raw_pdf,foldername=foldername)))
        
        return result_dict
    def estimate_option_pdfs(self,method,argument_dict,xlims=(-0.5,0.5)):
        """
        Computes state price densities from options

        Parameters
        ----------
        foldername : TYPE, optional
            DESCRIPTION. The default is "figure_plots".

        Returns
        -------
        None.

        """
        option_market=self.master_dataframe
        date_group=option_market.groupby(["date","exdate"])
        result_dict={}
        
        if method=="polynomial":
            date_group.apply(lambda x: result_dict.update(polynomial(x,argument_dict,interpolate_large = True,extrapolate_curve=True,xlims=xlims)))
        if method=="kernel ridge":
            date_group.apply(lambda x: result_dict.update(kernel_ridge_pdf(x,argument_dict,xlims=xlims)))
        if method=="SVR":
            date_group.apply(lambda x: result_dict.update(SVR_model(x,argument_dict)))
        if method=="SABR":
            date_group.apply(lambda x: result_dict.update(SABR_pdf(x,argument_dict,xlims=xlims)))
        if method=="local_polynomial":
            date_group.apply(lambda x: result_dict.update(local_polynomial(x,argument_dict,xlims=xlims)))
        if method=="lowess":
            date_group.apply(lambda x: result_dict.update(lowess(x,argument_dict,xlims=xlims)))
        return result_dict   
        
def clean_btc_optionsdx(option_df,stock_df,spot_curve_df,forward_curve_df):
    """
    Description of function: Merges stock data, yield data and option data into one dataframe, and does basic data cleaning.

    Parameters
    ----------
    option_df : dataframe
        dataframe of options
    stock_df : dataframe
        dataframe of stock prices.
    yield_curve_df : dataframe
        dataframe of yield curve.

    Returns
    option_df, a dataframe containing merged information and cleaning.
    None.

    """
    ###extracting header list and removing brackets
    header_list=option_df.columns.tolist()
    header_list = [entry.replace("[", "").replace("]", "") for entry in header_list]
    option_df.columns=header_list
    
    ##Renaming column headers to a standard convention, and creating date-time values        
    option_df=option_df.loc[:,["QUOTE_DATE","EXPIRY_DATE","ASK_PRICE","BID_PRICE","STRIKE",'MARK_IV',"VOLUME","BID_SIZE","ASK_SIZE","UNDERLYING_INDEX","UNDERLYING_PRICE","OPTION_RIGHT"]]
    option_df.columns=["date","exdate","best_offer","best_bid","strike","mid_iv","volume","bid_size","offer_size","underlying_future","underlying_price","option_right"]
    option_df['option_right'] = option_df['option_right'].str.strip()
    option_df['date'] = option_df['date'].str.strip()
    option_df['exdate'] = option_df['exdate'].str.strip()
    option_df['qoute_dt'] = pd.to_datetime(option_df['date'])# + pd.to_timedelta(8, unit='h') #qoute 8:00UTC
    option_df['expiry_dt'] = pd.to_datetime(option_df['exdate'])# + pd.to_timedelta(22, unit='h') #Expires 10PM central
            
    ###Scaling mid_iv to decimal
    option_df["mid_iv"]=option_df["mid_iv"]/100        
    ####Pasting stock data into option frame, and converts the options from BTC into US dollars.
    date_group=option_df.groupby(["date"])
    option_df = option_df.merge(stock_df, how='left', left_on='qoute_dt', right_on='Date_column')
    option_df = option_df.rename(columns={'price': 'stock_price'})
    
    #Adjusting datetimevalues #we use 0.001
    option_df=option_df[option_df["best_offer"]>0.001]
    
    option_df["best_offer"]=option_df["best_offer"]*option_df['stock_price']
    option_df["best_bid"]=option_df["best_bid"]*option_df['stock_price']
            
    ####Addressing deribit future issues See helper function
    date_group=option_df.groupby(['date','exdate'])
    print("--" * 20)  # Print a line of dashes for separation
    print("We are averaging bitcoin futures")    
    option_df=date_group.apply(lambda x: average_futures(x))
    option_df=option_df.reset_index(drop=True)

    ####Cleaning volume. Any missing data is replaced with zero. Leading space removed.
    option_df["volume"]=option_df["volume"].astype(str)
    option_df['volume'] =option_df['volume'].replace(' ', "0")
    option_df["volume"] =option_df["volume"].str.lstrip()
    option_df["volume"]=option_df["volume"].astype(float)
    option_df=option_df.reset_index(drop=True)
        
    #### Converting "calls" and "puts" to "c" and "p" respectively.
    option_df['option_right'] = option_df['option_right'].replace({'call': 'c', 'put': 'p'})
###############################################################General Cleaning

    ###Creating maturity in year fraction. For using black scholes formula I replace 0 with 0.001 for numerical reasons.
    option_df['maturity']= ((pd.to_datetime(option_df['exdate']) - pd.to_datetime(option_df['date'])).dt.days)/365
    option_df['maturity']=option_df['maturity']+(20/24)/365 #expires 10pm, qouted 2am.
    #option_df['maturity'] =option_df['maturity'].replace(0, 0.001)
    ##rounded maturity for grouping purposes
    option_df['rounded_maturity']= ((pd.to_datetime(option_df['exdate']) - pd.to_datetime(option_df['date'])).dt.days)/365
    ###Creating mid price. Removing any instance in which the mid price is negative
    option_df['mid_price']=(option_df['best_bid']+option_df['best_offer'])/2
    option_df=option_df[option_df['mid_price']>0]
    ##Creating Log-moneyness column
    option_df.loc[option_df['option_right']=='p','log moneyness'] = np.log(option_df['strike']/option_df['stock_price'])
    option_df.loc[option_df['option_right']=='c','log moneyness'] = np.log(option_df['stock_price']/option_df['strike'])
    
    ###################################################### Yield Curve Interpolation with Cubic Splines
    #######
    #option_df["risk_free_rate"]=0
    #option_df = attach_risk_free_rate(option_df, yield_curve)
    option_df=pasting_yields_option_df(option_df, spot_curve_df, "risk_free_rate")
    option_df=pasting_yields_option_df(option_df, forward_curve_df, "forward_risk_free_rate")
    ######
    # ##merging yield curve onto main option dataframe
    # option_df = option_df.merge(yield_curve_df, how='left', left_on='qoute_dt', right_on='Date')
    # option_df.drop(columns=['Date'], inplace=True)
    # date_group=option_df.groupby(['date'])
    # yield_curve_headers=yield_curve_df.columns.tolist()[1:]
    
    # print("--" * 20)  # Print a line of dashes for separation
    # print("We are interpolating the yield curve to match your options. This may take some time.")
    # ###Performing Yield Curve interpolation
    # option_df=option_df.reset_index(drop=True)
    # option_df=date_group.apply(lambda x: paste_rate(x,yield_curve_headers)) 
    # print("We are done interpolating Yield curve")
    # option_df=option_df.reset_index(drop=True)
    
    
    ################################################   Use only OTM options. Converts ITM puts to (OTM) calls via put-call parity. Remove ITM calls. 
    print("--" * 20)  # Print a line of dashes for separation
    print("Converting puts to calls. If both are present we use only OTM options.")
    date_group=option_df.groupby(['date','exdate'])
    option_df=date_group.apply(lambda x: use_only_OTM(x))
    option_df=option_df.reset_index(drop=True)

    ##############################################  Aggregates duplicate option prices if they exist
    #Aggregates duplicates mid prices *See remove_duplicates function
    
    date_group=option_df.groupby(['date','exdate'])
    print("--" * 20)  # Print a line of dashes for separation
    print("We are Aggregating duplicate option entries")    
    option_df=date_group.apply(lambda x: remove_duplicates(x))
    option_df=option_df.reset_index(drop=True)
    option_df['option_right'] = option_df['option_right'].replace({'call parity': 'c'})
    ###
    option_df["moneyness"]=option_df["strike"]/option_df['stock_price']-1
    
    ####keep only neccessary columns
    option_df=option_df[["date","exdate","maturity","rounded_maturity","risk_free_rate","forward_risk_free_rate","strike","best_bid","mid_price","best_offer","mid_iv","underlying_price","stock_price",
                        "volume","offer_size","bid_size","option_right","moneyness","qoute_dt","expiry_dt"]]
    option_df=option_df.reset_index(drop=True)
    ##Computes BS IV

    
    return option_df
def clean_optionmetrics(option_df,stock_df,spot_curve_df,forward_curve_df):
    """
    Description of function: To clean other option_metrics data. Coming soon.

    Parameters
    ----------
    option_df : TYPE
        DESCRIPTION.
    stock_df : TYPE
        DESCRIPTION.
    yield_curve_df : TYPE
        DESCRIPTION.

    Yields
    ------
    option_df : TYPE
        DESCRIPTION.

    """
    header_list=option_df.columns.tolist()
    try:
        option_df=option_df[["date","exdate","best_offer","best_bid","strike_price","cp_flag","volume"]]
    except Exception as e:
        print("Missing information. Check file has columns labeled,date,exdate,cp_flag,best_bid,best_offer")
    
    ###Renaming Columns
    option_df.columns=["date","exdate","best_offer","best_bid","strike","option_right","volume"]
    
    ##Quotes
    option_df['qoute_dt'] =pd.to_datetime(option_df['date'])
    option_df['expiry_dt'] =pd.to_datetime(option_df['exdate'])
    option_df = option_df.merge(stock_df, how='left', left_on='qoute_dt', right_on='Date_column')
    option_df = option_df.rename(columns={'price': 'stock_price'})
    option_df = option_df.sort_values(by=['qoute_dt', 'expiry_dt', 'strike'], ascending=[True, True, True])

    ###Strike
    option_df["strike"]=option_df["strike"]/1000
    #### Converting "calls" and "puts" to "c" and "p" respectively.
    option_df['option_right'] = option_df['option_right'].replace({'C': 'c', 'P': 'p'})

    ###Creating maturity in year fraction. For using black scholes formula I replace 0 with 0.001 for numerical reasons.
    option_df['maturity']= ((pd.to_datetime(option_df['exdate']) - pd.to_datetime(option_df['date'])).dt.days)/365
    option_df['maturity'] =option_df['maturity'].replace(0, 0.001)
    option_df['maturity']=option_df['maturity']+(15/24)/365 #expires 10pm, qouted 2am.
    option_df['rounded_maturity']= ((pd.to_datetime(option_df['exdate']) - pd.to_datetime(option_df['date'])).dt.days)/365

    ###Creating mid price. Removing any instance in which the mid price is negative
    option_df['mid_price']=(option_df['best_bid']+option_df['best_offer'])/2
    option_df=option_df[option_df['mid_price']>0]
    
    ##Creating Log-moneyness column
    option_df.loc[option_df['option_right']=='p','log moneyness'] = np.log(option_df['strike']/option_df['stock_price'])
    option_df.loc[option_df['option_right']=='c','log moneyness'] = np.log(option_df['stock_price']/option_df['strike'])
    #option_df=option_df[option_df["mid_price"]>0.5]
    ###
    
    
    option_df=pasting_yields_option_df(option_df, spot_curve_df, "risk_free_rate")
    option_df=pasting_yields_option_df(option_df, forward_curve_df, "forward_risk_free_rate")
    ##
    ##merging yield curve onto main option dataframe
    # option_df = option_df.merge(yield_curve_df, how='left', left_on='qoute_dt', right_on='Date')
    # option_df.drop(columns=['Date'], inplace=True)
    # date_group=option_df.groupby(['date'])
    # yield_curve_headers=yield_curve_df.columns.tolist()[1:]
    
    # print("--" * 20)  # Print a line of dashes for separation
    # print("We are interpolating the yield curve to match your options. This may take some time.")
    # ###Performing Yield Curve interpolation
    # option_df=option_df.reset_index(drop=True)
    # option_df=date_group.apply(lambda x: paste_rate(x,yield_curve_headers)) 
    # print("We are done interpolating Yield curve")
    # option_df=option_df.reset_index(drop=True)
    
    print("--" * 20)  # Print a line of dashes for separation
    print("Converting puts to calls. If both are present we use only OTM options.")
    date_group=option_df.groupby(['date','exdate'])
    option_df=date_group.apply(lambda x: use_only_OTM(x))
    option_df=option_df.reset_index(drop=True)
    
    date_group=option_df.groupby(['date','exdate'])
    print("--" * 20)  # Print a line of dashes for separation
    print("We are Aggregating duplicate option entries")    
    option_df=date_group.apply(lambda x: remove_duplicates(x))
    option_df=option_df.reset_index(drop=True)
    option_df['option_right'] = option_df['option_right'].replace({'call parity': 'c'})  
    
    ###Repair arbitrage
    option_df["underlying_price"]=option_df["stock_price"]
    option_df=option_df.reset_index(drop=True)
    date_group=option_df.groupby(['date','exdate'])
    option_df["repair_price"]=0
    option_df=date_group.apply(lambda x: pre_arbitrage_repair(x))
    ##
    
    
    mid_iv=py_vollib.black.implied_volatility.implied_volatility(option_df["mid_price"], 
                                                                 option_df["stock_price"], option_df["strike"], option_df["risk_free_rate"], option_df["maturity"], 'c', return_as='numpy')
    option_df["mid_iv"]=mid_iv
    option_df=option_df.reset_index(drop=True)
    option_df["bid_size"]=option_df["volume"]
    option_df["offer_size"]=option_df["volume"]
    option_df["moneyness"]=option_df["strike"]/option_df['stock_price']-1
    option_df=option_df[["date","exdate","maturity","rounded_maturity","risk_free_rate","forward_risk_free_rate","strike","best_bid","mid_price","best_offer","mid_iv","underlying_price","stock_price",
                        "volume","offer_size","bid_size","option_right","moneyness","qoute_dt","expiry_dt"]]
    
    return option_df


def use_only_OTM(option_df):
    """
    Description of function: Converts puts into calls via put call parity, and replaces puts with calls.

    Parameters
    ----------
    option_df : dataframe
        option dataframe (day-expiration slice).

    Returns
    -------
    option_df : dataframe
        Dataframe with puts replaced with calls.

    """
    option_df=option_df.reset_index(drop=True)

    contains_only_c = all(x == 'c' for x in option_df['option_right'].to_list())
    contains_only_p = all(x == 'p' for x in option_df['option_right'].to_list())

    if contains_only_c==True:
        #print("Keeping all Calls.")
        return option_df
    if contains_only_p==True:
        #print("Converting puts into calls via put call parity")
        condition = (option_df['option_right'] == "p")
        option_df.loc[condition,'mid_price']=option_df["mid_price"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        option_df.loc[condition,'best_bid']=option_df["best_bid"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        option_df.loc[condition,'best_offer']=option_df["best_offer"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        
        option_df.loc[condition,'option_right']='c'
    if ((contains_only_c==False) & (contains_only_p==False)):
        #print("Replacing ITM calls with puts via put call parity")
        condition = (option_df['option_right'] == "p") & (option_df['strike'] < option_df['stock_price']) ##otm puts
        option_df.loc[condition,'mid_price']=option_df["mid_price"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"] ###convert puts into calls
        option_df.loc[condition,'best_bid']=option_df["best_bid"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        option_df.loc[condition,'best_offer']=option_df["best_offer"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        
        option_df.loc[condition,'option_right']='call parity' #otm puts relabedl as call parity
        option_df = option_df[~(option_df['option_right'] == 'p')] #ITM puts removed
        condition=(option_df['option_right']=='c') & (option_df['strike']<option_df['stock_price']) ##ITM calls
        option_df=option_df.loc[~condition] ##ITM calls removed
        option_df.loc[condition,'option_right']='call parity' #otm puts relabedl as call parity

    return option_df


def remove_duplicates(df):
    """
    Helper function that removes prices in the option dataframe. Prices should be monotonically decreasing/decreasing with respect to strike. In deribit data sometimes option prices will have the same price
    even as the strike prices increases/decreases. This usually only happens at the end of the strike domain. I truncate the data at the first occurence of the price repeating.

    Parameters
    ----------
    df : Dataframe
        Option dataframe. Should be a Date-exdate slice.

    Returns
    -------
    df : Dataframe
        Original Dataframe bu with mid prices truncated.

    """
    df=df.reset_index(drop=True)
    df = df[~df.duplicated(subset=['mid_price'], keep='first')]
    strike_list=df.loc[:,'strike'].values
    
    unique_values, counts = np.unique(strike_list, return_counts=True)
    has_multiple_duplicates = np.any(counts > 1)

    if has_multiple_duplicates:
        def custom_agg_func(series):
            """
            Helper function that handles DF where there are duplicate strikes by averaging.
    
            Parameters
            ----------
            series : TYPE
                DESCRIPTION.
    
            Returns
            -------
            TYPE
                DESCRIPTION.
    
            """
            if pd.api.types.is_numeric_dtype(series):
                return series.mean()
            else:
                return series.iloc[0]
        
        # Apply the custom aggregation function
        df = df.groupby('strike', as_index=False).agg(custom_agg_func)
    #print('working remove')
    return df
def average_futures(df):
    """
    Helper function that averages the futures prices on a given day, for each maturity. This is a known issue with deribit, where for some reason they quote multiple futures prices for a single maturity.
    I take the mean of these and use it as the underlying.

    Parameters
    ----------
    df : Dataframe date-expiry slice
        Dataframe with date-expiry slice.

    Returns
    -------
    df : Dataframe
        Original dataframe with futures price averaged.

    """
    df=df.reset_index(drop=True)
    mean_underlying_future=np.mean(df['underlying_price'].values)
    df.loc[:,'underlying_price']=mean_underlying_future
    return df


def rolling_window(df,nearest_maturity):
    """
    Parameters
    ----------
    df : Date-slice expiration dataframe
        Date_group:
    nearest_maturity : float
        Keeps only options with maturity closest to this number.

    Returns
    -------
    df : dataframe
        contains only options with maturies nearest to the nearest maturity.

    """
    
    df=df.copy()
    unique_maturity_list = df["rounded_maturity"].unique()*365
    closest_maturity = round(unique_maturity_list[np.abs(unique_maturity_list - nearest_maturity).argmin()])/365
    
    df=df[df["rounded_maturity"]==closest_maturity]
    
    return df

def truncate_pdf_iteratively(x_axis, y_axis, stock_price, cdf_threshold=0.99):
    """
    Iteratively truncates the PDF starting from the given stock price. The truncation stops after 10 seconds.

    Parameters:
    x_axis (array-like): Array of X-axis values (e.g., strikes).
    y_axis (array-like): Array of Y-axis values (probabilities).
    stock_price (float): The central value (e.g., current trading price) to start truncating around.
    cdf_threshold (float): CDF threshold for truncation (default is 0.99 for 99%).

    Returns:
    tuple: Truncated x_axis and y_axis arrays around the stock price up to the specified CDF threshold.
    """
    # Create a DataFrame for easy manipulation
    pdf_df = pd.DataFrame({"x": x_axis, "y": y_axis})
    
    # Calculate interval widths (difference between consecutive x values)
    pdf_df['width'] = pdf_df['x'].diff().fillna(0)
    
    # Find the index closest to the stock price
    stock_index = (pdf_df['x'] - stock_price).abs().idxmin()
    
    # Initialize CDF and start iterating from the stock price outward
    cumulative_mass = 0.0
    total_mass = (pdf_df['y'] * pdf_df['width']).sum()
    threshold_mass = cdf_threshold * total_mass
    
    # Start accumulating CDF in both directions from the stock_index
    left_index = stock_index
    right_index = stock_index
    
    start_time = time.time()
    while cumulative_mass < threshold_mass:
        # Expand to the left if possible
        if left_index > 0:
            left_index -= 1
            cumulative_mass += pdf_df.loc[left_index, 'y'] * pdf_df.loc[left_index, 'width']
        
        # Expand to the right if possible
        if right_index < len(pdf_df) - 1 and cumulative_mass < threshold_mass:
            right_index += 1
            cumulative_mass += pdf_df.loc[right_index, 'y'] * pdf_df.loc[right_index, 'width']
        
        if time.time() - start_time > 10:
            print("Stopping pdf truncation because 10 seconds have elapsed.")
            break
    # Truncate the DataFrame within the computed range and return truncated arrays
    truncated_pdf_df = pdf_df.loc[left_index:right_index]
    return truncated_pdf_df['x'].values, truncated_pdf_df['y'].values

def normalize_pdf(strike,pdf,stock_price=True):
    """
    Description function: Normalizes PDF to add to 1

    Parameters
    ----------
    strike : numpy array
        array containing strikes.
    pdf : numpy array
        array containing pdf.

    Returns
    -------
    pdf : array
        pdf normalied to 1.

    """
    # if stock_price!=False:
    #     x_axis=strike/stock_price-1
    # else:
    #     x_axis=strike
    # pdf = np.clip(pdf, a_min=0, a_max=None)
    # bin_width =x_axis[1,]-x_axis[0,]  # Assuming uniform bin width
    # normalized_pdf=pdf/(np.sum(pdf)*bin_width)
    
    #####    
    
    strike=strike.reshape(-1)
    pdf=pdf.reshape(-1)
    pdf = np.clip(pdf, a_min=0, a_max=None)    
    integral = np.trapz(pdf, strike)
    
    normalized_pdf=pdf/integral
    ###Does not normalize
    return normalized_pdf
    
def arbitrage_repair(iv_curve,iv_strike,option_market):
    """
    Description of function: Wrapper function for the arbitrage repair package. I convert IV to call prices, 
    remove arbitrage with L1 repair (does not use bid/ask prices), then convert back to IV.

    Parameters
    ----------
    iv_curve : numpy array
        iv curve.
    iv_strike : numpy array
        array containing stirke.
    option_market : dataframe
        dataframing containing option data.

    Returns
    -------
    iv_curve : numpy array
        iv_curve with options removed..

    """
    R=option_market["risk_free_rate"].values[0]
    T=option_market["maturity"].values[0]
    strike=iv_strike
    underlying_price=option_market["underlying_price"].values[0]
    dense_calls=py_vollib.black_scholes.black_scholes('c', underlying_price, iv_strike, T, R,iv_curve).values.reshape(-1)
    
    c_fv=np.exp(R*T)*dense_calls
    F=np.array([option_market["underlying_price"].values[0]*np.exp(R*T) for i in range(0,len(dense_calls))])
    Tau_vect=np.array([T for i in range(0,len(dense_calls))])
    
    warnings.filterwarnings("ignore")

    normaliser = constraints.Normalise()
    normaliser.fit(Tau_vect, strike, c_fv, F)
    
    T1, K1, C1 = normaliser.transform(Tau_vect, strike, c_fv)
    
    mat_A, vec_b, _, _ = constraints.detect(T1, K1, C1, verbose=False)
    epsilon1 = repair.l1(mat_A, vec_b, C1)
    K0, C0 = normaliser.inverse_transform(K1, C1 + epsilon1)
    
    cleaned_calls=C0*np.exp(-R*T)
    iv = implied_volatility.vectorized_implied_volatility(cleaned_calls,underlying_price, strike,T,R,'c')
    warnings.resetwarnings()
    iv=iv.values.reshape(-1)
    print("cleaned successfully")
    return iv
def pre_arbitrage_repair(option_df):
    """
    Description of function: Wrapper function for the arbitrage repair package. I convert IV to call prices, 
    remove arbitrage with L1 repair (does not use bid/ask prices), then convert back to IV.

    Parameters
    ----------
    iv_curve : numpy array
        iv curve.
    iv_strike : numpy array
        array containing stirke.
    option_market : dataframe
        dataframing containing option data.

    Returns
    -------
    iv_curve : numpy array
        iv_curve with options removed..

    """
    option_df=option_df.copy()
    R=option_df["risk_free_rate"].values[0]
    T=option_df["maturity"].values[0]
    strike=option_df["strike"].values
    underlying_price=option_df["stock_price"].values[0]
    calls=option_df["mid_price"].values
    c_fv=np.exp(R*T)*calls
    F=np.array([underlying_price*np.exp(R*T) for i in range(0,len(option_df["strike"].values))])
    Tau_vect=np.array([T for i in range(0,len(option_df["strike"].values))])
    
    warnings.filterwarnings("ignore")

    normaliser = constraints.Normalise()
    normaliser.fit(Tau_vect, strike, c_fv, F)
    
    T1, K1, C1 = normaliser.transform(Tau_vect, strike, c_fv)
    
    mat_A, vec_b, _, _ = constraints.detect(T1, K1, C1, verbose=False)
    epsilon1 = repair.l1(mat_A, vec_b, C1)
    K0, C0 = normaliser.inverse_transform(K1, C1 + epsilon1)
    
    cleaned_calls=C0*np.exp(-R*T)
    #error=cleaned_calls-C0
    iv = implied_volatility.vectorized_implied_volatility(cleaned_calls,underlying_price, strike,T,R,'c')
    mid_iv=iv.values.reshape(-1,1)
    warnings.resetwarnings()
    option_df["mid_price"]=cleaned_calls
    option_df["mid_iv"]=0
    option_df["mid_iv"]=mid_iv
    print("cleaned successfully")
    return option_df


def numerical_differentiation(strike_vector, points):
    """
    Description of function: Finds the 2nd derivative of IV curve.

    Parameters
    ----------
    strike_vector : numpy array
        Strike array.
    points : numpy array
        IV curve array.

    Returns
    -------
    first_derivative : array
        first derivative.
    second_derivative : array
        second derivative.

    """

    # Ensure that the inputs are numpy arrays
    strike_vector = np.array(strike_vector, dtype=float).reshape(-1)
    points = np.array(points, dtype=float).reshape(-1)
    
    # Calculate the first derivative (dy/dx)
    first_derivative = np.gradient(points, strike_vector)
    
    # Calculate the second derivative (d^2y/dx^2)
    second_derivative = np.gradient(first_derivative, strike_vector)
    
    return first_derivative.reshape(-1), second_derivative.reshape(-1)  



def pasting_yields_option_df(option_df,yields_df,new_column_name):
    """
    Description of function: Pastes interpolated risk free rates from the yields_df onto the options df

    Parameters
    ----------
    option_df : TYPE
        DESCRIPTION.
    yields_df : TYPE
        DESCRIPTION.
    new_column_name : TYPE
        DESCRIPTION.

    Returns
    -------
    option_df : TYPE
        DESCRIPTION.

    """
    # Reshape spot_curves_df to long format
    yields_df_long = yields_df.reset_index().melt(
        id_vars=["index"], var_name="rounded_maturity", value_name=new_column_name
    )
    yields_df_long.rename(columns={"index": "qoute_dt"}, inplace=True)
    
    # Ensure the data types align for merging
    yields_df_long["qoute_dt"] = pd.to_datetime(yields_df_long["qoute_dt"])
    yields_df_long["rounded_maturity"] = yields_df_long["rounded_maturity"].astype(float)/365  # Convert maturity to float for matching
    
    # Assuming 'option_df' is your option dataframe
    option_df["qoute_dt"] = pd.to_datetime(option_df["qoute_dt"])  # Ensure 'date' is datetime
    option_df["maturity"] = option_df["maturity"].astype(float)  # Ensure 'maturity' is float for matching
    
    # Merge the option dataframe with the reshaped spot curves
    option_df = option_df.merge(
        yields_df_long,
        how="left",  # Use 'left' join to keep all rows in the option dataframe
        on=["qoute_dt", "rounded_maturity"]
    )
    return option_df
def polynomial(option_df,argument_dict,interpolate_large = True,extrapolate_curve=True,xlims=(-0.5,0.5)):
    """
    Description of function: This function takes in an option dataframe, which contains the option quotes on a particular day, with a specific maturity.
    It will fit a N order polynomial to this IV curve to smooth out the implied volatilities, then convert them to calls, numerically differentiate twice and save the plots.
    Do not change the inputs of this function, and do not change the output.

    Parameters
    ----------
    option_df : Pandas dataframe containing market information
        Dataframe  containing market information.
    foldername : Str
        Name of the folder created.

    Returns
    -------
    dictionary:
        Key is date and expiry date, value is a pandas dataframe containing the pdf

    """
    #### Some code to get you started.
    order=argument_dict["order"]
    foldername=argument_dict["folder"]
    asset=argument_dict["asset"]
    stock_price=option_df["stock_price"].values[0]
    ##Extract Implied volatility qoutes and strikes
    original_iv=option_df["mid_iv"].values.reshape(-1)
    original_strikes=option_df["strike"].values.reshape(-1)
    date=option_df["date"].values[0]
    exdate=option_df["exdate"].values[0]

    #######TO DO: 
    synthetic_used = False 
    if (interpolate_large):
        strikes, iv,synthetic_strikes,synthetic_iv= interpolate_large_gaps(original_strikes,original_iv)   
        if (len(synthetic_strikes))> 0:
            synthetic_used = True
    
    ###
    # plt.scatter(original_strikes,original_iv)
    # plt.scatter(synthetic_strikes,synthetic_iv)
    
    ### Extrapolation
    extrapolation_used=False
    if (extrapolate_curve):
        combined_strikes,combined_iv,extrapolated_strikes,extrapolated_iv=tail_extrapolation(strikes,iv)
        if (len(extrapolated_strikes))>0:
            extrapolation_used=True
    ##
    # plt.scatter(combined_strikes,combined_iv)
    # plt.scatter(extrapolated_strikes,extrapolated_ivs)
    coefficients = np.polyfit(combined_strikes.reshape(-1), combined_iv, deg=order)

    ###
    ##Define interpolation grid
    interpolated_strikes=np.linspace(combined_strikes[0], combined_strikes[-1],2000)
    # Evaluate the polynomial at the interpolated strike prices
    interpolated_iv = np.polyval(coefficients, interpolated_strikes)
    
    ##
    # plt.scatter(combined_strikes,combined_iv)
    # plt.plot(interpolated_strikes,interpolated_iv)

    
    ##Convert implied volatilities back to call prices
    R=option_df["risk_free_rate"].values[0] ##Risk free rate
    T=option_df["maturity"].values[0]   ##Option maturity
    underlying_price=option_df["underlying_price"].values[0] ##price of the underlying
    
    
    ##Black scholes formula to convert IV to prices
    interpolated_prices = py_vollib.black_scholes.black_scholes('c', underlying_price,interpolated_strikes, T, R,interpolated_iv).values.reshape(-1)
    
    ##visualize calls
    #plt.plot(interpolated_strikes,interpolated_prices)
    
    ##Numerically differentiate
    CDF,PDF=numerical_differentiation(interpolated_strikes, interpolated_prices)

  
    ##Normalize PDF (making sure it integrates to 1, clip any negative values to zero)
    PDF=normalize_pdf(interpolated_strikes,PDF)
    ##visualize PDF and CDF
    # plt.plot(interpolated_strikes,PDF)
    # plt.plot(interpolated_strikes,CDF)


    
    interpolated_strikes = interpolated_strikes[2:-2]
    interpolated_iv = interpolated_iv[2:-2]

    interpolated_prices = interpolated_prices[2:-2]
    CDF = CDF[2:-2]
    PDF = PDF[2:-2]
    
    return_axis,return_pdf=price_to_return_pdf(interpolated_strikes, PDF, stock_price)

    
    
    ##Extracting date


    ##Save plots in a folder in the user directory
    current_directory = os.getcwd()
    figure1_folder = os.path.join(current_directory, foldername)
    if not os.path.exists(figure1_folder):
        os.makedirs(figure1_folder)
        print(f"Created folder: {figure1_folder}")
        
        
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    ax1.scatter(original_strikes/stock_price-1, original_iv,color="red")
    if extrapolation_used:
        ax1.scatter(extrapolated_strikes/stock_price-1,extrapolated_iv,label="Extrapolated points")
    # Add synthetic points if interpolate_large is True
    if synthetic_used:
        ax1.scatter(synthetic_strikes/stock_price-1, synthetic_iv, color="blue", label="Synthetic Points")
    ax1.scatter(strikes/stock_price-1, iv,color="red",label="Original Points")
    ax1.plot(interpolated_strikes/stock_price-1, interpolated_iv)
    ax1.set_title(f"Order {order} polynomial interpolation on {date} to {exdate} {asset}")
    ax1.set_xlabel("Strikes")
    ax1.set_ylabel("Implied Volatility")
    ax1.legend()
    ax1.set_xlim(xlims[0],xlims[1])

    ax3.plot(interpolated_strikes,CDF)
    ax3.set_title("Option implied CDF")
    ax3.set_xlabel("Strikes")
    ax3.set_ylabel("CDF")
    
    ax2.plot(return_axis, return_pdf)
    ax2.set_title("Option implied PDF")
    ax2.set_xlabel("Strikes")
    ax2.set_ylabel("PDF")
    ax2.set_xlim(xlims[0],xlims[1])

    # Show the combined plot

    titlename_1=f"{date} on {exdate}"
    save_path = os.path.join(figure1_folder, titlename_1)
    plt.savefig(save_path)
    plt.close()

    ##final packaging
    density_df=pd.DataFrame({"strike":interpolated_strikes.reshape(-1),"PDF":PDF.reshape(-1)})
    iv_df=pd.DataFrame({"strike":interpolated_strikes.reshape(-1), "IV":interpolated_iv.reshape(-1)})
    dictionary_df={"PDF":density_df,"IV":iv_df}
    return {date +","+ exdate: dictionary_df}

def interpolate_large_gaps(strikes,iv,acceptable_number=3,large_gap_number=5):
    """
    This function is designed to handle the unusual behavior when we try to fit a curve to a function that is discontious. When there is a large gap between points,
    the fitted curve will do strange behavior. This function will first 
    (1) Calculate the average distance between strikes
    (2) If the gap between two consecutive IV points is more than say 5 times the average, linearly interpolate between those points until the average distance
    between those points less than 3 times the average distance.
    (3) Use numpy.interp() to interpolate between points.
    
    Take the Union of the original IVs and their respective strikes, with the interpolated strikes. 
    
    Parameters
    ----------
    strikes : N, np array
        DESCRIPTION.
    iv : N, np array
        DESCRIPTION.

    Returns
    Combined strikes: N, np array
    Combined IV: N,np array
    
    synthetic IV, N, Np array
    synthetic strike, N np array

    """

    ###Program goes here
    average_step=np.mean(np.diff(strikes.reshape(-1)))
    # Threshold for detecting large gaps
    large_gap_threshold = large_gap_number * average_step
    acceptable_gap = acceptable_number* average_step

    # Synthetic data to fill large gaps
    synthetic_strikes = []

    for i in range(len(strikes) - 1):
        gap = strikes[i + 1] - strikes[i]
        if gap > large_gap_threshold:
            # Linearly interpolate new strike prices within the gap
            new_points = np.linspace(strikes[i], strikes[i + 1], int(gap // acceptable_gap)+1 )[1:-1]
            new_points = np.round(new_points).astype(int)
            synthetic_strikes.extend(new_points)
    
     # Convert synthetic strikes to numpy array
    synthetic_strikes = np.array(synthetic_strikes)


    # Interpolate implied volatilities for synthetic strikes
    synthetic_iv = np.interp(synthetic_strikes, strikes, iv)
    
    # Combine original and synthetic data
    combined_strikes = np.concatenate((strikes, synthetic_strikes))
    combined_iv = np.concatenate((iv, synthetic_iv))


 
    # Sort combined data
    sorted_indices = np.argsort(combined_strikes)
    combined_strikes = combined_strikes[sorted_indices]
    combined_iv = combined_iv[sorted_indices]
    
    # plt.scatter(synthetic_strikes,synthetic_iv)
    # plt.scatter(strikes,iv)
    return combined_strikes,combined_iv,synthetic_strikes,synthetic_iv
def tail_extrapolation(strikes,iv):
    """
    This function performs fits a linear regression to the first 3 and last 3 IV qoutes. Then,generates synthetic IV qoutes at stepsize equal to the average
    difference between strikes.
    
    

    Parameters
    ----------
    strike : TYPE
        DESCRIPTION.
    iv : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ##Regression


    strikes=strikes.reshape(-1)
    iv=iv.reshape(-1,1)
    right_lm= sm.OLS(iv[-3:], strikes[-3:]).fit()
    
    iv_last_3 = iv[-3:]
    strikes_last_3 = strikes[-3:]
    distance=strikes[-1]-strikes[-3]
    average_step=np.mean(np.diff(strikes.reshape(-1)))

    if average_step>distance:
        average_step=distance
    # Add a constant for the intercept
    strikes_with_constant = sm.add_constant(strikes_last_3)  # Adds a column of ones
    
    # Fit the model
    right_lm = sm.OLS(iv_last_3, strikes_with_constant).fit()
    ##btc did 13 *average step. Gold did 13
    extrapolation_strikes=np.arange(strikes[-1]+average_step,strikes[-1]+5*average_step,average_step)
    extrapolation_strikes_constant=sm.add_constant(extrapolation_strikes,has_constant='add')
    synthetic_ivs=right_lm.predict(extrapolation_strikes_constant)
    # plt.scatter(extrapolation_strikes,synthetic_ivs)
    # plt.scatter(strikes,iv)
    # plt.show()
    #####left side
    iv_first_3 = iv[0:3]
    strikes_first_3 = strikes[0:3]
    distance=strikes[1]-strikes[0]
    average_step=np.average(np.diff(strikes.reshape(-1)))

    # if average_step>distance:
    #     average_step=distance
    if strikes[0]-average_step<0.1:
        average_step=strikes[0]/3
    strikes_with_constant = sm.add_constant(strikes_first_3)  # Adds a column of ones
    ##used average_step/3 for btc, 13
    left_lm = sm.OLS(iv_first_3, strikes_with_constant).fit()
    extrapolation_strikes_left=np.arange(max(strikes[0]-5*average_step,0.5),strikes[0],average_step)
    extrapolation_strikes_left = extrapolation_strikes_left.reshape(-1)
    extrapolation_strikes_constant_left = sm.add_constant(extrapolation_strikes_left, has_constant='add')
    synthetic_ivs_left=left_lm.predict(extrapolation_strikes_constant_left)
    # plt.scatter(extrapolation_strikes_left,synthetic_ivs_left)
    # plt.scatter(strikes,iv)
    # plt.show()
    #####
    # Combine into a single dataset
    combined_strikes = np.concatenate((strikes, extrapolation_strikes,extrapolation_strikes_left))
    combined_ivs = np.concatenate((iv.reshape(-1), synthetic_ivs,synthetic_ivs_left))

    # Create a DataFrame for easier sorting
    data = pd.DataFrame({'strikes': combined_strikes, 'ivs': combined_ivs})
    
    # Sort the data by strikes
    data_sorted = data.sort_values(by='strikes').reset_index(drop=True)
    #plt.scatter(data_sorted["strikes"],data_sorted["ivs"])
    
    new_strikes=data_sorted["strikes"].values.reshape(-1,1)
    new_ivs=data_sorted["ivs"].values.reshape(-1)
    
    ###
    extrapolation_strikes=np.concatenate((extrapolation_strikes_left,extrapolation_strikes))
    synthetic_ivs=np.concatenate((synthetic_ivs_left,synthetic_ivs))
    # plt.scatter(extrapolation_strikes,synthetic_ivs)
    # plt.scatter(strikes,iv)
    return new_strikes,new_ivs,extrapolation_strikes, synthetic_ivs

def kernel_ridge_pdf(option_df,argument_dict,interpolate_large=True,plotting=True,extrapolate_curve=True,xlims=(-0.5,0.5)):
    ##Misc information
    option_df=option_df.dropna()
    stock_price=option_df["stock_price"].values[0]
    R=option_df["risk_free_rate"].values[0]
    Rf=option_df["forward_risk_free_rate"].values[0]
    maturity=option_df["maturity"].values[0]
    ##date information
    date=option_df["date"].values[0]
    exdate=option_df["exdate"].values[0]
    
    #Extract strikes and IV
    original_iv=option_df["mid_iv"].values.reshape(-1)
    original_strikes=option_df["strike"].values.reshape(-1)

    #parse arguments
    cv=argument_dict["cv"]
    if cv=="loo":
        cv=LeaveOneOut()
    foldername=argument_dict["folder"]
    asset_name=argument_dict["asset"]
    KDE_name=argument_dict["KDE"]
    #cleaning IVs
    #original_iv=arbitrage_repair(original_iv.reshape(-1), original_strikes.reshape(-1), option_df)
    

    #######TO DO: 
    synthetic_used = False 
    if (interpolate_large):
        strikes, iv,synthetic_strikes,synthetic_ivs= interpolate_large_gaps(original_strikes.reshape(-1),original_iv.reshape(-1),2,2.5)   
        if (len(synthetic_strikes))> 0:
            synthetic_used = True
   
    # plt.scatter(original_strikes,original_iv)
    # plt.scatter(synthetic_strikes,synthetic_ivs)
    # plt.show()
    extrapolation_used=False
    if (extrapolate_curve):
        strikes,iv,extrapolated_strikes,extrapolated_iv=tail_extrapolation(strikes, iv)
        if (len(extrapolated_strikes))>0:
            extrapolation_used = True
    
    # plt.scatter(original_strikes,original_iv)
    # plt.scatter(extrapolated_strikes,extrapolated_iv)
    # plt.show()
    # Generate test data for interpolation
    ##5000 and 10000
    interpolated_strikes = np.linspace(min(strikes), max(strikes), 3000).reshape(-1, 1)  # Fine grid of strikes for smooth curve
    #interpolated_strikes = np.linspace(0.5, max(strikes)+stock_price*1.4, 2000).reshape(-1, 1)  # Fine grid of strikes for smooth curve
    strikes=strikes.reshape(-1,1)
    scaler = StandardScaler()
    strikes_scaled = scaler.fit_transform(strikes)  # Fit-transform training data
    interpolated_strikes_scaled = scaler.transform(interpolated_strikes)  # Transform test data only

    # Generate Kernel Ridge Object and train model on IV
    kr_rbf = KernelRidge(kernel="rbf")
    ##20/20
    param_grid = {
        "alpha": np.logspace(-10, 1,10),  # Regularization parameter
        "gamma": np.logspace(-0.75,0.75, 10),  # RBF kernel parameter
    }
    grid_search = GridSearchCV(kr_rbf, param_grid, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(strikes_scaled, iv)
    best_alpha = grid_search.best_params_["alpha"]
    best_gamma = grid_search.best_params_["gamma"]
    kr_best = KernelRidge(alpha=best_alpha, kernel="rbf", gamma=best_gamma)
    kr_best.fit(strikes_scaled, iv)
    
    ###Predictinv IV and cleaning
    interpolated_iv = kr_best.predict(interpolated_strikes_scaled)
    #interpolated_iv=arbitrage_repair(interpolated_iv.reshape(-1), interpolated_strikes.reshape(-1), option_df)
    #strike_test,iv_pred=tail_linear(strike_test, iv_pred, stock_price)
    # plt.plot(interpolated_strikes, interpolated_iv)
    # plt.scatter(strikes,iv)
    # plt.show()

    #plt.scatter(interpolated_strikes,interpolated_iv)
    interpolated_calls = py_vollib.black_scholes.black_scholes('c', option_df['underlying_price'].values[0],
                                                             interpolated_strikes, maturity, R,interpolated_iv).values.reshape(-1)

    
    ##################################### Deriving option implied pdf (4)
    cdf,pdf=numerical_differentiation(interpolated_strikes.reshape(-1), interpolated_calls)
    pdf=pdf*np.exp(-R*maturity)
    plotting=True
    pdf=pdf[2:-2]
    pdf_strike=interpolated_strikes[2:-2]
    cdf=cdf[2:-2]
    cdf_start=cdf[0]
    cdf_last=cdf[-1]
    
    # plt.plot(pdf_strike,pdf)
    # plt.show()
    pdf=normalize_pdf(pdf_strike, pdf,True)
    
    #######################KDE
    if KDE_name=="ISJ":
        KDE_grid=np.linspace(0, max(interpolated_strikes)+stock_price,4000)
        kde = NaiveKDE(bw=KDE_name).fit(pdf_strike,weights=pdf)
        kde_pdf=kde.evaluate(KDE_grid)
    else:
        KDE_grid=np.linspace(0, max(interpolated_strikes)+stock_price,4000)
        kde = NaiveKDE(bw="scott").fit(pdf_strike,weights=pdf)
        bw=kde.bw/KDE_name
        kde = NaiveKDE(bw=bw).fit(pdf_strike,weights=pdf)
        kde_pdf=kde.evaluate(KDE_grid)
        
        
    #cdf= compute_kde_cdf(KDE_grid,kde_pdf)
    # call_prices=compute_call_prices_from_cdf(KDE_grid, cdf, np.exp(-R*maturity))
    # iv = implied_volatility.vectorized_implied_volatility(call_prices[1000:],option_df['underlying_price'].values[0], KDE_grid[1000:],maturity,R,'c')
    # plt.plot(KDE_grid[1000:],iv)
    # plt.plot(interpolated_strikes,interpolated_iv)

    # plt.show()


    # ###statsmodels
    
    # print("statsmodels2")
    #kde = gaussian_kde(pdf_strike.reshape(-1,), weights=pdf.reshape(-1,))
    cdf_values =compute_kde_cdf(KDE_grid,kde_pdf) 
    cdf_last = round(cdf_values[-1], 4)
    # Evaluate the density estimate
    #kde_pdf = kde(KDE_grid.reshape(-1,))
    ###
    # ####Get return axis
    kde_return_axis,kde_return_pdf=price_to_return_pdf(KDE_grid, kde_pdf, stock_price)
    return_axis,return_pdf=price_to_return_pdf(pdf_strike, pdf, stock_price)
    ####
    if plotting==True:
        current_directory = os.getcwd()
        figure1_folder = os.path.join(current_directory, foldername)
        if not os.path.exists(figure1_folder):
            os.makedirs(figure1_folder)
            print(f"Created folder: {figure1_folder}")
            
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        ax1.scatter(original_strikes/stock_price-1, original_iv,color="red",label="IV Qoutes")
        if extrapolation_used:
            ax1.scatter(extrapolated_strikes/stock_price-1,extrapolated_iv,color="blue",label="Extrapolated IV")
        if synthetic_used:
            ax1.scatter(synthetic_strikes/stock_price-1, synthetic_ivs, color="orange",label="Synthetic IV")
        ax1.plot(interpolated_strikes/stock_price-1, interpolated_iv,label="Kernel Ridge IV")
        ax1.set_title(f"Kernel Ridge on {date} to {exdate}")
        ax1.set_xlabel("Rekative Return K/S-1")
        ax1.set_ylabel("Implied Volatility")
        ax1.legend()
        ax1.set_title("IV curve")
        ax1.set_xlim(xlims[0], xlims[1])

        

        ax2.plot(return_axis, return_pdf,label="PDF")
        ax2.plot(kde_return_axis, kde_return_pdf,label="KDE PDF")
        ax2.set_xlabel("Relative Return K/S-1")
        ax2.legend()
        ax2.set_title("PDF")
        ax2.set_xlim(xlims[0], xlims[1])

        ax3.plot(KDE_grid,cdf_values)
        ax3.set_title(f"CDF: alpha ={cdf_last}")
        ax3.set_xlabel("Strike")

        fig.suptitle(f"{asset_name} PDF for {date} to {exdate}", fontsize=16)

        # Show the combined plot
        titlename_1=date+"_"+exdate
        save_path = os.path.join(figure1_folder, titlename_1)
        plt.savefig(save_path)
        plt.close()
        
        iv_objects={"name":"kernel ridge","strikes":strikes,"ivs":iv,"object":kr_best}
        density_df=pd.DataFrame({"Return":kde_return_axis.reshape(-1),"KDE":kde_return_pdf.reshape(-1),"maturity":maturity,"risk":R,"forward":Rf})
        pdf_df=pd.DataFrame({"Return":return_axis.reshape(-1),"PDF":return_pdf.reshape(-1),"maturity":maturity,"risk":R,"forward":Rf})
        iv_df=pd.DataFrame({"Return":interpolated_strikes.reshape(-1),"IV":interpolated_iv.reshape(-1),"maturity":maturity,"risk":R,"forward":Rf})
        dictionary={"KDE":density_df,"PDF":pdf_df,"iv_df":iv_df,"iv_object":iv_objects,"KDE_object":kde}
        
    return {date +","+ exdate: dictionary}
def price_to_return_pdf(strikes,pdf,stock_price):
    return_axis=strikes/stock_price-1
    area=np.trapz(pdf.reshape(-1),return_axis.reshape(-1))
    return_pdf=pdf/area
    return return_axis,return_pdf

def SVR_model(option_df,argument_dict,interpolate_large = True):
    """
    Description of function: This function takes in an option dataframe, which contains the option quotes on a particular day, with a specific maturity.
    It will fit a N order polynomial to this IV curve to smooth out the implied volatilities, then convert them to calls, numerically differentiate twice and save the plots.
    Do not change the inputs of this function, and do not change the output.

    Parameters
    ----------
    option_df : Pandas dataframe containing market information
        Dataframe  containing market information.
    foldername : Str
        Name of the folder created.

    Returns
    -------
    dictionary:
        Key is date and expiry date, value is a pandas dataframe containing the pdf

    """
    #### Some code to get you started.

    foldername=argument_dict["folder"]
    cv = argument_dict["cv"]
    
    ##Extract Implied volatility qoutes and strikes
    original_iv=option_df["mid_iv"].values.reshape(-1)
    original_strikes=option_df["strike"].values.reshape(-1)


    #######TO DO: 
    synthetic_used = False 
    if (interpolate_large):
        strikes, iv,synthetic_strikes,synthetic_iv= interpolate_large_gaps(original_strikes,original_iv)   
        if (len(synthetic_strikes))> 0:
            synthetic_used = True

    
    ### Extrapolation
    combined_strikes,combined_iv,extrapolated_strikes,extrapolated_iv=tail_extrapolation(strikes,iv)



    # Scale the features using StandardScaler
    scaler = StandardScaler()
    combined_strikes_scaled = scaler.fit_transform(combined_strikes.reshape(-1,1))  # Fit on training data, transform both train and test
    #combined_iv_scaled = scaler.transform(combined_iv)


    param_dist = {
        'C': np.logspace(-3, 3, num=20, base=10),  # Range from 10^-3 to 10^3
        'gamma': np.logspace(-9, 1, num=20, base=10),  # Range from 10^-3 to 10^1
        'epsilon': np.logspace(-5, 3, num=20, base=10),  # Range from 10^-5 to 10^-1
        'kernel': ['rbf']  # Kernel type remains 'rbf'
    }

    # Initialize SVR model
    svr = SVR()

    # Set up RandomizedSearchCV
    randomized_search = RandomizedSearchCV(estimator=svr, param_distributions=param_dist, n_iter=100, cv=cv, scoring='neg_mean_squared_error', random_state=42)

    # Fit the model
    randomized_search.fit(combined_strikes_scaled, combined_iv)

    # Get the best model
    best_model = randomized_search.best_estimator_
    print("Best model hyperparameters:")
    print(randomized_search.best_params_)
    #pdb.set_trace()
    

    ##Define interpolation grid
    interpolated_strikes=np.linspace(combined_strikes[0], combined_strikes[-1],1000)
    # Evaluate the polynomial at the interpolated strike prices
    interpolated_strikes_scaled = scaler.transform(interpolated_strikes.reshape(-1,1))
    interpolated_iv = best_model.predict(interpolated_strikes_scaled)
    
    
    ##Convert implied volatilities back to call prices
    R=option_df["risk_free_rate"].values[0] ##Risk free rate
    T=option_df["maturity"].values[0]   ##Option maturity
    underlying_price=option_df["underlying_price"].values[0] ##price of the underlying
    
    
    ##Black scholes formula to convert IV to prices
    interpolated_prices = py_vollib.black_scholes.black_scholes('c', underlying_price,interpolated_strikes, T, R,interpolated_iv).values.reshape(-1)
    
    ##visualize calls
    #plt.plot(interpolated_strikes,interpolated_prices)
    
    ##Numerically differentiate
    CDF,PDF=numerical_differentiation(interpolated_strikes, interpolated_prices)

  
    ##Normalize PDF (making sure it integrates to 1, clip any negative values to zero)
    PDF=normalize_pdf(interpolated_strikes,PDF)
    ##visualize PDF and CDF
    # plt.plot(interpolated_strikes,PDF)
    # plt.plot(interpolated_strikes,CDF)


    
    interpolated_strikes = interpolated_strikes[2:-2]
    interpolated_iv = interpolated_iv[2:-2]

    interpolated_prices = interpolated_prices[2:-2]
    CDF = CDF[2:-2]
    PDF = PDF[2:-2]
    
    
    
    ##Extracting date
    date=option_df["date"].values[0]
    exdate=option_df["exdate"].values[0]

    ##Save plots in a folder in the user directory
    current_directory = os.getcwd()
    figure1_folder = os.path.join(current_directory, foldername)
    if not os.path.exists(figure1_folder):
        os.makedirs(figure1_folder)
        print(f"Created folder: {figure1_folder}")
        
        
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 15))
    ax1.scatter(original_strikes, original_iv,color="red")
    ax1.scatter(extrapolated_strikes,extrapolated_iv,label="Extrapolated points")
    # Add synthetic points if interpolate_large is True
    if synthetic_used:
        ax1.scatter(synthetic_strikes, synthetic_iv, color="orange", label="Synthetic Points")
    #ax1.scatter(strikes, iv,color="red",label="Original Points")
    ax1.plot(interpolated_strikes, interpolated_iv)
    ax1.set_title(f"SVR interpolation on {date} to {exdate}")
    ax1.set_xlabel("Strikes")
    ax1.set_ylabel("Implied Volatility")
    ax1.legend()

    ax2.plot(interpolated_strikes,CDF)
    ax2.set_title("Option implied CDF")
    ax2.set_xlabel("Strikes")
    ax2.set_ylabel("CDF")
    
    ax3.plot(interpolated_strikes, PDF)
    ax3.set_title("Option implied PDF")
    ax3.set_xlabel("Strikes")
    ax3.set_ylabel("PDF")
    # Show the combined plot
    titlename_1=f"{date} on {exdate}"
    save_path = os.path.join(figure1_folder, titlename_1)
    plt.savefig(save_path)
    plt.close()

    ##final packaging
    density_df=pd.DataFrame({"strike":interpolated_strikes.reshape(-1),"PDF":PDF.reshape(-1)})
    iv_df=pd.DataFrame({"strike":interpolated_strikes.reshape(-1),"IV":interpolated_iv.reshape(-1)})

    dictionary={"PDF":density_df,"IV":iv_df}

    return {date +","+ exdate: dictionary}
def lowess(option_df,argument_dict,interpolate_large=True,extrapolate_curve=True,xlims=(-0.5,0.5)):
    """
    

    Parameters
    ----------
    option_df : TYPE
        DESCRIPTION.
    argument_dict : TYPE
        DESCRIPTION.
    interpolate_large : TYPE, optional
        DESCRIPTION. The default is True.
    extrapolate_curve : TYPE, optional
        DESCRIPTION. The default is True.
    xlims : TYPE, optional
        DESCRIPTION. The default is (-0.5,0.5).

    Returns
    -------
    None.

    """
    option_df=option_df.dropna()
    stock_price=option_df["stock_price"].values[0]
    R=option_df["risk_free_rate"].values[0]
    Rf=option_df["forward_risk_free_rate"].values[0]
    maturity=option_df["maturity"].values[0]
    ##date information
    date=option_df["date"].values[0]
    exdate=option_df["exdate"].values[0]
    
    #Extract strikes and IV
    original_iv=option_df["mid_iv"].values.reshape(-1)
    original_strikes=option_df["strike"].values.reshape(-1)
    
    
    cv_errors = {}
    cv_type=argument_dict["cv"]
    asset_name=argument_dict["asset"]
    foldername=argument_dict["folder"]
    KDE_name=argument_dict["KDE"]
    
    synthetic_used = False 
    if (interpolate_large):
        strikes, iv,synthetic_strikes,synthetic_ivs= interpolate_large_gaps(original_strikes.reshape(-1),original_iv.reshape(-1),2,2.5)   
        if (len(synthetic_strikes))> 0:
            synthetic_used = True
    if (extrapolate_curve):
        strikes,iv,extrapolated_strikes,extrapolated_iv=tail_extrapolation(strikes, iv)
        if (len(extrapolated_strikes))> 0:
            extrapolate_used = True
            
    strikes=strikes.reshape(-1)
    iv=iv.reshape(-1)        
    #cv_type = 'loo'  # or cv_type = 5
    
    if cv_type == 'loo':
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=cv_type, shuffle=True, random_state=42)
    
    # Define candidate frac values (smoothing parameters for LOWESS) between 0.1 and 1.0
    candidate_fracs = np.linspace(0.01, 0.8, 20)
    
    cv_errors = {}    # To store the average error for each candidate frac
    best_error = np.inf
    best_frac = None
    
    # Loop over each candidate frac value
    for frac in candidate_fracs:
        errors = []
        # Perform CV splits
        for train_idx, test_idx in cv.split(strikes):
            strike_train, strike_test = strikes[train_idx], strikes[test_idx]
            iv_train, iv_test = iv[train_idx], iv[test_idx]
            
            # Use xvals to evaluate LOWESS predictions on the test strikes directly
            # LOWESS returns the smoothed values at the xvals points
            y_pred = sm.nonparametric.lowess(iv_train, strike_train, frac=frac, xvals=strike_test)
            
            if np.any(np.isnan(y_pred)):
                errors.append(np.inf)
            else:
                errors.append(mean_squared_error(iv_test, y_pred))
        
        mean_error = np.mean(errors)
        
        # Optional check: evaluate on a finer grid to ensure we don't get degenerate (zero) estimates
        strike_grid = np.linspace(np.min(strikes), np.max(strikes), 50)
        iv_est = sm.nonparametric.lowess(iv, strikes, frac=frac, xvals=strike_grid)
        if np.any(iv_est == 0):
            print(f"For frac {frac:.3f}, IV estimate on grid contains zero. Skipping this frac.")
            mean_error = np.inf
        
        cv_errors[frac] = mean_error
        if mean_error < best_error:
            best_error = mean_error
            best_frac = frac
    
    print("Best frac value based on cross-validation:", best_frac)
    
    # ---------------------------------------------------
    # 3. Re-estimate the entire IV curve using the best frac
    # ---------------------------------------------------
    # Create a finer strike grid for a smooth final curve
    interpolated_strikes = np.linspace(np.min(strikes), np.max(strikes), 200)
    # Use xvals to directly evaluate the LOWESS estimate on the fine grid
    interpolated_iv = sm.nonparametric.lowess(iv, strikes, frac=best_frac, xvals=interpolated_strikes)
    
    plt.plot(interpolated_strikes,interpolated_iv)
    plt.scatter(strikes,iv)
    plt.show()
    
    interpolated_calls = py_vollib.black_scholes.black_scholes('c', option_df["underlying_price"].values[0],
                                                             interpolated_strikes, maturity, R,interpolated_iv).values.reshape(-1)
    
    
    # plt.plot(interpolated_strikes,interpolated_calls)
    # #plt.scatter(strikes,iv)
    # plt.show()
    
    cdf,pdf=numerical_differentiation(interpolated_strikes.reshape(-1), interpolated_calls)
    pdf=pdf*np.exp(-R*maturity)
    plotting=True
    pdf=pdf[2:-2]
    pdf_strike=interpolated_strikes[2:-2]
    cdf=cdf[2:-2]

    

    pdf=normalize_pdf(pdf_strike, pdf,True)
    # plt.plot(pdf_strike,pdf)
    # plt.show()
    
    if KDE_name=="ISJ":
        KDE_grid=np.linspace(0, max(interpolated_strikes)+stock_price,4000)
        kde = NaiveKDE(bw=KDE_name).fit(pdf_strike,weights=pdf)
        kde_pdf=kde.evaluate(KDE_grid)
    else:
        KDE_grid=np.linspace(0, max(interpolated_strikes)+stock_price,4000)
        kde = NaiveKDE(bw="scott").fit(pdf_strike,weights=pdf)
        bw=kde.bw/KDE_name
        kde = NaiveKDE(bw=bw).fit(pdf_strike,weights=pdf)
        kde_pdf=kde.evaluate(KDE_grid)
    

    kde_pdf = kde(KDE_grid.reshape(-1,))
    cdf_values =compute_kde_cdf(KDE_grid,kde_pdf) 
    cdf_last = round(cdf_values[-1], 4)
    ###
    # ####Get return axis
    kde_return_axis,kde_return_pdf=price_to_return_pdf(KDE_grid, kde_pdf, stock_price)
    return_axis,return_pdf=price_to_return_pdf(pdf_strike, pdf, stock_price)
    ####
    if plotting==True:
        current_directory = os.getcwd()
        figure1_folder = os.path.join(current_directory, foldername)
        if not os.path.exists(figure1_folder):
            os.makedirs(figure1_folder)
            print(f"Created folder: {figure1_folder}")
            
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        ax1.scatter(original_strikes/stock_price-1, original_iv,color="red",label="IV Qoutes")
        if extrapolate_used:
            ax1.scatter(extrapolated_strikes/stock_price-1,extrapolated_iv,color="blue",label="Extrapolated IV")
        if synthetic_used:
            ax1.scatter(synthetic_strikes/stock_price-1, synthetic_ivs, color="orange",label="Synthetic IV")
        ax1.plot(interpolated_strikes/stock_price-1, interpolated_iv,label="Kernel Ridge IV")
        ax1.set_title(f"Kernel Ridge on {date} to {exdate}")
        ax1.set_xlabel("Rekative Return K/S-1")
        ax1.set_ylabel("Implied Volatility")
        ax1.legend()
        ax1.set_title("IV curve")
        ax1.set_xlim(xlims[0], xlims[1])

        #ax2.plot(pdf_strike, pdf)
        #ax2.plot(KDE_grid,kde_pdf,color="red")
        ax2.plot(return_axis, return_pdf,label="PDF")
        ax2.plot(kde_return_axis, kde_return_pdf,label="KDE PDF")
        ax2.set_xlabel("Relative Return K/S-1")
        ax2.set_xlim(xlims[0], xlims[1])
        ax2.legend()
        ax2.set_title(f"PDF with bandwith: {best_frac}")

        ax3.plot(KDE_grid,cdf_values)
        ax3.set_title(f"CDF: alpha ={cdf_last}")
        ax3.set_xlabel("Strike")

        fig.suptitle(f"{asset_name} PDF for {date} to {exdate}", fontsize=16)

        # Show the combined plot
        titlename_1=date+"_"+exdate
        save_path = os.path.join(figure1_folder, titlename_1)
        plt.savefig(save_path)
        plt.close()
        iv_objects={"name":"lowes","frac":best_frac}
        kde_df=pd.DataFrame({"Return":kde_return_axis.reshape(-1),"PDF":kde_return_pdf.reshape(-1),"maturity":maturity,"risk":R,"forward":Rf})
        pdf_df=pd.DataFrame({"Return":pdf_strike.reshape(-1),"PDF":pdf.reshape(-1),"maturity":maturity,"risk":R,"forward":Rf})
        iv_df=pd.DataFrame({"Return":pdf_strike.reshape(-1),"PDF":pdf.reshape(-1),"maturity":maturity,"risk":R,"forward":Rf})
        dictionary={"KDE":kde_df,"PDF":pdf_df,"IV":iv_df,"iv_object":iv_objects,"KDE_object":kde}
        print(f'{date}')
    return {date +","+ exdate: dictionary}
    
def local_polynomial(option_df, argument_dict,interpolate_large=True,extrapolate_curve=True,xlims=(-0.5,0.5)):
    """
    Performs cross validation to select the best bandwidth for KernelReg.

    Parameters:
        x (np.array): 1D array of the independent variable (e.g. strikes).
        y (np.array): 1D array of the dependent variable (e.g. iv).
        candidate_bw (array): Array of candidate bandwidth values.
        cv_type (str): 'kfold' (default) for KFold CV or 'loo' for Leave-One-Out CV.
        n_splits (int): Number of folds for KFold CV (ignored for LOO).

    Returns:
        best_bw (float): The candidate bandwidth with the lowest average MSE.
        cv_errors (dict): Dictionary mapping candidate bandwidths to their average MSE.
    """
    
    option_df=option_df.dropna()
    stock_price=option_df["stock_price"].values[0]
    R=option_df["risk_free_rate"].values[0]
    Rf=option_df["forward_risk_free_rate"].values[0]
    maturity=option_df["maturity"].values[0]
    ##date information
    date=option_df["date"].values[0]
    exdate=option_df["exdate"].values[0]
    
    #Extract strikes and IV
    original_iv=option_df["mid_iv"].values.reshape(-1)
    original_strikes=option_df["strike"].values.reshape(-1)
    
    
    best_bw = None
    best_error = np.inf
    cv_errors = {}
    cv_type=argument_dict["cv"]
    asset_name=argument_dict["asset"]
    foldername=argument_dict["folder"]
    KDE_name=argument_dict["KDE"]
    synthetic_used = False 
    if (interpolate_large):
        strikes, iv,synthetic_strikes,synthetic_ivs= interpolate_large_gaps(original_strikes.reshape(-1),original_iv.reshape(-1),2,2.5)   
        if (len(synthetic_strikes))> 0:
            synthetic_used = True
    if (extrapolate_curve):
        strikes,iv,extrapolated_strikes,extrapolated_iv=tail_extrapolation(strikes, iv)
        if (len(extrapolated_strikes))> 0:
            extrapolate_used = True

    # Choose the cross validation splitter based on cv_type
    if cv_type == 'loo':
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=cv_type, shuffle=True, random_state=42)
    candidate_bw=np.linspace(min(np.diff(original_strikes)),max(np.diff(original_strikes)),50)
    #candidate_bw=np.linspace(np.mean(np.diff(original_strikes)),2*max(np.diff(original_strikes)),50)

    for bw in candidate_bw:
        errors = []
        for train_idx, test_idx in cv.split(strikes):
            # Extract the training and test subsets
            strike_train, strike_test = strikes[train_idx], strikes[test_idx]
            iv_train, iv_test = iv[train_idx], iv[test_idx]
        
            # Reshape so each is (n_samples, 1) rather than 1D or scalar
            strike_train = strike_train.reshape(-1, 1)
            strike_test = strike_test.reshape(-1, 1)
        
            kr = KernelReg(endog=iv_train, exog=strike_train, reg_type="ll", var_type='c', bw=[bw])
        
            # Now strike_test is guaranteed to be 2D
            y_pred, _ = kr.fit(strike_test)
        
            errors.append(mean_squared_error(iv_test, y_pred))
            mean_error = np.mean(errors)
        
            # Checking for overfitting or zero estimates
            strike_grid = np.linspace(strikes[0], strikes[-1], 25).reshape(-1, 1)
            iv_est, _ = kr.fit(strike_grid)
            if np.any(iv_est == 0):
                print("At least one value in iv_est is zero. Discarding this bandwidth.")
                mean_error = np.inf

        cv_errors[bw] = mean_error
        
        if mean_error < best_error:
            best_error = mean_error
            best_bw = bw
    kr_final = KernelReg(endog=iv, exog=strikes, reg_type="ll", var_type='c', bw=[best_bw])
    
    interpolated_strikes = np.linspace(min(strikes), max(strikes),2000).reshape(-1)  
    interpolated_iv, _ = kr_final.fit(interpolated_strikes[:, None])
    # plt.scatter(strikes,iv)
    # plt.scatter(synthetic_strikes,synthetic_ivs)
    # plt.scatter(extrapolated_strikes,extrapolated_iv)
    # plt.plot(interpolated_strikes,interpolated_iv)
    # plt.show()
    
    interpolated_calls = py_vollib.black_scholes.black_scholes('c', option_df["underlying_price"].values[0],
                                                             interpolated_strikes, maturity, R,interpolated_iv).values.reshape(-1)
    
    
    
    
    cdf,pdf=numerical_differentiation(interpolated_strikes.reshape(-1), interpolated_calls)
    pdf=pdf*np.exp(-R*maturity)
    plotting=True
    pdf=pdf[2:-2]
    pdf_strike=interpolated_strikes[2:-2]
    cdf=cdf[2:-2]

    

    pdf=normalize_pdf(pdf_strike, pdf,True)
    
    #######################KDE
    if KDE_name=="ISJ":
        KDE_grid=np.linspace(0, max(interpolated_strikes)+stock_price,4000)
        kde = NaiveKDE(bw=KDE_name).fit(pdf_strike,weights=pdf)
        kde_pdf=kde.evaluate(KDE_grid)
    else:
        KDE_grid=np.linspace(0, max(interpolated_strikes)+stock_price,4000)
        kde = NaiveKDE(bw="scott").fit(pdf_strike,weights=pdf)
        bw=kde.bw/KDE_name
        kde = NaiveKDE(bw=bw).fit(pdf_strike,weights=pdf)
        kde_pdf=kde.evaluate(KDE_grid)
    

    kde_pdf = kde(KDE_grid.reshape(-1,))
    cdf_values =compute_kde_cdf(KDE_grid,kde_pdf) 
    cdf_last = round(cdf_values[-1], 4)
    ###
    # ####Get return axis
    kde_return_axis,kde_return_pdf=price_to_return_pdf(KDE_grid, kde_pdf, stock_price)
    return_axis,return_pdf=price_to_return_pdf(pdf_strike, pdf, stock_price)
    ####
    if plotting==True:
        current_directory = os.getcwd()
        figure1_folder = os.path.join(current_directory, foldername)
        if not os.path.exists(figure1_folder):
            os.makedirs(figure1_folder)
            print(f"Created folder: {figure1_folder}")
            
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        ax1.scatter(original_strikes/stock_price-1, original_iv,color="red",label="IV Qoutes")
        if extrapolate_used:
            ax1.scatter(extrapolated_strikes/stock_price-1,extrapolated_iv,color="blue",label="Extrapolated IV")
        if synthetic_used:
            ax1.scatter(synthetic_strikes/stock_price-1, synthetic_ivs, color="orange",label="Synthetic IV")
        ax1.plot(interpolated_strikes/stock_price-1, interpolated_iv,label="Kernel Ridge IV")
        ax1.set_title(f"Kernel Ridge on {date} to {exdate}")
        ax1.set_xlabel("Rekative Return K/S-1")
        ax1.set_ylabel("Implied Volatility")
        ax1.legend()
        ax1.set_title("IV curve")
        ax1.set_xlim(xlims[0], xlims[1])

        #ax2.plot(pdf_strike, pdf)
        #ax2.plot(KDE_grid,kde_pdf,color="red")
        ax2.plot(return_axis, return_pdf,label="PDF")
        ax2.plot(kde_return_axis, kde_return_pdf,label="KDE PDF")
        ax2.set_xlabel("Relative Return K/S-1")
        ax2.set_xlim(xlims[0], xlims[1])
        ax2.legend()
        ax2.set_title(f"PDF with bandwith: {best_bw}")

        ax3.plot(KDE_grid,cdf_values)
        ax3.set_title(f"CDF: alpha ={cdf_last}")
        ax3.set_xlabel("Strike")

        fig.suptitle(f"{asset_name} PDF for {date} to {exdate}", fontsize=16)

        # Show the combined plot
        titlename_1=date+"_"+exdate
        save_path = os.path.join(figure1_folder, titlename_1)
        plt.savefig(save_path)
        plt.close()
        iv_objects={"name":"local polynomial","bandwidth":best_bw,"object":kr_final}
        kde_df=pd.DataFrame({"Return":kde_return_axis.reshape(-1),"PDF":kde_return_pdf.reshape(-1),"maturity":maturity,"risk":R,"forward":Rf})
        pdf_df=pd.DataFrame({"Return":pdf_strike.reshape(-1),"PDF":pdf.reshape(-1),"maturity":maturity,"risk":R,"forward":Rf})
        iv_df=pd.DataFrame({"Return":interpolated_strikes.reshape(-1),"PDF":interpolated_iv.reshape(-1)})
        dictionary={"KDE":kde_df,"PDF":pdf_df,"IV":iv_df,"iv_object":iv_objects,"KDE_object":kde}
        print(f'{date}')
    return {date +","+ exdate: dictionary}


def find_cdf(pdf,x_values):
    """
    This function takes in the a vector of probabilities, that came from a pdf, and a vector of x_values. Then computes the CDF function.

    Parameters
    ----------
    pdf : TYPE
        DESCRIPTION.
    x_values : TYPE
        DESCRIPTION.

    Returns
    -------
    cdf: array
    The CDF function obtained by integrating the pdf.
    None.

    """
    return None
def SABR_pdf(option_df,argument_dict,xlims=(-0.5,0.5)):
    """
    

    Parameters
    ----------
    option_df : TYPE
        DESCRIPTION.
    argument_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ##Misc information
    foldername=argument_dict["folder"]
    asset_name=argument_dict["asset"]
    option_df=option_df.dropna()
    stock_price=option_df["stock_price"].values[0]
    R=option_df["risk_free_rate"].values[0]
    Rf=option_df["forward_risk_free_rate"].values[0]
    maturity=option_df["maturity"].values[0]
    ##date information
    date=option_df["date"].values[0]
    exdate=option_df["exdate"].values[0]
    forward=stock_price*np.exp(R*maturity)
    #Extract strikes and IV
    original_iv=option_df["mid_iv"].values.reshape(-1)
    original_strikes=option_df["strike"].values.reshape(-1)

    (alpha, beta, rho, nu), opt_result = calibrate_sabr(forward, original_strikes, maturity, original_iv)
    
    interpolated_strikes=np.linspace(max(original_strikes[0]-0.4*stock_price,0.5), original_strikes[-1]+stock_price*5,5000)
    interpolated_iv = np.array([sabr_volatility(forward, K, maturity, alpha, beta, rho, nu) for K in interpolated_strikes])
    
    # plt.plot(interpolated_strikes,interpolated_iv)
    # plt.scatter(original_strikes,original_iv)
    # plt.show()
    
    interpolated_calls = py_vollib.black_scholes.black_scholes('c', option_df['underlying_price'].values[0],
                                                             interpolated_strikes, maturity, R,interpolated_iv).values.reshape(-1)
    
    
    cdf,pdf=numerical_differentiation(interpolated_strikes.reshape(-1), interpolated_calls)
    pdf=pdf*np.exp(-R*maturity)
    
    
    plotting=True
    pdf=pdf[2:-2]
    pdf_strike=interpolated_strikes[2:-2]
    cdf=cdf[2:-2]
    cdf=cdf+1
    # start_cdf=index_closest_to_0p01_after_last_negative(cdf)
    # pdf_strike=pdf_strike[start_cdf:]
    # cdf=cdf[start_cdf:]
    # pdf=pdf[start_cdf:]
    # cdf_start=cdf[0]
    cdf_last=cdf[-1]-cdf[0]
    pdf=normalize_pdf(pdf_strike, pdf,True)
    return_axis,return_pdf=price_to_return_pdf(pdf_strike, pdf, stock_price)

    # plt.plot(return_axis,return_pdf)
    # plt.show()
    plotting=True
    if plotting==True:
        current_directory = os.getcwd()
        figure1_folder = os.path.join(current_directory, foldername)
        if not os.path.exists(figure1_folder):
            os.makedirs(figure1_folder)
            print(f"Created folder: {figure1_folder}")
            
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        ax1.scatter(original_strikes/stock_price-1, original_iv,color="red",label="IV Qoutes")
        #ax1.scatter(extrapolated_strikes/stock_price-1,extrapolated_iv,color="blue",label="Extrapolated IV")

        ax1.plot(interpolated_strikes/stock_price-1, interpolated_iv,label="Kernel Ridge IV")
        ax1.set_title(f"SABR on {date} to {exdate}")
        ax1.set_xlabel("Rekative Return K/S-1")
        ax1.set_ylabel("Implied Volatility")
        ax1.legend()
        ax1.set_title("IV curve")
        ax1.set_xlim(xlims[0], xlims[1])  # Set x-axis limit
        ax1.set_ylim(0,2.5)
        #ax2.plot(pdf_strike, pdf)
        #ax2.plot(KDE_grid,kde_pdf,color="red")
        ax2.plot(return_axis, return_pdf,label="PDF")
        #ax2.plot(kde_return_axis, kde_return_pdf,label="KDE PDF")
        ax2.set_xlim(xlims[0], xlims[1])  # Set x-axis limit
        ax2.set_xlabel("Relative Return K/S-1")
        ax2.legend()
        ax2.set_title("PDF")

        ax3.plot(pdf_strike,cdf)
        ax3.set_title(f"CDF: alpha ={cdf_last}")
        ax3.set_xlabel("Strike")
        ax3.legend()

        fig.suptitle(f"{asset_name} PDF for {date} to {exdate}", fontsize=16)

        # Show the combined plot
        titlename_1=date+"_"+exdate
        save_path = os.path.join(figure1_folder, titlename_1)
        plt.savefig(save_path)
        plt.close()
        density_df=pd.DataFrame({"Return":return_axis.reshape(-1),"PDF":return_pdf.reshape(-1),"maturity":maturity,"risk":R,"forward":Rf})
        iv_df=pd.DataFrame({"Return":interpolated_strikes.reshape(-1),"IV":interpolated_iv.reshape(-1),"maturity":maturity,"risk":R,"forward":Rf})

        dictionary={"PDF":density_df,"IV":iv_df}
        print(f'{date}')
    return {date +","+ exdate: dictionary}

def sabr_volatility(f, K, T, alpha, beta, rho, nu):
    """
    Computes the SABR implied volatility using Hagan's asymptotic formula.
    """
    # Handle ATM case to avoid division by zero in the log term.
    if abs(f - K) < 1e-12:
        term1 = ((1 - beta)**2 / 24) * (alpha**2 / (f**(2 - 2*beta)))
        term2 = (rho * beta * nu * alpha) / (4 * f**(1 - beta))
        term3 = ((2 - 3*rho**2) / 24) * nu**2
        return (alpha / (f**(1 - beta))) * (1 + (term1 + term2 + term3) * T)
    
    log_fk = np.log(f / K)
    fk_beta = (f * K)**((1 - beta) / 2)
    z = (nu / alpha) * fk_beta * log_fk

    # Compute x(z)
    sqrt_term = np.sqrt(1 - 2*rho*z + z**2)
    numerator = sqrt_term + z - rho
    denominator = 1 - rho
    x_z = np.log(numerator / denominator) if numerator > 0 else np.log(1e-12)
    if abs(x_z) < 1e-12:
        x_z = 1e-12

    # Correction terms
    term1 = ((1 - beta)**2 / 24) * (alpha**2 / ((f * K)**(1 - beta)))
    term2 = (rho * beta * nu * alpha) / (4 * fk_beta)
    term3 = ((2 - 3*rho**2) / 24) * nu**2
    correction = 1 + (term1 + term2 + term3) * T

    # Final SABR implied volatility
    sabr_iv = (alpha / fk_beta) * (z / x_z) * correction
    return sabr_iv
def objective(params, f, strikes, T, market_vols):
    """
    Objective function: Sum of squared errors between SABR volatilities and market volatilities.
    """
    alpha, beta, rho, nu = params
    model_vols = np.array([sabr_volatility(f, K, T, alpha, beta, rho, nu) for K in strikes])
    return np.sum((model_vols - market_vols)**2)

def calibrate_sabr(f, strikes, T, market_vols):
    """
    Calibrates SABR parameters (alpha, beta, rho, nu) using least-squares.
    """
    # Initial guesses: [alpha, beta, rho, nu]
    initial_guess = [0.2, 0.5, 0.0, 0.2]
    bounds = [
        (1e-9, None),       # alpha > 0
        (0.0, 1.0),         # beta in [0,1]
        (-0.9999, 0.9999),   # rho in (-1,1)
        (1e-9, None)        # nu > 0
    ]
    result = minimize(
        objective, 
        x0=initial_guess,
        args=(f, strikes, T, market_vols),
        bounds=bounds,
        method='L-BFGS-B'
    )
    alpha, beta, rho, nu = result.x
    return (alpha, beta, rho, nu), result
def index_closest_to_0p01_after_last_negative(arr):
    """
    Returns the index of the element closest to 0.01 after the last negative number in 'arr'.
    
    If there are no negative numbers in 'arr', returns the index of the element
    closest to 0.01 in the entire array.
    
    If there are negative numbers but no elements after the last negative,
    returns -1.
    """
    # Find indices where the array is negative
    negative_indices = np.where(arr < 0)[0]
    
    # If there are no negative numbers, search the entire array
    if len(negative_indices) == 0:
        return np.argmin(np.abs(arr - 0.005))
    
    # Otherwise, find the last negative index
    last_neg_idx = negative_indices[-1]
    
    # Slice the array after the last negative
    subsequent_slice = arr[last_neg_idx + 1:]
    
    # If there's nothing after the last negative, return -1
    if subsequent_slice.size == 0:
        return -1
    
    # Find the index of the element in the slice that is closest to 0.01
    sub_idx = np.argmin(np.abs(subsequent_slice - 0.005))
    return last_neg_idx + 1 + sub_idx
def index_of_smallest(arr):
    """
    Returns the index of the smallest element in the array 'arr'.
    
    Parameters:
    -----------
    arr : list or numpy.ndarray
        The input array containing numbers.
    
    Returns:
    --------
    int
        The index of the smallest element in 'arr'.
    """
    arr = arr[:len(arr) // 2]  # Use integer division to avoid float index issues
    
    #masked_arr = np.where(arr < 0, np.inf, arr)
    
    # Find the index of the smallest non-negative number
    smallest_index = np.argmin(arr)
    print(smallest_index)
    return smallest_index
def compute_kde_cdf(KDE_grid, kde_pdf):
    """
    Compute the CDF from the estimated KDE PDF using numerical integration.
    
    Parameters:
    KDE_grid (numpy array): The grid points where the PDF is evaluated.
    kde_pdf (numpy array): The estimated PDF values at the grid points.
    
    Returns:
    kde_cdf (numpy array): The estimated CDF values at the grid points.
    """
    KDE_grid = np.array(KDE_grid).ravel()
    kde_pdf = np.array(kde_pdf).ravel()
    kde_cdf = np.cumsum(kde_pdf * np.diff(KDE_grid, prepend=KDE_grid[0]))
    #kde_cdf /= kde_cdf[-1]  # Normalize to ensure the CDF reaches 1
    
    
    return kde_cdf

def compute_call_prices_from_cdf(KDE_grid, kde_cdf, discount_factor):
    """
    Compute the entire call price spectrum from the CDF.
    
    For each strike K in KDE_grid, the call price is computed as:
    
        C(K) = discount_factor * ∫[K,∞] (1 - F(x)) dx
    
    Parameters:
        KDE_grid (np.array): 1D array of underlying asset prices (or strikes) where the CDF is evaluated.
        kde_cdf (np.array): 1D array of the CDF values at the grid points.
        discount_factor (float): e^(-rT), the discount factor.
        
    Returns:
        call_prices (np.array): Array of call option prices for each strike in KDE_grid.
    """
    KDE_grid = np.array(KDE_grid).ravel()
    kde_cdf = np.array(kde_cdf).ravel()
    
    # Compute the survival function: 1 - F(x)
    survival = 1 - kde_cdf
    call_prices = np.empty_like(survival)
    
    # For each grid point, compute the integral from that point to the end using np.trapz.
    for i in range(len(KDE_grid)):
        call_prices[i] = discount_factor * np.trapz(survival[i:], KDE_grid[i:])
    
    return call_prices