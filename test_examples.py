from main_option_market import*
import pickle

with open("yields_dict.pkl", "rb") as file:
    loaded_dict = pickle.load(file)
    
    
spot=loaded_dict["spot"]  
forward=loaded_dict["forward"]


option_data_prefix="demo_options_dow_etf.csv"
stock_price_data_prefix='dow_etf_stock.csv'

####Initialize Option_market with risk free rate, forward rate, option data, stock data
option_market=OptionMarket(option_data_prefix,stock_price_data_prefix,spot,forward)
option_df=option_market.original_market_df
###Lets look at subset of data: All options from 2022-08-31 to 2022-09-01, selecting the options with a maturity closest to 15 days
option_market.set_master_dataframe(volume=-1, maturity=[0,30],rolling=15,moneyness=[-3.5,3.5],date=["2022-08-31","2022-09-01"])
option_df=option_market.master_dataframe

##Create Kernel ridge argument dictionary
kernel_ridge_arguments={"cv_method":5,"interpolate_large_gap":True,"extrapolate_curve":False,"stock_df":None,"real":False,
               "folder_name":"dow_kernel_ridge","asset":"Chevron","kernel":"rbf",'kde_method':"ISJ","xlims":(-0.9,0.9)}


kernel_ridge_pdf=option_market.estimate_option_pdfs("kernel ridge",kernel_ridge_arguments)

##Create Local Linear Regression argument dictionary

local_reg_arguments={"cv_method":5,"interpolate_large_gap":True,"extrapolate_curve":False,"stock_df":None,"real":False,
               "folder_name":"dow_local_reg","asset":"Chevron","bw_setting":10,"reg_type":"ll",'kde_method':"ISJ","xlims":(-0.9,0.9)}

local_reg_pdf=option_market.estimate_option_pdfs("local_polynomial",local_reg_arguments)


