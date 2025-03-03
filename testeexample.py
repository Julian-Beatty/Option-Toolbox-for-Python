from main_option_market import*
from create_yield_curve import*
###If manually want to run yield curves##
#spot,forward=create_yield_curves("fed_yield_curve.csv","2024-11-01")
###Otherwise just load mine

with open("yields_dict.pkl", "rb") as file:
    loaded_dict = pickle.load(file)
    
    
spot=loaded_dict["spot"]  
forward=loaded_dict["forward"] 
###Datafiles
option_data_prefix="btc_eod"
stock_price_data_prefix='BTC_data_new.csv'


option_data_prefix="optionmetrics_gld.csv"
stock_price_data_prefix='GLD price.csv'
#################Setting up Option_market with Gold options 
####

option_market=OptionMarket(option_data_prefix,stock_price_data_prefix,spot,forward) #Initialization
option_df=option_market.original_market_df #Raw dataframe
option_market.set_master_dataframe(volume=-1, maturity=[0,900],moneyness=[-15.2,15.2],rolling=30,date=["2022-08-31","2022-09-02"]) #Filter options
option_df=option_market.master_dataframe 

###Computation of PDFs





argument_dict={"folder":"GLD 30_SABR","asset":"30 Day GLD SABR"}
test=option_market.estimate_option_pdfs("SABR",argument_dict,xlims=(-1.35,1.35))

argument_dict={"cv":"loo","folder":"GLD 30_KRR","asset":"30 Day GLD KRR","KDE":4}
test=option_market.estimate_option_pdfs("kernel ridge",argument_dict,xlims=(-1.35,1.35))


argument_dict={"cv":"loo","folder":"GLD 30_LP","asset":"30 Day GLD LP","KDE":3}
test=option_market.estimate_option_pdfs("local_polynomial",argument_dict,xlims=(-1.35,1.35))

argument_dict={"cv":"loo","folder":"GLD 30_Lowess","asset":"30 Day GLD","KDE":2}
test=option_market.estimate_option_pdfs("lowess",argument_dict,xlims=(-1.5,1.5))
    
argument_dict={"order":4,"folder":"GLD 30_quartic","asset":"30 Day GLD polynomial"}
test=option_market.estimate_option_pdfs("polynomial",argument_dict,xlims=(-1,1.6))



