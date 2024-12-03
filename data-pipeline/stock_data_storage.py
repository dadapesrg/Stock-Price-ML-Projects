from sqlalchemy import create_engine

def load_data_to_db(df, database_url):
    # SQLAlchemy Engine Setup      
        
    engine = create_engine(database_url)
    
    # Insert data into the table
    df.to_sql('stock_prices', con=engine, if_exists="replace", index=False)
