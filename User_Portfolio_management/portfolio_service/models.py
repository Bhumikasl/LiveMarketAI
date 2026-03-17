from sqlalchemy import Column, Integer, String, Float
from database import Base

class UserBalance(Base):
    __tablename__ = "user_balance"

    user_id = Column(Integer, primary_key=True)
    available_balance = Column(Float)
    reserved_balance = Column(Float)


class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    stock_symbol = Column(String)
    quantity = Column(Integer)
    avg_price = Column(Float)


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    stock_symbol = Column(String)
    quantity = Column(Integer)
    price = Column(Float)
    order_type = Column(String)