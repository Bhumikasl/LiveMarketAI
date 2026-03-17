from fastapi import FastAPI

app = FastAPI(title="Portfolio Management Service")

# -------------------------------
# In-memory databases (for demo)
# -------------------------------

balances = {
    101: {"available_balance": 5000, "reserved_balance": 0}
}

portfolio = {
    101: [
        {"stock": "AAPL", "quantity": 10, "avg_price": 150},
        {"stock": "TSLA", "quantity": 5, "avg_price": 700}
    ]
}

transactions = []

# Simulated market prices
market_prices = {
    "AAPL": 170,
    "TSLA": 750,
    "GOOG": 2800
}

# -------------------------------
# Root API
# -------------------------------

@app.get("/")
def home():
    return {"message": "Portfolio Service Running"}

# -------------------------------
# Get User Balance
# -------------------------------

@app.get("/balance/{user_id}")
def get_balance(user_id: int):
    return balances.get(user_id, {"error": "User not found"})

# -------------------------------
# Get Portfolio Holdings
# -------------------------------

@app.get("/portfolio/{user_id}")
def get_portfolio(user_id: int):
    return portfolio.get(user_id, [])

# -------------------------------
# Portfolio Valuation
# -------------------------------

@app.get("/portfolio/value/{user_id}")
def portfolio_value(user_id: int):
    holdings = portfolio.get(user_id, [])
    total = 0

    for stock in holdings:
        price = market_prices.get(stock["stock"], 0)
        total += stock["quantity"] * price

    return {
        "user_id": user_id,
        "portfolio_value": total
    }

# -------------------------------
# Reserve Funds for Order
# -------------------------------

@app.post("/reserve-funds/{user_id}/{amount}")
def reserve_funds(user_id: int, amount: float):
    user_balance = balances.get(user_id)

    if not user_balance:
        return {"error": "User not found"}

    if user_balance["available_balance"] < amount:
        return {"error": "Insufficient funds"}

    user_balance["available_balance"] -= amount
    user_balance["reserved_balance"] += amount

    return {"message": "Funds reserved", "balance": user_balance}

# -------------------------------
# Execute Buy Order
# -------------------------------

@app.post("/buy/{user_id}/{stock}/{qty}/{price}")
def buy_stock(user_id: int, stock: str, qty: int, price: float):

    cost = qty * price
    user_balance = balances.get(user_id)

    if user_balance["available_balance"] < cost:
        return {"error": "Insufficient balance"}

    user_balance["available_balance"] -= cost

    holdings = portfolio.setdefault(user_id, [])

    for item in holdings:
        if item["stock"] == stock:
            item["quantity"] += qty
            break
    else:
        holdings.append({
            "stock": stock,
            "quantity": qty,
            "avg_price": price
        })

    txn = {
        "user": user_id,
        "stock": stock,
        "quantity": qty,
        "price": price,
        "type": "BUY"
    }

    transactions.append(txn)

    return {"message": "Stock purchased", "transaction": txn}

# -------------------------------
# Transaction History
# -------------------------------

@app.get("/transactions/{user_id}")
def get_transactions(user_id: int):
    user_txn = [t for t in transactions if t["user"] == user_id]
    return user_txn