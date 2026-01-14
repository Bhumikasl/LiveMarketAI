# Enhanced Financial Analysis Tool - With Hardcoded API Keys
# This version includes hardcoded API keys for easier setup
# Essential libraries for basic financial analysis
!pip install yfinance pandas numpy matplotlib plotly requests

# For advanced plotting (seaborn style)
!pip install seaborn

# For multimodal analysis (YouTube + Audio transcription)
!pip install google-api-python-client yt-dlp openai-whisper

# Alternative whisper installation if the above doesn't work
# !pip install git+https://github.com/openai/whisper.git

# For audio processing (whisper dependency)
!pip install torch torchvision torchaudio

# Optional: For better performance with audio
!apt update &> /dev/null
!apt install ffmpeg &> /dev/null

# Check installations
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
print("✓ Core libraries installed successfully")

try:
    from googleapiclient.discovery import build
    print("✓ Google API client available")
except ImportError:
    print("⚠ Google API client not available")

try:
    import yt_dlp
    print("✓ yt-dlp available")
except ImportError:
    print("⚠ yt-dlp not available")

try:
    import whisper
    print("✓ Whisper available")
except ImportError:
    print("⚠ Whisper not available")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
from datetime import datetime, timedelta
import warnings
import json
import os
from typing import Optional, List, Dict, Any, Tuple, Union

# Optional imports for multimodal analysis (with fallback)
try:
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    print("Warning: Google API client not available. YouTube functionality will be disabled.")

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("Warning: yt-dlp not available. YouTube downloading functionality will be disabled.")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Audio transcription functionality will be disabled.")

warnings.filterwarnings('ignore')

# Configuration
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

# =============================================================================
# HARDCODED API KEYS - REPLACE WITH YOUR ACTUAL KEYS
# =============================================================================
YOUTUBE_API_KEY = "your api"
# You can also add other API keys here if needed in the future:
ALPHA_VANTAGE_API_KEY = "your api"
NEWS_API_KEY = "your api"
# =============================================================================

class FinancialAnalyzer:
    """Comprehensive Financial Analysis Tool"""

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.data = None
        self.analysis_results = {}

    def fetch_data(self, period: str = '1y', start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> bool:
        """Fetch stock data from Yahoo Finance with flexible date options"""
        try:
            print(f"Fetching data for {self.symbol}...")
            ticker = yf.Ticker(self.symbol)

            # Handle different date input methods
            if start_date and end_date:
                # Custom date range
                self.data = ticker.history(start=start_date, end=end_date)
                print(f"Using custom date range: {start_date} to {end_date}")
            elif start_date and not end_date:
                # From start_date to present
                self.data = ticker.history(start=start_date)
                print(f"Using date range: {start_date} to present")
            else:
                # Use period
                self.data = ticker.history(period=period)
                print(f"Using period: {period}")

            if self.data.empty:
                raise Exception("No data retrieved")

            print(f"Retrieved {len(self.data)} days of data")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def calculate_technical_indicators(self) -> None:
        """Calculate various technical indicators"""
        if self.data is None:
            print("No data available")
            return

        try:
            print("Calculating technical indicators...")
            df = self.data.copy()

            # Moving Averages
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']

            # Volatility
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)

            # Support and Resistance (simplified)
            df['Resistance'] = df['High'].rolling(window=20).max()
            df['Support'] = df['Low'].rolling(window=20).min()

            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

            self.data = df
            print("Technical indicators calculated")

        except Exception as e:
            print(f"Error calculating indicators: {e}")

    def generate_signals(self) -> None:
        """Generate trading signals"""
        if self.data is None:
            return

        try:
            print("Generating trading signals...")
            df = self.data.copy()

            # Initialize signals
            df['Signal'] = 0
            df['Position'] = 0

            # Moving Average Crossover
            df['MA_Signal'] = np.where(df['SMA_10'] > df['SMA_50'], 1, -1)

            # RSI Signals
            df['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))

            # MACD Signals
            df['MACD_Signal_Flag'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)

            # Bollinger Band Signals
            df['BB_Signal'] = np.where(df['Close'] < df['BB_Lower'], 1,
                                     np.where(df['Close'] > df['BB_Upper'], -1, 0))

            # Combined Signal (simple majority voting)
            signals = ['MA_Signal', 'RSI_Signal', 'MACD_Signal_Flag', 'BB_Signal']
            df['Combined_Signal'] = df[signals].sum(axis=1)
            df['Final_Signal'] = np.where(df['Combined_Signal'] >= 2, 1,
                                        np.where(df['Combined_Signal'] <= -2, -1, 0))

            self.data = df
            print("Trading signals generated")

        except Exception as e:
            print(f"Error generating signals: {e}")

    def simple_backtest(self, initial_capital: float = 10000) -> None:
        """Simple backtesting"""
        if self.data is None:
            return

        try:
            print("Running backtest...")
            df = self.data.copy()

            # Initialize backtest variables
            capital = initial_capital
            shares = 0
            portfolio_value = []
            trades = []

            for i in range(1, len(df)):
                current_price = df['Close'].iloc[i]
                signal = df['Final_Signal'].iloc[i]
                prev_signal = df['Final_Signal'].iloc[i-1]

                # Buy signal
                if signal == 1 and prev_signal != 1 and shares == 0:
                    shares = capital / current_price
                    capital = 0
                    trades.append(('BUY', df.index[i], current_price, shares))

                # Sell signal
                elif signal == -1 and prev_signal != -1 and shares > 0:
                    capital = shares * current_price
                    trades.append(('SELL', df.index[i], current_price, shares))
                    shares = 0

                # Calculate portfolio value
                total_value = capital + (shares * current_price)
                portfolio_value.append(total_value)

            # Final portfolio value
            final_price = df['Close'].iloc[-1]
            if shares > 0:
                capital = shares * final_price
                shares = 0

            final_value = capital + (shares * final_price)
            total_return = (final_value - initial_capital) / initial_capital

            # Calculate buy and hold return
            buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]

            # Store results
            self.analysis_results['backtest'] = {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': total_return - buy_hold_return,
                'number_of_trades': len(trades),
                'trades': trades
            }

            print(f"Backtest completed:")
            print(f"   • Initial Capital: ${initial_capital:,.2f}")
            print(f"   • Final Value: ${final_value:,.2f}")
            print(f"   • Strategy Return: {total_return:.2%}")
            print(f"   • Buy & Hold Return: {buy_hold_return:.2%}")
            print(f"   • Excess Return: {total_return - buy_hold_return:.2%}")
            print(f"   • Number of Trades: {len(trades)}")

        except Exception as e:
            print(f"Backtest error: {e}")

    def simple_forecast(self, days: int = 30) -> Optional[pd.DataFrame]:
        """Simple price forecast using moving averages and trend"""
        if self.data is None:
            return None

        try:
            print(f"Generating {days}-day forecast...")

            # Calculate trend using linear regression on recent data
            recent_data = self.data['Close'].tail(60).reset_index(drop=True)
            x = np.arange(len(recent_data))
            z = np.polyfit(x, recent_data, 1)

            # Current moving averages
            current_sma20 = self.data['SMA_20'].iloc[-1]
            current_volatility = self.data['Volatility'].iloc[-1]

            # Generate forecast dates
            last_date = self.data.index[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1),
                                         periods=days, freq='D')

            # Simple forecast: trend + moving average
            trend_slope = z[0]
            base_price = self.data['Close'].iloc[-1]

            forecast_prices = []
            for i in range(days):
                # Trend component
                trend_price = base_price + (trend_slope * (i + 1))

                # Add some mean reversion to moving average
                ma_pull = (current_sma20 - trend_price) * 0.1

                # Add volatility (simplified)
                noise = np.random.normal(0, current_volatility * 0.1)

                forecast_price = trend_price + ma_pull + noise
                forecast_prices.append(max(forecast_price, 0))  # Ensure positive price

            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast_prices,
                'Lower_Bound': [p * 0.9 for p in forecast_prices],
                'Upper_Bound': [p * 1.1 for p in forecast_prices]
            })

            self.analysis_results['forecast'] = forecast_df
            print("Forecast generated")
            return forecast_df

        except Exception as e:
            print(f"Forecast error: {e}")
            return None

    def plot_comprehensive_analysis(self) -> None:
        """Create comprehensive analysis plots"""
        if self.data is None:
            return

        try:
            print("Creating comprehensive analysis plots...")

            # Create subplots
            fig = make_subplots(
                rows=5, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=(
                    f'{self.symbol} Price & Moving Averages',
                    'RSI',
                    'MACD',
                    'Bollinger Bands',
                    'Volume & Signals'
                ),
                row_heights=[0.3, 0.15, 0.15, 0.2, 0.2]
            )

            # 1. Price and Moving Averages
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Close'],
                                   name='Close', line=dict(color='blue', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['SMA_10'],
                                   name='SMA 10', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['SMA_50'],
                                   name='SMA 50', line=dict(color='red')), row=1, col=1)

            # 2. RSI
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['RSI'],
                                   name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # 3. MACD
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['MACD'],
                                   name='MACD', line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['MACD_Signal'],
                                   name='Signal', line=dict(color='red')), row=3, col=1)
            colors = ['red' if x < 0 else 'green' for x in self.data['MACD_Histogram']]
            fig.add_trace(go.Bar(x=self.data.index, y=self.data['MACD_Histogram'],
                               name='Histogram', marker_color=colors), row=3, col=1)

            # 4. Bollinger Bands
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Close'],
                                   name='Close', line=dict(color='blue')), row=4, col=1)
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['BB_Upper'],
                                   name='BB Upper', line=dict(color='gray', dash='dash')), row=4, col=1)
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['BB_Lower'],
                                   name='BB Lower', line=dict(color='gray', dash='dash'),
                                   fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=4, col=1)
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['BB_Middle'],
                                   name='BB Middle', line=dict(color='orange')), row=4, col=1)

            # 5. Volume and Signals
            fig.add_trace(go.Bar(x=self.data.index, y=self.data['Volume'],
                               name='Volume', marker_color='lightblue'), row=5, col=1)

            # Add buy/sell signals
            buy_signals = self.data[self.data['Final_Signal'] == 1]
            sell_signals = self.data[self.data['Final_Signal'] == -1]

            if not buy_signals.empty:
                fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                       mode='markers', name='Buy Signal',
                                       marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)

            if not sell_signals.empty:
                fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                       mode='markers', name='Sell Signal',
                                       marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

            fig.update_layout(height=1000, showlegend=True,
                            title=f'{self.symbol} Comprehensive Technical Analysis')
            fig.show()

            print("Comprehensive analysis plot created")

        except Exception as e:
            print(f"Plotting error: {e}")

    def plot_forecast(self) -> None:
        """Plot price forecast"""
        if 'forecast' not in self.analysis_results:
            return

        try:
            forecast = self.analysis_results['forecast']

            fig = go.Figure()

            # Historical data (last 60 days)
            recent_data = self.data.tail(60)
            fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'],
                                   mode='lines', name='Historical Price',
                                   line=dict(color='blue', width=2)))

            # Forecast
            fig.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Forecast'],
                                   mode='lines', name='Forecast',
                                   line=dict(color='red', dash='dash', width=2)))

            # Confidence interval
            fig.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Upper_Bound'],
                                   mode='lines', name='Upper Bound',
                                   line=dict(color='lightgray'), showlegend=False))

            fig.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Lower_Bound'],
                                   mode='lines', name='Lower Bound', fill='tonexty',
                                   fillcolor='rgba(255,0,0,0.1)',
                                   line=dict(color='lightgray'), showlegend=False))

            fig.update_layout(
                title=f'{self.symbol} Price Forecast (30 Days)',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500
            )
            fig.show()

            print("Forecast plot created")

        except Exception as e:
            print(f"Forecast plotting error: {e}")

    def export_results(self) -> None:
        """Export analysis results to CSV"""
        try:
            print("Exporting results...")

            # Export main data with indicators
            self.data.to_csv(f'{self.symbol}_technical_analysis.csv')
            print(f"Technical analysis exported to {self.symbol}_technical_analysis.csv")

            # Export forecast if available
            if 'forecast' in self.analysis_results:
                self.analysis_results['forecast'].to_csv(f'{self.symbol}_forecast.csv', index=False)
                print(f"Forecast exported to {self.symbol}_forecast.csv")

            # Export backtest results
            if 'backtest' in self.analysis_results:
                backtest_summary = pd.DataFrame([self.analysis_results['backtest']])
                backtest_summary.to_csv(f'{self.symbol}_backtest_summary.csv', index=False)
                print(f"Backtest summary exported to {self.symbol}_backtest_summary.csv")

        except Exception as e:
            print(f"Export error: {e}")

    def run_complete_analysis(self, period: str = '1y', start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> bool:
        """Run complete analysis pipeline with flexible date options"""
        print(f"Starting complete analysis for {self.symbol}")
        print("=" * 60)

        # Step 1: Fetch Data
        if not self.fetch_data(period=period, start_date=start_date, end_date=end_date):
            return False

        # Step 2: Calculate Technical Indicators
        self.calculate_technical_indicators()

        # Step 3: Generate Trading Signals
        self.generate_signals()

        # Step 4: Run Backtest
        self.simple_backtest()

        # Step 5: Generate Forecast
        self.simple_forecast()

        # Step 6: Create Visualizations
        self.plot_comprehensive_analysis()
        self.plot_forecast()

        # Step 7: Export Results
        self.export_results()

        print("=" * 60)
        print("Complete analysis finished!")
        return True

# Multi-Stock Portfolio Analyzer
class PortfolioAnalyzer:
    """Simple Portfolio Analysis"""

    def __init__(self, symbols: List[str]):
        self.symbols = [s.upper() for s in symbols]
        self.data = {}
        self.returns = None
        self.portfolio_stats = {}

    def fetch_portfolio_data(self, period: str = '1y', start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> bool:
        """Fetch data for all stocks in portfolio with flexible date options"""
        print(f"Fetching data for portfolio: {', '.join(self.symbols)}")

        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)

                # Handle different date input methods
                if start_date and end_date:
                    data = ticker.history(start=start_date, end=end_date)
                elif start_date and not end_date:
                    data = ticker.history(start=start_date)
                else:
                    data = ticker.history(period=period)

                if not data.empty:
                    self.data[symbol] = data['Close']
                    print(f"{symbol}: {len(data)} days")
                else:
                    print(f"{symbol}: No data")
            except Exception as e:
                print(f"{symbol}: Error - {e}")

        # Create combined dataframe
        if self.data:
            portfolio_df = pd.DataFrame(self.data)
            self.returns = portfolio_df.pct_change().dropna()
            return True
        return False

    def calculate_portfolio_stats(self) -> None:
        """Calculate portfolio statistics"""
        if self.returns is None:
            return

        try:
            print("Calculating portfolio statistics...")

            # Individual stock statistics
            annual_returns = self.returns.mean() * 252
            annual_volatility = self.returns.std() * np.sqrt(252)
            sharpe_ratios = annual_returns / annual_volatility

            # Correlation matrix
            correlation_matrix = self.returns.corr()

            # Simple equal-weight portfolio
            equal_weights = np.array([1/len(self.symbols)] * len(self.symbols))
            portfolio_return = np.sum(annual_returns * equal_weights)
            portfolio_variance = np.dot(equal_weights.T, np.dot(correlation_matrix *
                                      (annual_volatility.values * annual_volatility.values[:, np.newaxis]),
                                      equal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            portfolio_sharpe = portfolio_return / portfolio_volatility

            # Risk-based weighting (inverse volatility)
            inv_volatility = 1 / annual_volatility
            risk_weights = inv_volatility / inv_volatility.sum()
            risk_portfolio_return = np.sum(annual_returns * risk_weights)
            risk_portfolio_variance = np.dot(risk_weights.T, np.dot(correlation_matrix *
                                           (annual_volatility.values * annual_volatility.values[:, np.newaxis]),
                                           risk_weights))
            risk_portfolio_volatility = np.sqrt(risk_portfolio_variance)
            risk_portfolio_sharpe = risk_portfolio_return / risk_portfolio_volatility

            self.portfolio_stats = {
                'individual_returns': annual_returns.to_dict(),
                'individual_volatility': annual_volatility.to_dict(),
                'individual_sharpe': sharpe_ratios.to_dict(),
                'correlation_matrix': correlation_matrix,
                'equal_weight': {
                    'weights': dict(zip(self.symbols, equal_weights)),
                    'return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe': portfolio_sharpe
                },
                'risk_weighted': {
                    'weights': risk_weights.to_dict(),
                    'return': risk_portfolio_return,
                    'volatility': risk_portfolio_volatility,
                    'sharpe': risk_portfolio_sharpe
                }
            }

            print("Portfolio statistics calculated")

        except Exception as e:
            print(f"Portfolio calculation error: {e}")

    def plot_portfolio_analysis(self) -> None:
        """Plot portfolio analysis"""
        if not self.portfolio_stats:
            return

        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Individual Stock Returns vs Risk',
                    'Correlation Heatmap',
                    'Equal Weight Allocation',
                    'Risk-Based Weight Allocation'
                ),
                specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                       [{"type": "pie"}, {"type": "pie"}]]
            )

            # 1. Risk-Return scatter
            returns = list(self.portfolio_stats['individual_returns'].values())
            volatilities = list(self.portfolio_stats['individual_volatility'].values())

            fig.add_trace(go.Scatter(
                x=volatilities, y=returns,
                mode='markers+text',
                text=self.symbols,
                textposition="middle right",
                marker=dict(size=10, color='blue'),
                name='Stocks'
            ), row=1, col=1)

            # 2. Correlation heatmap
            corr_matrix = self.portfolio_stats['correlation_matrix']
            fig.add_trace(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0
            ), row=1, col=2)

            # 3. Equal weight pie
            equal_weights = self.portfolio_stats['equal_weight']['weights']
            fig.add_trace(go.Pie(
                labels=list(equal_weights.keys()),
                values=[w*100 for w in equal_weights.values()],
                name="Equal Weight"
            ), row=2, col=1)

            # 4. Risk-based weight pie
            risk_weights = self.portfolio_stats['risk_weighted']['weights']
            fig.add_trace(go.Pie(
                labels=list(risk_weights.keys()),
                values=[w*100 for w in risk_weights.values()],
                name="Risk-Based"
            ), row=2, col=2)

            fig.update_layout(height=800, showlegend=False,
                            title="Portfolio Analysis Dashboard")
            fig.show()

            # Print summary
            print("\nPortfolio Analysis Summary:")
            print("-" * 50)

            equal_stats = self.portfolio_stats['equal_weight']
            risk_stats = self.portfolio_stats['risk_weighted']

            print(f"Equal Weight Portfolio:")
            print(f"  • Expected Return: {equal_stats['return']:.2%}")
            print(f"  • Volatility: {equal_stats['volatility']:.2%}")
            print(f"  • Sharpe Ratio: {equal_stats['sharpe']:.2f}")

            print(f"\nRisk-Based Portfolio:")
            print(f"  • Expected Return: {risk_stats['return']:.2%}")
            print(f"  • Volatility: {risk_stats['volatility']:.2%}")
            print(f"  • Sharpe Ratio: {risk_stats['sharpe']:.2f}")

            print(f"\nIndividual Stock Performance:")
            for symbol in self.symbols:
                ret = self.portfolio_stats['individual_returns'][symbol]
                vol = self.portfolio_stats['individual_volatility'][symbol]
                sharpe = self.portfolio_stats['individual_sharpe'][symbol]
                print(f"  • {symbol}: {ret:.2%} return, {vol:.2%} volatility, {sharpe:.2f} Sharpe")

        except Exception as e:
            print(f"Portfolio plotting error: {e}")

    def analyze_portfolio(self, period: str = '1y', start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> bool:
        """Complete portfolio analysis with flexible date options"""
        print(f"Starting portfolio analysis")
        print("=" * 60)

        if self.fetch_portfolio_data(period=period, start_date=start_date, end_date=end_date):
            self.calculate_portfolio_stats()
            self.plot_portfolio_analysis()
            print("=" * 60)
            print("Portfolio analysis completed!")
            return True
        else:
            print("Portfolio analysis failed!")
            return False

# Multimodal Analysis Functions
class MultimodalAnalyzer:
    """Handles multimodal data analysis including GDELT and YouTube data"""

    def __init__(self, youtube_api_key: str = ""):
        # Use hardcoded API key if no key is provided
        self.youtube_api_key = youtube_api_key or YOUTUBE_API_KEY
        if not self.youtube_api_key or self.youtube_api_key == "YOUR_YOUTUBE_API_KEY_HERE":
            print("Warning: YouTube API key not properly configured. YouTube functionality will be limited.")

    def download_gdelt_csv(self) -> Optional[pd.DataFrame]:
        """Download latest GDELT data"""
        try:
            print("Downloading GDELT data...")
            GDELT_URL = 'http://data.gdeltproject.org/gdeltv2/lastupdate.txt'
            response = requests.get(GDELT_URL, timeout=30)
            last_update = response.text.split()[-1]
            gdelt_csv_url = f'http://data.gdeltproject.org/gdeltv2/{last_update}'
            csv_response = requests.get(gdelt_csv_url, timeout=60)
            data = StringIO(csv_response.text)
            df = pd.read_csv(data, sep='\t', header=None, low_memory=False)
            print(f"Downloaded GDELT data with {len(df)} rows")
            return df
        except Exception as e:
            print(f"Error downloading GDELT data: {e}")
            return None

    def get_video_ids(self, channel_id: str, max_results: int = 3) -> List[str]:
        """Get video IDs from YouTube channel"""
        if not GOOGLE_API_AVAILABLE or not self.youtube_api_key or self.youtube_api_key == "YOUR_YOUTUBE_API_KEY_HERE":
            print("YouTube API not available or API key not properly configured")
            return []

        try:
            youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
            request = youtube.search().list(
                part="id",
                channelId=channel_id,
                maxResults=max_results,
                order="date",
                type="video"
            )
            response = request.execute()
            video_ids = [item['id']['videoId'] for item in response['items']]
            print(f"Found {len(video_ids)} videos")
            return video_ids
        except Exception as e:
            print(f"Error getting video IDs: {e}")
            return []

    def download_audio(self, video_url: str, output_path: str = "audio.mp3") -> Optional[str]:
        """Download audio from YouTube video"""
        if not YT_DLP_AVAILABLE:
            print("yt-dlp not available for audio download")
            return None

        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_path,
                'quiet': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            return output_path
        except Exception as e:
            print(f"Error downloading audio: {e}")
            return None

    def transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Transcribe audio file using Whisper"""
        if not WHISPER_AVAILABLE:
            print("Whisper not available for transcription")
            return None

        try:
            model = whisper.load_model("base")
            result = model.transcribe(audio_file)
            transcript = result['text']
            print("Audio transcribed successfully")
            return transcript
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None

    def fetch_youtube_transcripts(self, channel_id: str, max_videos: int = 3) -> pd.DataFrame:
        """Fetch and transcribe YouTube videos from a channel"""
        print(f"Fetching YouTube transcripts for channel: {channel_id}")
        transcripts = []
        video_ids = self.get_video_ids(channel_id, max_results=max_videos)

        for vid_id in video_ids:
            video_url = f"https://www.youtube.com/watch?v={vid_id}"
            audio_file = f"audio_{vid_id}.mp3"

            try:
                # Download audio
                downloaded_file = self.download_audio(video_url, audio_file)
                if downloaded_file:
                    # Transcribe audio
                    transcript = self.transcribe_audio(downloaded_file)
                    if transcript:
                        transcripts.append({
                            'video_id': vid_id,
                            'video_url': video_url,
                            'transcript': transcript
                        })

                # Clean up audio file
                if os.path.exists(audio_file):
                    os.remove(audio_file)

            except Exception as e:
                print(f"Error processing video {vid_id}: {e}")
                continue

        print(f"Successfully transcribed {len(transcripts)} videos")
        return pd.DataFrame(transcripts)

    def run_multimodal_analysis(self, stock_channel_id: str, output_dir: str = ".") -> Dict[str, Any]:
        """Run complete multimodal analysis"""
        results = {}

        # Download GDELT data
        gdelt_df = self.download_gdelt_csv()
        if gdelt_df is not None:
            gdelt_path = os.path.join(output_dir, "gdelt_data.csv")
            gdelt_df.to_csv(gdelt_path, index=False)
            results['gdelt_data'] = gdelt_path
            print(f"GDELT data saved to {gdelt_path}")

        # Fetch YouTube transcripts
        transcripts_df = self.fetch_youtube_transcripts(stock_channel_id, max_videos=3)
        if not transcripts_df.empty:
            transcripts_path = os.path.join(output_dir, "youtube_transcripts.csv")
            transcripts_df.to_csv(transcripts_path, index=False)
            results['youtube_transcripts'] = transcripts_path
            print(f"YouTube transcripts saved to {transcripts_path}")

        return results

# Date selection utility functions
def get_date_presets() -> Dict[str, Dict[str, Any]]:
    """Get common date presets for easy selection"""
    today = datetime.now()
    presets = {
        '1_week': {'period': '5d', 'description': 'Last 1 week'},
        '1_month': {'period': '1mo', 'description': 'Last 1 month'},
        '3_months': {'period': '3mo', 'description': 'Last 3 months'},
        '6_months': {'period': '6mo', 'description': 'Last 6 months'},
        '1_year': {'period': '1y', 'description': 'Last 1 year'},
        '2_years': {'period': '2y', 'description': 'Last 2 years'},
        '5_years': {'period': '5y', 'description': 'Last 5 years'},
        '10_years': {'period': '10y', 'description': 'Last 10 years'},
        'max': {'period': 'max', 'description': 'All available data'},

        # Specific year ranges
        'ytd': {
            'start_date': f'{today.year}-01-01',
            'description': f'Year to date ({today.year})'
        },
        'last_year': {
            'start_date': f'{today.year-1}-01-01',
            'end_date': f'{today.year-1}-12-31',
            'description': f'Full year {today.year-1}'
        },
        'covid_period': {
            'start_date': '2020-01-01',
            'end_date': '2022-12-31',
            'description': 'COVID period (2020-2022)'
        },
        'pre_covid': {
            'start_date': '2018-01-01',
            'end_date': '2019-12-31',
            'description': 'Pre-COVID (2018-2019)'
        },
        'post_covid': {
            'start_date': '2023-01-01',
            'description': 'Post-COVID recovery (2023-present)'
        }
    }
    return presets

def display_date_options() -> None:
    """Display available date preset options"""
    presets = get_date_presets()

    print("Available Date Presets:")
    print("=" * 50)

    for key, value in presets.items():
        print(f"'{key}': {value['description']}")

    print("\nUsage Examples:")
    print("# Use preset with period")
    print("analyzer = FinancialAnalyzer('AAPL')")
    print("analyzer.run_complete_analysis(period='1y')")
    print("\n# Custom date range")
    print("analyzer.run_complete_analysis(start_date='2020-01-01', end_date='2023-12-31')")
    print("\n# From specific date to present")
    print("analyzer.run_complete_analysis(start_date='2022-01-01')")

def run_analysis_with_preset(symbol: str, preset_key: str) -> bool:
    """Run analysis with a date preset"""
    presets = get_date_presets()

    if preset_key not in presets:
        print(f"Unknown preset: {preset_key}")
        print("Available presets:", list(presets.keys()))
        return False

    preset = presets[preset_key]
    print(f"Using preset: {preset['description']}")

    analyzer = FinancialAnalyzer(symbol)

    # Extract parameters
    period = preset.get('period')
    start_date = preset.get('start_date')
    end_date = preset.get('end_date')

    return analyzer.run_complete_analysis(period=period, start_date=start_date, end_date=end_date)

def run_portfolio_analysis_with_preset(symbols: List[str], preset_key: str) -> bool:
    """Run portfolio analysis with a date preset"""
    presets = get_date_presets()

    if preset_key not in presets:
        print(f"Unknown preset: {preset_key}")
        print("Available presets:", list(presets.keys()))
        return False

    preset = presets[preset_key]
    print(f"Using preset: {preset['description']}")

    portfolio = PortfolioAnalyzer(symbols)

    # Extract parameters
    period = preset.get('period')
    start_date = preset.get('start_date')
    end_date = preset.get('end_date')

    return portfolio.analyze_portfolio(period=period, start_date=start_date, end_date=end_date)

# Interactive analysis functions
def interactive_single_stock_analysis() -> None:
    """Interactive function for single stock analysis"""
    print("Interactive Single Stock Analysis")
    print("=" * 40)

    # Get stock symbol
    symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
    if not symbol:
        symbol = "AAPL"
        print(f"Using default: {symbol}")

    # Display date options
    print("\nDate Selection Options:")
    print("1. Use preset (recommended)")
    print("2. Custom date range (YYYY-MM-DD format)")
    print("3. From specific date to present")
    print("4. Use period (e.g., 1y, 6mo, 2y)")

    choice = input("\nSelect option (1/2/3/4): ").strip()

    analyzer = FinancialAnalyzer(symbol)

    if choice == '1':
        display_date_options()
        preset = input("\nEnter preset key (e.g., '1_year', 'covid_period'): ").strip()
        run_analysis_with_preset(symbol, preset)

    elif choice == '2':
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD): ").strip()
        analyzer.run_complete_analysis(start_date=start_date, end_date=end_date)

    elif choice == '3':
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        analyzer.run_complete_analysis(start_date=start_date)

    elif choice == '4':
        period = input("Enter period (e.g., 1y, 6mo, 2y, max): ").strip()
        analyzer.run_complete_analysis(period=period)

    else:
        print("Using default: 1 year period")
        analyzer.run_complete_analysis(period='1y')

def interactive_portfolio_analysis() -> None:
    """Interactive function for portfolio analysis"""
    print("Interactive Portfolio Analysis")
    print("=" * 40)

    # Get stock symbols
    symbols_input = input("Enter stock symbols separated by commas (e.g., AAPL,MSFT,GOOGL): ").strip().upper()
    if not symbols_input:
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        print(f"Using default portfolio: {', '.join(symbols)}")
    else:
        symbols = [s.strip() for s in symbols_input.split(',')]

    # Display date options
    print("\nDate Selection Options:")
    print("1. Use preset (recommended)")
    print("2. Custom date range (YYYY-MM-DD format)")
    print("3. From specific date to present")
    print("4. Use period (e.g., 1y, 6mo, 2y)")

    choice = input("\nSelect option (1/2/3/4): ").strip()

    portfolio = PortfolioAnalyzer(symbols)

    if choice == '1':
        display_date_options()
        preset = input("\nEnter preset key (e.g., '1_year', 'covid_period'): ").strip()
        run_portfolio_analysis_with_preset(symbols, preset)

    elif choice == '2':
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD): ").strip()
        portfolio.analyze_portfolio(start_date=start_date, end_date=end_date)

    elif choice == '3':
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        portfolio.analyze_portfolio(start_date=start_date)

    elif choice == '4':
        period = input("Enter period (e.g., 1y, 6mo, 2y, max): ").strip()
        portfolio.analyze_portfolio(period=period)

    else:
        print("Using default: 1 year period")
        portfolio.analyze_portfolio(period='1y')

def interactive_multimodal_analysis() -> None:
    """Interactive function for multimodal analysis"""
    print("Interactive Multimodal Analysis")
    print("=" * 40)

    if not (GOOGLE_API_AVAILABLE and YT_DLP_AVAILABLE and WHISPER_AVAILABLE):
        print("Warning: Some required libraries are not available.")
        print(f"Google API: {'Available' if GOOGLE_API_AVAILABLE else 'Not Available'}")
        print(f"yt-dlp: {'Available' if YT_DLP_AVAILABLE else 'Not Available'}")
        print(f"Whisper: {'Available' if WHISPER_AVAILABLE else 'Not Available'}")
        print("\nYou may proceed, but some features will be disabled.")

    # Check if API key is configured
    if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
        print("Warning: YouTube API key is not configured in the hardcoded section.")
        api_key = input("Enter YouTube API key (or press Enter to use hardcoded key): ").strip()
    else:
        print("Using hardcoded YouTube API key.")
        api_key = ""

    # Get channel ID
    channel_id = input("Enter YouTube channel ID for financial content: ").strip()
    if not channel_id:
        print("No channel ID provided. Skipping YouTube analysis.")
        return

    # Get output directory
    output_dir = input("Enter output directory (default: current directory): ").strip()
    if not output_dir:
        output_dir = "."

    # Run analysis
    analyzer = MultimodalAnalyzer(api_key)
    results = analyzer.run_multimodal_analysis(channel_id, output_dir)

    print("\nMultimodal Analysis Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

def main_demo() -> None:
    """Main demonstration function"""
    print("Enhanced Financial Analysis Tool - Demo")
    print("=" * 60)

    # Display available date options
    display_date_options()

    print("\n" + "=" * 60)
    print("Running Demo Analyses...")

    # Demo 1: Single stock analysis with preset
    print("\nDemo 1: AAPL analysis for last 1 year")
    run_analysis_with_preset("AAPL", "1_year")

    # Demo 2: Portfolio analysis with preset
    print("\nDemo 2: Tech portfolio analysis for COVID period")
    tech_portfolio = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    run_portfolio_analysis_with_preset(tech_portfolio, "covid_period")

    print("\nDemo Complete!")

def main_menu() -> None:
    """Main menu for interactive use"""
    print("Enhanced Financial Analysis Tool")
    print("=" * 50)

    while True:
        print("\nMain Menu:")
        print("1. Single Stock Analysis")
        print("2. Portfolio Analysis")
        print("3. Multimodal Analysis (GDELT + YouTube)")
        print("4. Show Date Presets")
        print("5. Run Demo")
        print("6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == '1':
            interactive_single_stock_analysis()
        elif choice == '2':
            interactive_portfolio_analysis()
        elif choice == '3':
            interactive_multimodal_analysis()
        elif choice == '4':
            display_date_options()
        elif choice == '5':
            main_demo()
        elif choice == '6':
            print("Thank you for using the Financial Analysis Tool!")
            break
        else:
            print("Invalid choice. Please select 1-6.")

# Quick start examples
def quick_examples() -> None:
    """Quick examples for immediate use"""
    print("Quick Start Examples")
    print("=" * 40)

    print("1. Single Stock Analysis:")
    print("   analyzer = FinancialAnalyzer('AAPL')")
    print("   analyzer.run_complete_analysis(period='1y')")

    print("\n2. Custom Date Range:")
    print("   analyzer.run_complete_analysis(start_date='2020-01-01', end_date='2022-12-31')")

    print("\n3. Portfolio Analysis:")
    print("   portfolio = PortfolioAnalyzer(['AAPL', 'MSFT', 'GOOGL'])")
    print("   portfolio.analyze_portfolio(period='1y')")

    print("\n4. Using Presets:")
    print("   run_analysis_with_preset('TSLA', 'covid_period')")
    print("   run_portfolio_analysis_with_preset(['AAPL', 'MSFT'], 'ytd')")

    print("\n5. Multimodal Analysis:")
    print("   analyzer = MultimodalAnalyzer()  # Uses hardcoded API key")
    print("   analyzer.run_multimodal_analysis('CHANNEL_ID')")

# Legacy compatibility functions (simplified versions from the original duplicate code)
def download_gdelt_csv() -> Optional[pd.DataFrame]:
    """Legacy function - use MultimodalAnalyzer.download_gdelt_csv() instead"""
    analyzer = MultimodalAnalyzer()
    return analyzer.download_gdelt_csv()

def run_multimodal_analysis(youtube_api_key: str, stock_channel_id: str) -> Dict[str, Any]:
    """Legacy function - use MultimodalAnalyzer.run_multimodal_analysis() instead"""
    analyzer = MultimodalAnalyzer(youtube_api_key)
    return analyzer.run_multimodal_analysis(stock_channel_id)

# API Key Configuration Helper
def configure_api_keys():
    """Helper function to display API key configuration instructions"""
    print("API Key Configuration Instructions")
    print("=" * 50)
    print("To use the multimodal features, you need to configure your API keys in the code.")
    print("\nRequired API Keys:")
    print("1. YouTube Data API v3 Key")
    print("   - Go to: https://console.cloud.google.com/")
    print("   - Enable YouTube Data API v3")
    print("   - Create credentials (API Key)")
    print("   - Replace 'YOUR_YOUTUBE_API_KEY_HERE' with your actual key")
    print("\nOptional API Keys (for future enhancements):")
    print("- Alpha Vantage API Key")
    print("- News API Key")
    print("- Financial Modeling Prep API Key")
    print("\nExample configuration:")
    print('YOUTUBE_API_KEY = "AIzaSyA1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7"')

    # Display current configuration status
    print(f"\nCurrent YouTube API Key Status:")
    if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
        print("❌ Not configured - please add your API key")
    elif YOUTUBE_API_KEY:
        print(f"✅ Configured - Key starts with: {YOUTUBE_API_KEY[:10]}...")
    else:
        print("❌ Empty - please add your API key")

# Example usage and main entry point
if __name__ == "__main__":
    print("Enhanced Financial Analysis Tool - With Hardcoded API Keys")
    print("=" * 60)

    # Display API key configuration status
    configure_api_keys()
    print("\n" + "=" * 60)

    # Uncomment one of the following to run:

    # 1. Run interactive menu
    main_menu()

    # 2. Run quick demo
    # main_demo()

    # 3. Show quick examples
    # quick_examples()

    # 4. Run specific analysis
    # analyzer = FinancialAnalyzer("AAPL")
    # analyzer.run_complete_analysis(period='1y')

    # 5. Run multimodal analysis (uses hardcoded API key)
    # multimodal = MultimodalAnalyzer()
    # results = multimodal.run_multimodal_analysis("YOUR_CHANNEL_ID")
    # print("Multimodal results:", results)
